# OneContext Python SDK 底层原理分析

> 版本: 3.3.13 | 包名: `onecontext` (PyPI)

## 1. 项目概述

OneContext 是一个 **RAG (Retrieval-Augmented Generation) 平台的 Python SDK**。它帮助开发者将文档上传到云端，自动完成分块 (chunking)、向量嵌入 (embedding)、索引建立，然后通过**混合搜索 (Hybrid Search)** 检索相关内容。

核心定位：**客户端 SDK** — 所有重计算（嵌入生成、向量索引、全文索引、OCR 等）都在 OneContext 服务端完成，SDK 只负责 API 交互和文件传输。

## 2. 源码文件结构

```
onecontext/
├── __init__.py        # 入口，导出 OneContext, Chunk, File
├── __about__.py       # 版本号 (__version__ = "3.3.13")
├── client.py          # HTTP 客户端 (ApiClient) + URL 路由 (URLS)
├── context.py         # 核心业务逻辑 (Context 类)：上传、搜索、元数据管理
├── models.py          # 数据模型：Chunk, File, PydanticV2BaseModel protocol
├── utils.py           # 工具函数：batch_by_size
└── scrathc.py         # 空文件 (可能是开发草稿)
```

## 3. 架构分层

```
┌─────────────────────────────────────────────┐
│              用户代码                         │
│  oc = OneContext(api_key="...")              │
│  ctx = oc.create_context("my_kb")           │
│  ctx.upload_files([...])                    │
│  chunks = ctx.search("query")              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           OneContext (main.py)               │
│  - 管理 Context 的 CRUD                     │
│  - 持有 ApiClient 和 URLS 实例              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Context (context.py)               │
│  - 文件上传 (presigned URL 方式)             │
│  - 混合搜索 (语义 + 全文)                    │
│  - 结构化输出提取                            │
│  - Chunk/File 管理                           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           ApiClient (client.py)              │
│  - requests.Session 封装                     │
│  - API-KEY 认证                              │
│  - 统一错误处理                              │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          OneContext 云端 API
      (https://app.onecontext.ai/api/v6/)
```

## 4. 核心原理详解

### 4.1 HTTP 客户端层 (`client.py`)

```python
class ApiClient:
    def __init__(self, api_key, extra_headers=None):
        self.session = requests.Session()
        self.session.headers.update({"API-KEY": api_key})
```

**要点：**
- 基于 `requests.Session` 实现连接复用
- 认证方式：自定义 `API-KEY` 请求头（不是标准 Bearer Token）
- 支持透传 `OPENAI-API-KEY` 和 `ANTHROPIC-API-KEY` 头，供服务端调用 LLM 时使用
- 统一的 `_handle_response` 方法：解析 JSON、检查 HTTP 状态码、提取服务端错误信息

**URL 路由** (`URLS` dataclass) 用 `urljoin` 拼接所有 API 端点，基础 URL 默认为 `https://app.onecontext.ai/api/v6/`。

### 4.2 文件上传机制 (`context.py` 的 upload_files)

这是最复杂的流程，采用**三阶段上传**：

```
阶段 1: 获取预签名 URL
   客户端 → POST /context/file/presigned-upload-url/
         ← 返回 {presignedUrl, fileId, gcsUri, fileType}

阶段 2: 直传云存储
   客户端 → PUT presignedUrl (直接上传文件到 GCS)
         ← 200 OK

阶段 3: 通知服务端处理
   客户端 → POST /context/file/process-uploaded/
         ← 服务端开始：分块 → 嵌入 → 索引
```

**关键设计细节：**

1. **Presigned URL 模式**：客户端不经过 OneContext 服务器传输文件内容，而是直接上传到 Google Cloud Storage (GCS)。这避免了服务端成为传输瓶颈。

2. **并发上传**：使用 `ThreadPoolExecutor(max_workers=10)` 并行上传多个文件，配合 `tqdm` 进度条。

3. **批量处理通知**：`batch_by_size(upload_files_spec, 3)` 按 3MB 为一批发送处理通知，避免单次请求 payload 过大。

4. **重试机制**：上传到 GCS 时使用 `urllib3.Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])`。

5. **支持的文件类型**：PDF、DOCX、PPTX、图片 (PNG/JPG/TIFF/BMP/HEIC)、HTML、Markdown、TXT、XML、EML 等约 20 种格式。

6. **OCR 支持**：`force_ocr=True` 参数可强制对图片/PDF 使用 OCR 提取文本。

7. **分块大小控制**：`max_chunk_size` 参数控制服务端分块的最大字数（默认 200 词）。

### 4.3 混合搜索 (`context.py` 的 search)

```python
def search(self, query, top_k=10, semantic_weight=0.5, full_text_weight=0.5, rrf_k=60, ...):
```

**混合搜索 (Hybrid Search)** 是 OneContext 的核心检索策略，结合两种搜索方式：

| 维度 | 语义搜索 (Semantic) | 全文搜索 (Full-text) |
|------|---------------------|---------------------|
| 原理 | 将 query 转为向量，与文档 chunk 向量做相似度匹配 | 基于关键词的 BM25/倒排索引匹配 |
| 擅长 | 理解语义相似性 ("汽车" ≈ "车辆") | 精确匹配术语、专有名词 |
| 权重 | `semantic_weight` (默认 0.5) | `full_text_weight` (默认 0.5) |

**融合算法 — Reciprocal Rank Fusion (RRF):**

```
RRF_score(d) = Σ [ weight / (rrf_k + rank_i(d)) ]
```

其中 `rrf_k` 默认 60，用于平衡头部和尾部结果的得分差距。两种搜索分别产生排序，再通过 RRF 公式加权融合为最终排序。

**返回的 Chunk 对象包含三个分数：**
- `semantic_score` — 语义搜索得分
- `fulltext_score` — 全文搜索得分
- `combined_score` — RRF 融合后的最终得分

### 4.4 结构化输出提取 (`extract_from_search` / `extract_from_chunks`)

```python
def extract_from_search(self, query, schema, extraction_prompt, model="gpt-4o-mini", ...):
```

这是 RAG + 结构化输出的完整管线：

```
用户 query → 混合搜索 → 取 top_k 个 chunks → LLM 提取 → 返回结构化 JSON
```

**要点：**
- 支持传入 Pydantic V2 BaseModel 或原始 JSON Schema 作为输出格式
- 通过 `PydanticV2BaseModel` Protocol 类做鸭子类型检查（`runtime_checkable`）
- 可选模型：`gpt-4o-mini`, `gpt-4o`, `claude-35`（在服务端执行）
- 提取结果作为 `(output_dict, chunks_list)` 元组返回，可追溯来源

### 4.5 数据模型 (`models.py`)

**Chunk** — 向量数据库中的最小检索单元：
```python
@dataclass
class Chunk:
    id: str               # chunk 唯一 ID
    content: str           # 文本内容
    file_id: str           # 所属文件
    context_id: str        # 所属 context
    embedding: List[float] # 向量嵌入 (可选返回)
    metadata_json: Dict    # 用户自定义元数据
    semantic_score: float  # 语义搜索得分
    fulltext_score: float  # 全文搜索得分
    combined_score: float  # 融合得分
```

**File** — 上传的文件记录：
```python
@dataclass
class File:
    id: str
    name: str
    status: str            # 处理状态 (processing/ready/failed 等)
    path: str              # GCS 存储路径
    metadata_json: Dict
    download_url: str      # 预签名下载 URL (可选)
```

### 4.6 元数据系统

SDK 提供完整的元数据管理能力：

- **上传时附加元数据**：每个文件可携带 JSON 格式的 metadata
- **元数据过滤搜索**：支持类 MongoDB 查询语法，如 `{'author': {'$eq': 'Jane Doe'}}`
- **更新/清除元数据**：`update_file_meta()` / `clear_file_meta()`
- **字典扁平化**：`flatten_metadata=True` 将嵌套 dict 展平为 `key_subkey` 格式，使嵌套字段可查询
- **Key 校验**：禁止 `.`, `-`, `\` 字符出现在 metadata key 中

### 4.7 工具函数 (`utils.py`)

```python
def batch_by_size(items, max_size_mb):
    size_limit = max_size_mb << 20  # MB 转字节 (位运算)
```

按 JSON 序列化后的字节大小将列表分批，避免单次 HTTP 请求 payload 过大。使用生成器模式 (`yield`) 实现惰性求值。

## 5. 依赖关系

```
onecontext
├── requests          # HTTP 客户端
├── tqdm              # 进度条
└── typing_extensions # Literal, Protocol, get_args (向后兼容)
```

极度轻量级 — 仅 3 个依赖。

## 6. 整体数据流

```
               ┌──────────────────────────────────────┐
               │          OneContext 云端               │
               │                                       │
    upload     │  ┌─────────┐   ┌──────────────────┐  │
   ──────────► │  │  GCS    │──►│ Document Pipeline │  │
   (presigned) │  │ Storage │   │  - 文档解析       │  │
               │  └─────────┘   │  - 文本分块       │  │
               │                │  - 向量嵌入       │  │
               │                │  - 索引构建       │  │
               │                └────────┬─────────┘  │
               │                         ▼             │
               │              ┌──────────────────┐    │
    search     │              │  Vector + FTS     │    │
   ──────────► │              │  Index            │    │ ──► chunks
               │              │  (混合搜索引擎)    │    │
               │              └──────────────────┘    │
               │                         │             │
               │              ┌──────────▼─────────┐  │
   extract     │              │  LLM (GPT/Claude)  │  │
   ──────────► │              │  结构化输出提取      │  │ ──► JSON
               │              └────────────────────┘  │
               └──────────────────────────────────────┘
```

## 7. 设计特点总结

| 特点 | 说明 |
|------|------|
| **客户端轻量** | SDK 仅 ~300 行代码，所有重计算在服务端 |
| **Presigned URL 上传** | 文件直传 GCS，绕过应用服务器瓶颈 |
| **混合搜索** | 语义 + 全文，RRF 融合，可调权重 |
| **结构化输出** | RAG 搜索 + LLM 提取，支持 Pydantic Schema |
| **并发上传** | ThreadPoolExecutor + 重试 + 批量处理 |
| **元数据驱动** | 完善的元数据 CRUD + 过滤查询 |
| **多格式支持** | ~20 种文件格式，含图片 OCR |
| **多 LLM 支持** | 可透传 OpenAI / Anthropic API Key |
