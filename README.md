# DocRAG · 私有文档问答

- 作者：朱春潮
- 更新日期：2026-10-19
- 项目类型：个人学习 / 求职作品原型

## 项目简介

基于检索增强生成（RAG）的私有文档问答 Demo：上传 PDF → 解析分块 → 向量入库 → **混合检索（向量 + BM25 + RRF）** → 可选 **BGE Rerank** → 大模型依据上下文回答，并展示引用来源（文件名 + 页码 + 原文摘要）。

当前重点是把「检索 → 重排 → 生成 → 溯源」主链路跑通，并提供 Gradio 演示与 **20 题评测脚本**，便于本地验收和面试讲解。

## 已实现功能

- PDF 加载与递归字符分块（默认 chunk_size=500，overlap=80）
- 中文向量化：`BAAI/bge-small-zh-v1.5`（本地 CPU）
- 向量库：Milvus Lite（本地文件 `milvus_demo.db`）
- **混合检索**：向量召回 + BM25（jieba 分词）+ RRF 融合；入库同步写 `data/chunks.jsonl`
- **Rerank**：`BAAI/bge-reranker-base`（CrossEncoder，CPU；HF 镜像 `hf-mirror.com`）
- RAG 问答：DeepSeek Chat（OpenAI 兼容接口）
- **引用溯源**：返回来源编号、文件名、页码、片段摘要；回答中可用 `[来源n]` 标注
- **Gradio 界面**：上传入库、提问、混合检索/Rerank 开关、查看回答与来源
- FastAPI：`/ingest`、`/query`（支持 `use_hybrid` / `use_rerank`）
- **评测**：`eval/eval_set.json`（20 题）+ `eval/run_eval.py`（Hit@K / 关键词覆盖 / 时延）

## 暂未实现（计划中）

- LangGraph 工具调用 Agent
- 线上部署 / 鉴权 / 多用户会话

> 简历与面试请只写已实现部分；**具体 Hit@K / 时延数字以本地运行 `eval/run_eval.py` 生成的 `eval/results.json` 为准**，请勿手写未跑通的指标。

## 技术栈

| 层级 | 选型 |
|------|------|
| 语言 | Python 3 |
| 编排 | LangChain |
| Embedding | HuggingFace `bge-small-zh-v1.5` |
| 稀疏检索 | jieba + BM25Okapi |
| 融合 | Reciprocal Rank Fusion (RRF, k=60) |
| Rerank | `bge-reranker-base` |
| 向量库 | Milvus Lite（`langchain-milvus`） |
| LLM | DeepSeek Chat |
| 界面 | Gradio |
| API（可选） | FastAPI + Uvicorn |
| 配置 | `.env` + `python-dotenv` |

## 目录结构

```text
DocRAG/
├── README.md
├── requirements.txt
├── .env                 # 本地密钥（勿提交）
├── .gitignore
├── app_gradio.py        # Gradio：入库 + 问答 + 检索开关
├── main.py              # FastAPI：/ingest、/query
├── rag_pipeline.py      # 检索、Prompt、调用模型、组装 sources
├── retrieval.py         # 混合检索 + Rerank
├── vector_store.py      # Embedding + Milvus Lite + chunks 落盘
├── pdf_split_demo.py    # PDF 加载与分块
├── data/chunks.jsonl    # 入库后的切块（BM25 语料，自动生成）
├── eval/
│   ├── eval_set.json    # 20 题评测集
│   ├── prepare_index.py # 重建索引（必须显式传 --pdf / --docs）
│   ├── run_eval.py      # 三种模式对比评测
│   └── results.json     # 评测输出（运行后生成）
├── 本地文档/            # PDF 存放目录
└── milvus_demo.db       # 本地向量库数据
```

## 环境准备

1. 进入项目目录，激活虚拟环境：

```bash
cd DocRAG
.\.venv\Scripts\activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置 `.env`（不要把 Key 写进代码）：

```env
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

4. 国内拉取 HuggingFace 模型时，代码已设置：

```text
HF_ENDPOINT=https://hf-mirror.com
```

首次会下载 `bge-small-zh-v1.5` 与 `bge-reranker-base`；本机已有缓存时可离线加载。

## 启动方式

### 方式一：Gradio 界面（推荐演示）

```bash
.\.venv\Scripts\python.exe app_gradio.py
```

浏览器打开：<http://127.0.0.1:7860>

1. 上传 PDF → 点击「入库」（会写向量库 + `data/chunks.jsonl`）
2. 确认「混合检索」「BGE Rerank」开关（默认开启）
3. 输入问题 → 「提问」
4. 右侧查看回答与「来源 n：文件名 第 x 页」片段

### 方式二：FastAPI

```bash
.\.venv\Scripts\python.exe main.py
```

- `POST /ingest`：`{"pdf_path": "本地文档/政治.pdf"}`
- `POST /query`：`{"question": "你的问题", "top_k": 5, "use_hybrid": true, "use_rerank": true}`

## 混合检索与 Rerank 说明

1. **向量检索**：Milvus + bge-small-zh-v1.5，先取 `fetch_k`（默认 20）
2. **BM25**：对 `data/chunks.jsonl` 用 jieba 分词建索引，取 `fetch_k`
3. **RRF**：对两路排序列表融合去重（k=60）
4. **Rerank**：`CrossEncoder(BAAI/bge-reranker-base)` 对融合候选重打分，截断到 `top_k`（默认 5）
5. Gradio / FastAPI 可将 `use_hybrid` / `use_rerank` 设为 `false`，退回纯向量（或混合但不重排）

### Reranker 加载策略（网络受限环境）

`retrieval.get_reranker()` 会**优先加载本地权重**，顺序为：

1. 环境变量 `RERANKER_MODEL_PATH` 指向的目录
2. `models/bge-reranker-base/`
3. `models/models/BAAI--bge-reranker-base/snapshots/master/`（ModelScope `cache_dir` 默认落点）

以上都没有时，**默认不联网下载**，直接跳过重排（评测会如实标记 `rerank_applied=false`），避免卡在下载重试上。
确需联网拉取时显式开启：

```powershell
$env:RERANKER_ALLOW_DOWNLOAD='1'   # 走 HF_ENDPOINT（默认 hf-mirror.com）
```

本机可用 ModelScope 预下载权重：

```powershell
.\.venv\Scripts\python.exe -c "from modelscope import snapshot_download; print(snapshot_download('BAAI/bge-reranker-base', cache_dir='models'))"
```


## 评测怎么跑

**必须显式传入** `--pdf` 或 `--docs`，没有任何默认文档路径；未传会报错并提示用法。

```bash
# 单个 PDF
.\.venv\Scripts\python.exe eval\prepare_index.py --pdf 本地文档\你的文件.pdf

# 多个 PDF
.\.venv\Scripts\python.exe eval\prepare_index.py --pdf doc1.pdf --pdf doc2.pdf

# 整个目录下全部 PDF
.\.venv\Scripts\python.exe eval\prepare_index.py --docs 你的文档目录

# 绝对路径
.\.venv\Scripts\python.exe eval\prepare_index.py --pdf D:\data\my_manual.pdf
```

只评检索（不调 LLM，推荐先跑这个）。若尚无索引，须带上同样的 `--pdf`/`--docs` 才会自动建库：

```bash
.\.venv\Scripts\python.exe eval\run_eval.py --no-llm --pdf 本地文档\你的文件.pdf

# 换语料并强制重建索引后再评
.\.venv\Scripts\python.exe eval\run_eval.py --no-llm --rebuild-index --docs 你的文档目录
```

全链路（含 DeepSeek 答案关键词命中）：

```bash
.\.venv\Scripts\python.exe eval\run_eval.py --pdf 本地文档\你的文件.pdf
```

> 注意：`eval/eval_set.json` 当前 20 题按「近现代史政治考点」编写，仅当你入库的是对应笔记时 Hit@K 才有意义；换了无关语料请同步改评测集。

## 空库与拒答

- 未入库（无 `data/chunks.jsonl`）时提问：直接返回「知识库为空…请先入库」，不编造。
- 已入库但检索无命中：返回「不知道。知识库中不存在该内容」。
- Prompt 亦要求模型仅依据上下文作答，否则明确说不知道。

输出：

- 控制台 Markdown 对比表：`vector` / `hybrid` / `hybrid+rerank`
- 详细 JSON：`eval/results.json`

指标含义：

- **Hit@K**：Top-K 检索上下文是否命中任一 `must_keywords`
- **Keyword Coverage**：命中关键词占该题关键词比例的平均
- **Avg / P95 Latency**：单题耗时（秒）
- **Answer Hit**（仅全链路）：模型答案是否含任一关键词

> 请以你机器上 `results.json` 的真实数字写简历；本 README 不预填指标。

## 引用溯源说明

检索到的每个文档块会整理为：

- `index`：来源编号
- `label`：文件名 + 页码（PyPDFLoader 页码从 0 起，展示时 +1）
- `snippet`：原文前约 300 字

Prompt 要求模型严格依据上下文作答；上下文缺失时回答「不知道」，减少胡编。

## 安全提醒

- API Key 仅放在 `.env`，已加入 `.gitignore`
- 若 Key 曾出现在代码或聊天记录中，请到 DeepSeek 控制台轮换
- 不要把 `.env`、`milvus_demo.db`、`data/chunks.jsonl` 推到公开仓库

## 后续迭代建议

1. 用评测结果指导分块参数 / fetch_k / top_k
2. 用 LangGraph 扩展工具调用 Agent（二期）
3. 补架构图与演示录屏，便于投递时附链接

## 许可证

个人学习项目，仅供学习与求职展示。
