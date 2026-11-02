# DocRAG · 中文 PDF 私有文档问答

一个可以在本地跑起来的 RAG 小项目：上传 PDF → 检索相关段落 → 大模型基于段落回答 → 标出文件名和页码。

作者：朱春潮

## 做了什么

- PDF 按页解析，递归分块（chunk_size=500，overlap=80）
- 中文 embedding：`BAAI/bge-small-zh-v1.5`，Milvus Lite 本地向量库
- 混合检索：向量召回 + jieba 分词 BM25，用 RRF（k=60）融合两路排名
- 可选 BGE Rerank：`BAAI/bge-reranker-base` 对候选精排后截断到 top_k
- DeepSeek 生成回答，Prompt 约束只依据上下文；检索不到依据时直接拒答
- 回答附带引用来源（文件名 + 页码 + 原文片段）
- 20 题评测脚本，对比 vector / hybrid / hybrid+rerank 的 Hit@5、关键词覆盖和时延

## 目录

```text
DocRAG/
├── app_gradio.py        # Gradio：上传入库 + 提问 + 检索开关
├── main.py              # FastAPI：/ingest、/query
├── rag_pipeline.py      # Prompt、调用模型、组装 sources
├── retrieval.py         # 混合检索 + Rerank
├── vector_store.py      # Embedding + Milvus Lite + chunks 落盘
├── pdf_split_demo.py    # PDF 加载与分块
├── eval/
│   ├── eval_set.json    # 20 题评测集
│   ├── prepare_index.py # 重建索引
│   └── run_eval.py      # 三种模式对比
├── index.html           # GitHub Pages 上的静态演示页
└── requirements.txt
```

## 运行

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

在项目根目录建 `.env`：

```env
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

Gradio 界面：

```bash
.\.venv\Scripts\python.exe app_gradio.py
```

打开 http://127.0.0.1:7860，上传 PDF 后点「入库」，再提问。

FastAPI：

```bash
.\.venv\Scripts\python.exe main.py
```

- `POST /ingest`：`{"pdf_path": "本地文档/你的文件.pdf"}`
- `POST /query`：`{"question": "你的问题", "top_k": 5, "use_hybrid": true, "use_rerank": true}`

## 评测

评测脚本要求显式传入文档，没有默认路径：

```bash
# 建索引
.\.venv\Scripts\python.exe eval\prepare_index.py --pdf 本地文档\你的文件.pdf

# 只评检索，不调 LLM
.\.venv\Scripts\python.exe eval\run_eval.py --no-llm --pdf 本地文档\你的文件.pdf

# 全链路（含答案关键词命中）
.\.venv\Scripts\python.exe eval\run_eval.py --pdf 本地文档\你的文件.pdf
```

输出控制台对比表和 `eval/results.json`，指标是 Hit@K、Keyword Coverage、Avg / P95 Latency，全链路另有 Answer Hit。

## 已知限制

- Reranker 权重没有放进仓库。本地找不到权重时会跳过重排，评测里会如实标出 `rerank_applied=false`；需要联网下载时设置 `RERANKER_ALLOW_DOWNLOAD=1`。
- 扫描版 PDF 没有文本层，解析不到内容。
- `eval/eval_set.json` 的 20 题是按中共党史材料的 PDF 写的，换语料时要同步改评测集，否则 Hit@K 没有参考意义。

## 关于静态页

`index.html` 是 GitHub Pages 上的演示页，浏览器里只跑 BM25 关键词检索，用来展示「检索 → 引用 → 拒答」的交互。向量检索、BGE 重排和 DeepSeek 生成需要按上面的方式本地启动。
