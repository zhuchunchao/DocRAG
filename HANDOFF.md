# RAG Demo 交接文件（HANDOFF）

- 更新时间：2026-10-19 18:52（Asia/Shanghai）
- 项目：`D:\Users\zhuchunchao\Desktop\agi_code\DocRAG`
- machineId：`fb127c0d-0ae1-42c1-a66a-d1b76df3ecaf`
- venv：`.venv`
- venv 路径别名（若有）：`C:\Users\zhuchunchao\Documents\rag_demo_patch\`

## 关键约束

1. **`--pdf` / `--docs` 无默认值**，必须显式传入；未传报错。禁止静默用政治.pdf。
2. **空库 / 无检索依据必须拒答**：未入库 →「知识库为空…请先入库」；无命中 →「不知道。知识库中不存在该内容」。
3. Gradio / FastAPI 共用 `rag_query` 拒答。
4. 指标以 `eval/results.json` 为准，**勿手写进 README**（README 只写"以 results.json 为准"）。

## 已完成（含本次续做）

- [x] `retrieval.py`：向量 + BM25(jieba) + RRF；可选 CrossEncoder rerank（**本地优先，默认不联网**）
- [x] `vector_store.py`：`uri=./milvus_demo.db`、chunks 落盘、`_vs_cache` 单例、`HF_ENDPOINT=hf-mirror.com`
- [x] `rag_pipeline.py`：hybrid/rerank 参数、懒加载 LLM、空库/无命中拒答
- [x] `app_gradio.py` / `main.py`：开关与 API 字段
- [x] `eval/eval_set.json`（20 题，Python 官方教程语料）/ `prepare_index.py` / `run_eval.py`（强制 --pdf/--docs）
- [x] `requirements.txt` / `README.md` / `.gitignore`（新增忽略 `models/`）
- [x] **`BAAI/bge-reranker-base` 权重已落地**：`models/models/BAAI--bge-reranker-base/snapshots/master/model.safetensors`（1.11GB，完整）
- [x] **`eval/results.json` 已生成**（2026-10-21，`--no-llm`，`pdf=本地文档\Python官方教程.pdf`，`reranker_loaded=true`）
- [x] 清理 `milvus_lite_test.db` / `milvus_lite_test2.db` / 半成品 `models/bge-reranker-base/` 与 HF 缓存残留

## 真实评测数字（源：`eval/results.json`）

| mode | Hit@K | KwCoverage | AvgLatency(s) | P95Latency(s) |
|------|-------|-----------|---------------|---------------|
| vector | 95.00% | 95.00% | 0.086 | 0.095 |
| hybrid | 100.00% | 100.00% | 0.094 | 0.100 |
| hybrid+rerank | 100.00% | 100.00% | 3.901 | 4.678 |

- 20 题、top_k=5、fetch_k=20、仅检索（未调 DeepSeek），语料=Python 3 中文官方教程（351 块）
- `rerank_applied=true` 仅第三模式；前两模式为 false（如实标记，不假装重排）
- 本语料上 Rerank 未再提升指标（hybrid 已达 100/100），代价是 CPU 上约 +3.8s/题

## 未完成（可选项）

- [ ] `eval/run_eval.py --pdf ...`（带 LLM 全链路，需 `.env` 里有效 DEEPSEEK_API_KEY）→ 产出 `answer_hit`
- [ ] LangGraph Agent 二期；FastAPI 工程化（MySQL/Redis/鉴权/日志）

## 文件清单

| 文件 | 说明 |
|------|------|
| `retrieval.py` | 混合检索 + RRF + Rerank（本地优先/快速降级） |
| `vector_store.py` | Milvus Lite + chunks 落盘 |
| `rag_pipeline.py` | 检索 → Prompt → LLM → sources |
| `app_gradio.py` / `main.py` | 界面 / FastAPI |
| `eval/eval_set.json` | 20 题评测集 |
| `eval/prepare_index.py` / `run_eval.py` | 建库 / 三模式评测 |
| `eval/results.json` | **已生成**（指标以此为准） |
| `models/models/BAAI--bge-reranker-base/snapshots/master/` | reranker 权重（本地，已在 .gitignore） |
| `data/chunks.jsonl` | 运行生成 |
| `HANDOFF.md` | 本文件 |

## 下一步最短指令

```powershell
cd D:\Users\zhuchunchao\Desktop\agi_code\DocRAG
$env:PYTHONIOENCODING='utf-8'

# 1) 重跑检索评测（三模式，reranker 走本地，无需联网）
.\.venv\Scripts\python.exe eval\run_eval.py --no-llm --pdf "本地文档\Python官方教程.pdf"

# 2) 全链路（含 LLM 答案关键词命中，需 .env 配好 Key）
.\.venv\Scripts\python.exe eval\run_eval.py --pdf "本地文档\Python官方教程.pdf"

# 3) Gradio 演示
.\.venv\Scripts\python.exe app_gradio.py   # http://127.0.0.1:7860
```

## 已知坑

1. **HF xet/cas-bridge + 公司网关自签证书 → SSL CERTIFICATE_VERIFY_FAILED**；本机结论：走 **ModelScope** 拉权重最稳（`snapshot_download(..., cache_dir='models')`），HF 镜像基本不可用。
2. `retrieval.get_reranker()` 顺序：`RERANKER_MODEL_PATH` → `models/bge-reranker-base/` → `models/models/BAAI--bge-reranker-base/snapshots/master/`；都找不到就**跳过重排**（不会卡下载）。要联网拉需显式 `RERANKER_ALLOW_DOWNLOAD=1`。
3. Milvus Lite 必须 `connection_args={"uri":"./milvus_demo.db"}`；`url` 会连 19530。
4. 频繁新建连接 → `too_many_pings`；已单例 + `similarity_search`。
5. `prepare_index` / `run_eval` **无默认 PDF**。
6. PowerShell 写中文易乱码；长命令优先 `*>` 落盘再读日志，别用 `Select-Object -Last N` 管道（会缓冲到结束才输出）。
7. 后台长任务（下载/评测）不要用 `Select-Object` 管道，否则日志看似"空"其实还在跑。

## 续做检查

1. `eval/results.json` 的 `meta.reranker_loaded` 应为 `true`；第三模式 `rerank_applied=true`。
2. 若换语料，需先 `--rebuild-index --docs <目录>`，并同步更新 `eval/eval_set.json`。
3. 简历数字只从 `results.json` 抄（Hit@K / KwCoverage / 时延），不要再手写未跑通的指标。
