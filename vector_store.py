# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

# hf 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 关闭 windows 软链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
# pymilvus Lite 必须用 uri（不能用 url），且以 .db 结尾
MILVUS_URI = os.environ.get("MILVUS_URI", "./milvus_demo.db")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
)

_vs_cache = {"store": None, "uri": None}


def save_chunks(docs) -> Path:
    """将切块落盘，供 BM25 混合检索使用。会写入 metadata['chunk_id']。"""
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            meta = dict(doc.metadata or {})
            meta["chunk_id"] = str(i)
            doc.metadata = meta
            record = {
                "id": str(i),
                "text": doc.page_content or "",
                "metadata": meta,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return CHUNKS_PATH


def load_chunks():
    """读取 data/chunks.jsonl，返回 list[dict]。"""
    if not CHUNKS_PATH.exists():
        return []
    rows = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def create_vector_store(docs):
    save_chunks(docs)
    vector_store = Milvus.from_documents(
        documents=docs,
        embedding=embedding_model,
        connection_args={"uri": MILVUS_URI},
        drop_old=True,
    )
    _vs_cache["store"] = vector_store
    _vs_cache["uri"] = MILVUS_URI
    return vector_store


def get_vector_store():
    """复用同一个 Milvus Lite 连接，避免评测时 too_many_pings。"""
    if _vs_cache["store"] is not None and _vs_cache["uri"] == MILVUS_URI:
        return _vs_cache["store"]
    vs = Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": MILVUS_URI},
    )
    _vs_cache["store"] = vs
    _vs_cache["uri"] = MILVUS_URI
    return vs


def get_retriever(top_k=5):
    vs = get_vector_store()
    return vs.as_retriever(search_kwargs={"k": top_k})


def reset_vector_store_cache():
    _vs_cache["store"] = None
    _vs_cache["uri"] = None


if __name__ == "__main__":
    import argparse

    from pdf_split_demo import load_and_split_pdf, resolve_pdf_path

    parser = argparse.ArgumentParser(description="把 PDF 入库到 Milvus Lite 并做一次检索自测")
    parser.add_argument("--pdf", help="待入库的 PDF 路径；也可用环境变量 RAG_DOC_PATH 指定")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    print(f"入库文档：{pdf_path}")
    docs = load_and_split_pdf(str(pdf_path))
    vs = create_vector_store(docs)
    retriever = get_retriever()
    res = retriever.invoke("测试问题")
    print(res[0].page_content)
