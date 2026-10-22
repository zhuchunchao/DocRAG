# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel

from pdf_split_demo import load_and_split_pdf
from rag_pipeline import rag_query
from vector_store import create_vector_store

app = FastAPI(title="RAG后端服务")


class IngestRequest(BaseModel):
    pdf_path: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_hybrid: bool = True
    use_rerank: bool = True


@app.post("/ingest")
def ingest_document(req: IngestRequest):
    """接口1：上传/加载PDF文档入库（同步写 data/chunks.jsonl）"""
    docs = load_and_split_pdf(req.pdf_path)
    create_vector_store(docs)
    return {"code": 0, "msg": "文档入库完成", "chunk_count": len(docs)}


@app.post("/query")
def query_rag(req: QueryRequest):
    """接口2：RAG问答（含引用来源；可选混合检索/Rerank）"""
    return rag_query(
        req.question,
        top_k=req.top_k,
        use_hybrid=req.use_hybrid,
        use_rerank=req.use_rerank,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
