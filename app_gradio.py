# -*- coding: utf-8 -*-
"""Gradio 界面：问答 + 引用溯源 + 混合检索 / Rerank 开关"""
from pathlib import Path

import gradio as gr

from pdf_split_demo import load_and_split_pdf
from rag_pipeline import rag_query
from vector_store import create_vector_store

DOC_DIR = Path(__file__).resolve().parent / "本地文档"
DOC_DIR.mkdir(exist_ok=True)


def format_sources(sources: list) -> str:
    if not sources:
        return "未检索到相关片段。"
    lines = []
    for s in sources:
        lines.append(f"### 来源{s['index']}：{s['label']}")
        lines.append(s["snippet"])
        lines.append("")
    return "\n".join(lines).strip()


def ask(question: str, top_k: int, use_hybrid: bool, use_rerank: bool):
    question = (question or "").strip()
    if not question:
        return "请先输入问题。", ""
    result = rag_query(
        question,
        top_k=int(top_k),
        use_hybrid=bool(use_hybrid),
        use_rerank=bool(use_rerank),
    )
    return result["answer"], format_sources(result["sources"])


def ingest_pdf(file_obj):
    if file_obj is None:
        return "请先选择 PDF 文件。"
    src = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    if not src.exists():
        return f"找不到文件：{src}"
    dest = DOC_DIR / src.name
    dest.write_bytes(src.read_bytes())
    docs = load_and_split_pdf(str(dest))
    create_vector_store(docs)
    return (
        f"入库完成：{src.name}，共切成 {len(docs)} 个文本块；"
        f"已同步写入 data/chunks.jsonl（供 BM25）。"
    )


with gr.Blocks(title="RAG 文档问答") as demo:
    gr.Markdown(
        "# 私有文档 RAG 问答\n"
        "上传 PDF 入库后提问；右侧显示引用来源。"
        "默认开启**混合检索（向量+BM25+RRF）**与 **BGE Rerank**。未入库提问会拒答，不会编造。"
    )
    with gr.Row():
        with gr.Column():
            pdf = gr.File(label="上传 PDF", file_types=[".pdf"])
            ingest_btn = gr.Button("入库", variant="secondary")
            ingest_msg = gr.Textbox(label="入库结果", interactive=False)
            question = gr.Textbox(
                label="问题",
                lines=3,
                placeholder="例如：中共一大是什么时候召开的？",
            )
            top_k = gr.Slider(1, 8, value=5, step=1, label="检索条数 Top-K")
            use_hybrid = gr.Checkbox(value=True, label="混合检索（向量 + BM25 + RRF）")
            use_rerank = gr.Checkbox(value=True, label="BGE Rerank 重排序")
            ask_btn = gr.Button("提问", variant="primary")
        with gr.Column():
            answer = gr.Markdown(label="回答")
            sources = gr.Markdown(label="引用来源")

    ingest_btn.click(ingest_pdf, inputs=[pdf], outputs=[ingest_msg])
    ask_btn.click(
        ask,
        inputs=[question, top_k, use_hybrid, use_rerank],
        outputs=[answer, sources],
    )
    question.submit(
        ask,
        inputs=[question, top_k, use_hybrid, use_rerank],
        outputs=[answer, sources],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
