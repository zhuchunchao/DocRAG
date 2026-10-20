# -*- coding: utf-8 -*-
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from retrieval import MIN_LEXICAL_RATIO, lexical_overlap_ratio, retrieve
from vector_store import CHUNKS_PATH, load_chunks

load_dotenv(Path(__file__).resolve().parent / ".env")

EMPTY_KB_ANSWER = (
    "知识库为空或尚未入库任何文档，无法回答。"
    "请先通过 Gradio「入库」、FastAPI /ingest，"
    "或 `python eval/prepare_index.py --pdf 你的文件.pdf` 上传文档后再提问。"
)
NO_EVIDENCE_ANSWER = "不知道。知识库中不存在该内容（检索未找到足够依据），请换个问法或补充相关文档后再问。"

RAG_PROMPT_TEMPLATE = """你是文档问答助手，请严格依据下面的上下文回答用户问题。
规则：
1. 只能使用上下文中的信息，禁止用外部知识或猜测补全。
2. 如果上下文为空、无关、或不足以回答，必须明确回答：不知道。知识库中不存在该内容。
3. 不要编造时间、人物、会议名称或数字。
4. 回答时如有依据，在关键事实后用 [来源n] 标注对应上下文编号。

【上下文】
{context}

【用户问题】
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

_llm = None


def knowledge_base_ready() -> bool:
    """是否已有可用切块落盘（入库完成的标志之一）。"""
    if not CHUNKS_PATH.exists() or CHUNKS_PATH.stat().st_size <= 0:
        return False
    return len(load_chunks()) > 0


def get_llm():
    """懒加载 DeepSeek（ChatOpenAI），避免纯检索评测时强依赖 API Key。"""
    global _llm
    if _llm is not None:
        return _llm
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key.startswith("请在这里"):
        raise RuntimeError("请在 .env 中配置 DEEPSEEK_API_KEY")
    _llm = ChatOpenAI(
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=api_key,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.1,
    )
    return _llm


def _source_label(doc) -> str:
    meta = doc.metadata or {}
    source = meta.get("source") or meta.get("file_path") or "未知文件"
    source_name = Path(str(source)).name
    page = meta.get("page")
    if page is None:
        page = meta.get("page_number")
    if page is not None:
        try:
            page = int(page) + 1  # PyPDFLoader 页码从 0 开始
        except (TypeError, ValueError):
            page = page
        return f"{source_name} 第{page}页"
    return source_name


def _empty_result(question: str, answer: str):
    return {
        "question": question,
        "answer": answer,
        "sources": [],
        "context": "",
        "docs": [],
        "refused": True,
    }


def retrieve_only(
    question: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    fetch_k: int = 20,
):
    """只做检索，不调用 LLM。返回 docs 与 sources 结构。"""
    if not knowledge_base_ready():
        return {
            "question": question,
            "docs": [],
            "sources": [],
            "context": "",
            "refused": True,
            "refuse_reason": "empty_kb",
        }

    doc_list = retrieve(
        question,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        fetch_k=fetch_k,
    )
    sources = []
    context_blocks = []
    for i, doc in enumerate(doc_list, start=1):
        label = _source_label(doc)
        snippet = (doc.page_content or "").strip()
        sources.append(
            {
                "index": i,
                "label": label,
                "snippet": snippet[:300] + ("..." if len(snippet) > 300 else ""),
                "metadata": doc.metadata or {},
                "text": snippet,
            }
        )
        context_blocks.append(f"[来源{i}] ({label})\n{snippet}")
    context_text = "\n\n".join(context_blocks) if context_blocks else ""
    return {
        "question": question,
        "docs": doc_list,
        "sources": sources,
        "context": context_text,
        "refused": False,
    }


def rag_query(
    question: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    fetch_k: int = 20,
):
    question = (question or "").strip()
    if not question:
        return _empty_result("", "请先输入问题。")

    if not knowledge_base_ready():
        return _empty_result(question, EMPTY_KB_ANSWER)

    # 离题闸门：问题实词在语料里覆盖率过低（语料中根本没有这种说法）→ 直接拒答，不调 LLM。
    # 不加这层的话，向量/BM25 总会返回“最接近”的几块，LLM 就可能拿无关片段硬答。
    ratio, matched = lexical_overlap_ratio(question)
    if matched == 0 or ratio < MIN_LEXICAL_RATIO:
        return _empty_result(question, NO_EVIDENCE_ANSWER)

    packed = retrieve_only(
        question,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        fetch_k=fetch_k,
    )

    # 检索无命中任何片段：直接拒答，不调用 LLM，避免编造
    if not packed.get("docs"):
        out = _empty_result(question, NO_EVIDENCE_ANSWER)
        out["context"] = packed.get("context") or ""
        return out

    llm = get_llm()
    final_prompt = prompt.format(
        context=packed["context"] or "（无相关上下文）",
        question=question,
    )
    answer = llm.invoke(final_prompt)
    return {
        "question": question,
        "answer": answer.content,
        "sources": [
            {k: v for k, v in s.items() if k != "text"} for s in packed["sources"]
        ],
        "context": packed["context"],
        "refused": False,
    }


if __name__ == "__main__":
    output = rag_query("中共一大是什么时候召开的？")
    print(output["answer"])
    for s in output["sources"]:
        print(s["index"], s["label"])
