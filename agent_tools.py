# -*- coding: utf-8 -*-
"""Agent 的工具层：把项目已有的检索能力包装成「模型可以主动调用」的工具。

设计要点（面试可讲）：
1. 工具只负责取数据，不做生成——生成统一放在图的最后一步，便于统一做溯源与拒答。
2. 工具返回字符串给模型看，同时内部保留结构化 sources 给人看（前端/评测要用）。
3. search_docs 复用 rag_pipeline.retrieve_only，保证 Agent 与普通问答走同一套检索逻辑。
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from langchain_core.tools import tool

from rag_pipeline import knowledge_base_ready, retrieve_only
from vector_store import load_chunks


def _fmt_sources(result: dict) -> str:
    """把检索结果拼成模型能读的上下文文本。"""
    if result.get("refused"):
        return "知识库为空：尚未入库任何文档，无法检索。"
    sources = result.get("sources") or []
    if not sources:
        return "未检索到与该问题相关的内容。"
    return "\n\n".join(f"[来源{s['index']}] {s['label']}\n{s['text']}" for s in sources)


def _search_impl(query: str, top_k: int = 5) -> Tuple[str, List[dict]]:
    result = retrieve_only(query, top_k=top_k, use_hybrid=True, use_rerank=False)
    return _fmt_sources(result), result.get("sources") or []


def _list_docs_impl() -> str:
    rows = load_chunks()
    if not rows:
        return "知识库为空，暂无已入库文档。"
    names = Counter()
    for r in rows:
        src = (r.get("metadata") or {}).get("source") or "未知来源"
        names[str(src)] += 1
    lines = [f"共 {len(names)} 份文档、{len(rows)} 个文本块："]
    lines += [f"- {name}（{n} 块）" for name, n in names.most_common()]
    return "\n".join(lines)


def _stats_impl() -> str:
    rows = load_chunks()
    if not rows:
        return "知识库为空，统计结果为 0。"
    docs = {str((r.get("metadata") or {}).get("source") or "未知来源") for r in rows}
    avg_len = sum(len(r.get("text") or "") for r in rows) / max(len(rows), 1)
    return f"文档数 {len(docs)}，文本块 {len(rows)}，块均长度 {avg_len:.0f} 字。"


@tool
def search_docs(query: str, top_k: int = 5) -> str:
    """在本地已入库的文档中检索与问题最相关的片段。任何关于文档内容的问题都应先调用它。"""
    text, _ = _search_impl(query, top_k)
    return text


@tool
def list_docs() -> str:
    """列出当前知识库里已入库的文档清单及各自的文本块数量。用户问「有哪些文档」时使用。"""
    return _list_docs_impl()


@tool
def doc_stats() -> str:
    """统计知识库规模（文档数、文本块数、块均长度）。用户问「库有多大」时使用。"""
    return _stats_impl()


TOOLS = [search_docs, list_docs, doc_stats]

# 名字 -> 实现：自定义工具节点用它执行，才能拿到 sources 这类结构化返回值。
TOOL_IMPL = {
    "search_docs": _search_impl,
    "list_docs": lambda: (_list_docs_impl(), []),
    "doc_stats": lambda: (_stats_impl(), []),
}


def kb_ready() -> bool:
    return knowledge_base_ready()
