# -*- coding: utf-8 -*-
"""LangGraph 工具调用 Agent。

图结构：
    START → agent ─┬─(有 tool_calls 且未超轮数)→ tools → agent
                   └─(无 tool_calls / 超轮数)→ END

三个关键设计：
1. 用「自定义工具节点」而不是内置 ToolNode：因为要在执行检索工具时顺手把 sources 收进 state，
   内置 ToolNode 只回传字符串，溯源信息会丢。
2. 用 steps 计数限制最大工具轮数（默认 3），避免模型反复调同一个工具打转。
3. 模型不可用时直接降级走 rag_query（复用已有的降级逻辑），不把异常抛给前端。
"""
from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent_tools import TOOLS, TOOL_IMPL, kb_ready
from rag_pipeline import NO_EVIDENCE_ANSWER, get_llm, rag_query

MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = (
    "你是 DocRAG 的文档问答助手，只能依据工具返回的文档片段回答。\n"
    "规则：\n"
    "1. 涉及文档内容的问题，必须先调用 search_docs 检索，不得凭记忆作答。\n"
    "2. 用户问「有哪些文档/库有多大」时，调用 list_docs 或 doc_stats。\n"
    "3. 检索结果为空或与问题无关时，直接回答「不知道。知识库中不存在该内容」，不要编造。\n"
    "4. 同一问题最多调用一次 search_docs；只有首轮结果不足以回答时，才换关键词重试一次。\n"
    "5. 回答文档内容类问题时不要调用 list_docs / doc_stats，它们只用于查询知识库自身信息。\n"
    "6. 回答用中文，简洁，可在句末用 [来源n] 标注依据。"
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sources: List[dict]
    steps: int
    degraded: bool
    degrade_reason: str


def agent_node(state: AgentState) -> dict:
    """模型节点：带上工具定义，让模型自行决定调不调工具、调哪个。"""
    llm = get_llm().bind_tools(TOOLS)
    msg = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [msg], "steps": state.get("steps", 0) + 1}


def tools_node(state: AgentState) -> dict:
    """工具节点：按模型给出的 tool_calls 逐个执行，并收集 sources。"""
    last = state["messages"][-1]
    sources = list(state.get("sources") or [])
    out_msgs = []
    for call in last.tool_calls:
        name = call.get("name")
        args = call.get("args") or {}
        impl = TOOL_IMPL.get(name)
        if impl is None:
            out_msgs.append(
                ToolMessage(content=f"未知工具：{name}", tool_call_id=call.get("id"), name=name)
            )
            continue
        try:
            if name == "search_docs":
                text, srcs = impl(args.get("query") or "", int(args.get("top_k") or 5))
                sources.extend(srcs)
            else:
                text, _ = impl()
        except Exception as e:  # noqa: BLE001 工具异常不能让整轮对话崩掉
            text = f"工具执行失败：{type(e).__name__}: {e}"
        out_msgs.append(ToolMessage(content=text, tool_call_id=call.get("id"), name=name))
    return {"messages": out_msgs, "sources": sources}


def route(state: AgentState) -> str:
    """条件边：还有工具调用且没超轮数就继续，否则结束。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        if state.get("steps", 0) <= MAX_TOOL_ROUNDS:
            return "tools"
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(question: str, top_k: int = 5) -> dict:
    """对外接口：返回 {answer, sources, steps, degraded, tool_calls}。"""
    question = (question or "").strip()
    if not question:
        return {"answer": "请先输入问题。", "sources": [], "steps": 0,
                "degraded": False, "tool_calls": []}
    if not kb_ready():
        return {"answer": "知识库为空，请先上传或入库文档后再提问。", "sources": [],
                "steps": 0, "degraded": False, "tool_calls": []}

    init = {"messages": [HumanMessage(content=question)], "sources": [],
            "steps": 0, "degraded": False, "degrade_reason": ""}
    try:
        final = get_graph().invoke(init, {"recursion_limit": 12})
    except Exception as e:  # noqa: BLE001 模型/网络故障 → 退回普通 RAG（自带降级）
        fallback = rag_query(question, top_k=top_k, use_hybrid=True, use_rerank=False)
        return {"answer": fallback.get("answer") or NO_EVIDENCE_ANSWER,
                "sources": fallback.get("sources") or [], "steps": 0,
                "degraded": True, "degrade_reason": f"{type(e).__name__}: {e}",
                "tool_calls": []}

    tool_calls = []
    for m in final["messages"]:
        for c in getattr(m, "tool_calls", None) or []:
            tool_calls.append({"name": c.get("name"), "args": c.get("args")})
    last = final["messages"][-1]
    answer = last.content if isinstance(last, AIMessage) else ""
    return {"answer": answer or NO_EVIDENCE_ANSWER,
            "sources": [{k: v for k, v in s.items() if k != "text"} for s in final.get("sources") or []],
            "steps": final.get("steps", 0), "degraded": False,
            "tool_calls": tool_calls}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "DeepSeek 的 base_url 是什么？"
    r = run_agent(q)
    print(f"\n问题：{q}")
    print(f"工具调用：{r['tool_calls']}")
    print(f"来源数：{len(r['sources'])}  降级：{r['degraded']}")
    print(f"\n回答：\n{r['answer']}")
