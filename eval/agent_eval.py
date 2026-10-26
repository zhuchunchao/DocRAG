# -*- coding: utf-8 -*-
"""Agent 评测：20 题文档问答 + 5 题工具选择，量化「工具选择准确率」与「答案命中率」。

为什么单独评工具选择：
    普通 RAG 只考「检索准不准」，Agent 多了一个失败点——「该调的工具没调 / 调错工具」。
    不单独量化的话，Agent 只看起来更复杂，说不出好在哪。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_graph import run_agent  # noqa: E402

# 工具选择题：expect_tool=None 表示不该调任何工具（应直接拒答）
TOOL_CASES = [
    {"id": "t01", "question": "你现在的知识库里都有哪些文档？", "expect_tool": "list_docs", "must_keywords": []},
    {"id": "t02", "question": "知识库一共收了多少个文本块？", "expect_tool": "doc_stats", "must_keywords": []},
    {"id": "t03", "question": "库里面一共有几份文档？", "expect_tool": "doc_stats", "must_keywords": []},
    {"id": "t04", "question": "今天上海天气怎么样？", "expect_tool": None, "must_keywords": []},
    {"id": "t05", "question": "帮我写一首关于春天的诗。", "expect_tool": None, "must_keywords": []},
]


def _hits(text: str, kws) -> int:
    return sum(1 for k in kws if k and k in (text or ""))


def main():
    eval_set = Path(__file__).with_name("eval_set.json")
    items = json.loads(eval_set.read_text(encoding="utf-8"))

    rows = []
    lat = []
    tool_ok = []
    ans_hit = []

    print("== 文档问答（期望调 search_docs）==")
    for it in items:
        t0 = time.perf_counter()
        r = run_agent(it["question"])
        dt = time.perf_counter() - t0
        lat.append(dt)
        used = [c["name"] for c in r["tool_calls"]]
        ok = "search_docs" in used
        hit = 1 if _hits(r["answer"], it.get("must_keywords") or []) else 0
        tool_ok.append(1 if ok else 0)
        ans_hit.append(hit)
        rows.append({"id": it["id"], "question": it["question"], "tools": used,
                     "tool_ok": ok, "answer_hit": hit, "latency_s": round(dt, 3)})
        print(f"  [{it['id']}] tools={used} hit={hit} {dt:.2f}s", flush=True)

    print("\n== 工具选择（期望精确匹配）==")
    for c in TOOL_CASES:
        t0 = time.perf_counter()
        r = run_agent(c["question"])
        dt = time.perf_counter() - t0
        lat.append(dt)
        used = [x["name"] for x in r["tool_calls"]]
        if c["expect_tool"] is None:
            ok = len(used) == 0
        else:
            ok = used == [c["expect_tool"]]
        tool_ok.append(1 if ok else 0)
        rows.append({"id": c["id"], "question": c["question"], "tools": used,
                     "expect": c["expect_tool"], "tool_ok": ok,
                     "latency_s": round(dt, 3)})
        print(f"  [{c['id']}] 期望={c['expect_tool']} 实际={used} ok={ok}", flush=True)

    summary = {
        "n": len(rows),
        "tool_choice_accuracy": round(sum(tool_ok) / len(tool_ok), 4),
        "answer_hit": round(sum(ans_hit) / len(ans_hit), 4) if ans_hit else 0.0,
        "latency_avg_s": round(statistics.mean(lat), 3),
        "latency_p95_s": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 3),
    }
    print("\n### Agent 汇总")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = Path(__file__).with_name("agent_results.json")
    out.write_text(json.dumps({"summary": summary, "items": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n明细：{out}")


if __name__ == "__main__":
    main()
