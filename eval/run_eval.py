# -*- coding: utf-8 -*-
"""20 题评测：对比 vector / hybrid / hybrid+rerank 的 Hit@K、关键词覆盖与时延。"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vector_store import CHUNKS_PATH  # noqa: E402

PREPARE_USAGE = (
    "建索引必须显式传入文档，例如：\n"
    "  python eval/run_eval.py --no-llm --pdf 本地文档/你的文件.pdf\n"
    "  python eval/run_eval.py --no-llm --docs 你的文档目录\n"
    "或先: python eval/prepare_index.py --pdf 你的文件.pdf"
)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def _keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    text = text or ""
    return [kw for kw in keywords if kw and kw in text]


def _prepare_cmd(pdf_args, docs_dir: str):
    prepare_py = Path(__file__).with_name("prepare_index.py")
    cmd = [sys.executable, str(prepare_py)]
    for pdf in pdf_args or []:
        cmd.extend(["--pdf", pdf])
    if docs_dir:
        cmd.extend(["--docs", docs_dir])
    return cmd


def ensure_index(
    auto_prepare: bool,
    *,
    pdf_args=None,
    docs_dir: str = "",
    force_rebuild: bool = False,
) -> None:
    has_index = CHUNKS_PATH.exists() and CHUNKS_PATH.stat().st_size > 0
    need = force_rebuild or not has_index
    if not need:
        return

    has_docs = bool(pdf_args) or bool(docs_dir)
    if not has_docs:
        raise SystemExit(
            f"未找到索引 ({CHUNKS_PATH})，且未传入 --pdf / --docs。\n{PREPARE_USAGE}"
        )

    if not auto_prepare and not force_rebuild:
        raise SystemExit(
            f"未找到索引: {CHUNKS_PATH}\n请先 prepare_index 或去掉 --no-auto-prepare。\n{PREPARE_USAGE}"
        )

    print("执行 prepare_index.py …")
    import subprocess

    subprocess.check_call(_prepare_cmd(pdf_args, docs_dir), cwd=str(ROOT))


def load_eval_set(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 1:
        raise SystemExit(f"评测集无效: {path}")
    return data


def eval_mode(
    items: Sequence[dict],
    *,
    use_hybrid: bool,
    use_rerank: bool,
    top_k: int,
    fetch_k: int,
    with_llm: bool,
    rerank_available: bool = True,
) -> Dict[str, Any]:
    from rag_pipeline import rag_query, retrieve_only

    mode = (
        "hybrid+rerank"
        if use_hybrid and use_rerank
        else ("hybrid" if use_hybrid else "vector")
    )
    per_q = []
    latencies = []
    hit_flags = []
    coverages = []
    answer_hits = []

    for item in items:
        q = item["question"]
        kws = item.get("must_keywords") or []
        print(f"  [{item.get('id')}] ...", flush=True)
        t0 = time.perf_counter()
        if with_llm:
            result = rag_query(
                q,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                fetch_k=fetch_k,
            )
            ctx = result.get("context") or ""
            answer = result.get("answer") or ""
        else:
            result = retrieve_only(
                q,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                fetch_k=fetch_k,
            )
            ctx = result.get("context") or ""
            answer = ""
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        hit_list = _keyword_hits(ctx, kws)
        hit = 1 if hit_list else 0
        cov = (len(hit_list) / len(kws)) if kws else 0.0
        hit_flags.append(hit)
        coverages.append(cov)

        ans_hit = None
        if with_llm:
            ans_hit_list = _keyword_hits(answer, kws)
            ans_hit = 1 if ans_hit_list else 0
            answer_hits.append(ans_hit)

        per_q.append(
            {
                "id": item.get("id"),
                "question": q,
                "hit_at_k": hit,
                "keyword_coverage": round(cov, 4),
                "matched_keywords": hit_list,
                "latency_s": round(elapsed, 4),
                "answer_hit": ans_hit,
                "num_sources": len(result.get("sources") or []),
            }
        )

    summary = {
        "mode": mode,
        "n": len(items),
        "top_k": top_k,
        "fetch_k": fetch_k,
        "with_llm": with_llm,
        "rerank_requested": use_rerank,
        "rerank_applied": bool(use_rerank and rerank_available),
        "hit_at_k": round(sum(hit_flags) / len(hit_flags), 4) if hit_flags else 0.0,
        "keyword_coverage_avg": round(statistics.mean(coverages), 4) if coverages else 0.0,
        "latency_avg_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "latency_p95_s": round(_percentile(latencies, 95), 4) if latencies else 0.0,
    }
    if with_llm and answer_hits:
        summary["answer_hit"] = round(sum(answer_hits) / len(answer_hits), 4)

    return {"summary": summary, "per_question": per_q}


def print_markdown_table(summaries: Sequence[dict]) -> None:
    headers = [
        "mode",
        "Hit@K",
        "KwCoverage",
        "AvgLatency(s)",
        "P95Latency(s)",
        "AnswerHit",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for s in summaries:
        row = [
            str(s.get("mode")),
            f"{s.get('hit_at_k', 0):.2%}",
            f"{s.get('keyword_coverage_avg', 0):.2%}",
            f"{s.get('latency_avg_s', 0):.3f}",
            f"{s.get('latency_p95_s', 0):.3f}",
            (f"{s['answer_hit']:.2%}" if "answer_hit" in s else "-"),
        ]
        print("| " + " | ".join(row) + " |")


def main():
    parser = argparse.ArgumentParser(
        description="RAG 检索/问答评测",
        epilog=PREPARE_USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        default=str(Path(__file__).with_name("eval_set.json")),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="只评检索，不调用 DeepSeek",
    )
    parser.add_argument(
        "--auto-prepare",
        action="store_true",
        default=True,
        help="缺少索引时自动建库（仍必须提供 --pdf 或 --docs）",
    )
    parser.add_argument(
        "--no-auto-prepare",
        action="store_true",
        help="缺少索引时直接退出，不自动建库",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="评测前强制重建索引（必须同时给 --pdf 或 --docs）",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="建库用 PDF，可重复；无默认值，需显式传入",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default="",
        help="建库用文档目录（该目录下全部 *.pdf）；无默认值",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).with_name("results.json")),
    )
    args = parser.parse_args()
    auto_prepare = args.auto_prepare and not args.no_auto_prepare

    ensure_index(
        auto_prepare=auto_prepare,
        pdf_args=args.pdf,
        docs_dir=args.docs,
        force_rebuild=args.rebuild_index,
    )
    items = load_eval_set(Path(args.eval_set))

    # 先确定 reranker 是否可用（失败即降级，且如实记录，避免"假装重排过"）
    from retrieval import get_reranker

    rerank_available = get_reranker() is not None
    print(f"Reranker 可用: {rerank_available}")

    modes = [
        {"use_hybrid": False, "use_rerank": False},
        {"use_hybrid": True, "use_rerank": False},
        {"use_hybrid": True, "use_rerank": True},
    ]

    results = []
    summaries = []
    for m in modes:
        print(
            f"\n== 评测模式: hybrid={m['use_hybrid']} rerank={m['use_rerank']} =="
        )
        block = eval_mode(
            items,
            use_hybrid=m["use_hybrid"],
            use_rerank=m["use_rerank"],
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            with_llm=not args.no_llm,
            rerank_available=rerank_available,
        )
        results.append(block)
        summaries.append(block["summary"])
        print(
            f"Hit@K={block['summary']['hit_at_k']:.2%} "
            f"KwCov={block['summary']['keyword_coverage_avg']:.2%} "
            f"avg={block['summary']['latency_avg_s']:.3f}s "
            f"p95={block['summary']['latency_p95_s']:.3f}s"
        )

    print("\n### 汇总\n")
    print_markdown_table(summaries)

    out_path = Path(args.out)
    payload = {
        "meta": {
            "eval_set": str(Path(args.eval_set).resolve()),
            "top_k": args.top_k,
            "fetch_k": args.fetch_k,
            "no_llm": args.no_llm,
            "chunks_path": str(CHUNKS_PATH),
            "pdf": args.pdf,
            "docs": args.docs,
            "rebuild_index": args.rebuild_index,
            "reranker_loaded": rerank_available,
        },
        "modes": results,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
