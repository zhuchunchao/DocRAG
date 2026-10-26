# -*- coding: utf-8 -*-
"""参数扫描：用评测集（20 题）反向选 chunk_size / fetch_k / top_k。

用法：
  python eval/sweep_params.py --pdf a.pdf --pdf b.pdf
说明：每个 chunk_size 会重建一次索引，再用固定评测集跑 hybrid 检索，
      输出 Hit@5 / 关键词覆盖 / 时延，供人工选型（不做 LLM 调用）。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import eval_mode, load_eval_set  # noqa: E402


def rebuild(pdfs, chunk_size: int, overlap_ratio: float):
    from pdf_split_demo import load_and_split_pdf
    from vector_store import create_vector_store

    overlap = max(1, int(chunk_size * overlap_ratio))
    docs = []
    for p in pdfs:
        docs.extend(load_and_split_pdf(str(p), chunk_size=chunk_size, chunk_overlap=overlap))
    create_vector_store(docs)
    return len(docs), overlap


def main():
    ap = argparse.ArgumentParser(description="参数扫描：chunk_size / fetch_k / top_k")
    ap.add_argument("--pdf", action="append", default=[], required=True,
                    help="建库用 PDF，可重复，相对路径相对项目根")
    ap.add_argument("--eval-set", default=str(Path(__file__).with_name("eval_set.json")))
    ap.add_argument("--chunk-sizes", default="300,500,800")
    ap.add_argument("--fetch-ks", default="10,20,40")
    ap.add_argument("--top-ks", default="3,5")
    ap.add_argument("--overlap-ratio", type=float, default=0.16,
                    help="重叠占块长比例，默认 0.16（500 块→80 重叠，与线上配置一致）")
    ap.add_argument("--out", default=str(Path(__file__).with_name("sweep_results.json")))
    args = ap.parse_args()

    pdfs = []
    for raw in args.pdf:
        p = Path(raw)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        pdfs.append(p)

    items = load_eval_set(Path(args.eval_set))
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]
    fetch_ks = [int(x) for x in args.fetch_ks.split(",") if x.strip()]
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]

    rows = []
    for cs in chunk_sizes:
        n_chunks, overlap = rebuild(pdfs, cs, args.overlap_ratio)
        print(f"\n### chunk_size={cs} overlap={overlap} → {n_chunks} 块", flush=True)
        for fk in fetch_ks:
            if fk < max(top_ks):
                continue  # fetch_k 必须不小于 top_k
            for tk in top_ks:
                if fk < tk:
                    continue
                block = eval_mode(
                    items,
                    use_hybrid=True,
                    use_rerank=False,
                    top_k=tk,
                    fetch_k=fk,
                    with_llm=False,
                )
                s = block["summary"]
                row = {
                    "chunk_size": cs,
                    "overlap": overlap,
                    "n_chunks": n_chunks,
                    "fetch_k": fk,
                    "top_k": tk,
                    "hit_at_k": s["hit_at_k"],
                    "keyword_coverage": s["keyword_coverage_avg"],
                    "latency_avg_s": s["latency_avg_s"],
                    "latency_p95_s": s["latency_p95_s"],
                }
                rows.append(row)
                print(
                    f"  fetch_k={fk:>2} top_k={tk} | Hit={s['hit_at_k']:.2%} "
                    f"KwCov={s['keyword_coverage_avg']:.2%} "
                    f"avg={s['latency_avg_s']:.3f}s",
                    flush=True,
                )

    # 排序：先看 Hit，再看覆盖率，再看时延
    rows.sort(key=lambda r: (-r["hit_at_k"], -r["keyword_coverage"], r["latency_avg_s"]))
    print("\n### 排名（Hit@K 优先，其次覆盖率，再次时延）\n")
    print("| rank | chunk_size | fetch_k | top_k | chunks | Hit@K | KwCov | avg(s) |")
    print("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        print(
            f"| {i} | {r['chunk_size']} | {r['fetch_k']} | {r['top_k']} | "
            f"{r['n_chunks']} | {r['hit_at_k']:.2%} | {r['keyword_coverage']:.2%} | "
            f"{r['latency_avg_s']:.3f} |"
        )

    out = Path(args.out)
    out.write_text(
        json.dumps({"pdf": [str(p) for p in pdfs], "eval_n": len(items), "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细已写入: {out}")


if __name__ == "__main__":
    main()
