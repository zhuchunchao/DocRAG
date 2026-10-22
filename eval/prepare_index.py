# -*- coding: utf-8 -*-
"""重建向量库与 data/chunks.jsonl。必须显式传入 --pdf 或 --docs，无默认文档。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

USAGE = (
    "必须显式指定文档，例如：\n"
    "  python eval/prepare_index.py --pdf 本地文档/你的文件.pdf\n"
    "  python eval/prepare_index.py --pdf a.pdf --pdf b.pdf\n"
    "  python eval/prepare_index.py --docs 你的文档目录\n"
    "不要省略 --pdf / --docs（不会自动选用任何默认 PDF）。"
)


def resolve_pdfs(pdf_args: List[str], docs_dir: str) -> List[Path]:
    if not pdf_args and not docs_dir:
        raise SystemExit(USAGE)

    paths: List[Path] = []
    for raw in pdf_args:
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            p = Path(part)
            if not p.is_absolute():
                p = (ROOT / p).resolve()
            else:
                p = p.resolve()
            if not p.exists():
                raise FileNotFoundError(f"PDF 不存在: {p}")
            if p.suffix.lower() != ".pdf":
                raise ValueError(f"不是 PDF 文件: {p}")
            paths.append(p)

    if docs_dir:
        d = Path(docs_dir)
        if not d.is_absolute():
            d = (ROOT / d).resolve()
        else:
            d = d.resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"文档目录不存在: {d}")
        found = sorted(d.glob("*.pdf"))
        if not found:
            raise FileNotFoundError(f"目录中没有 PDF: {d}")
        paths.extend(found)

    seen = set()
    uniq: List[Path] = []
    for p in paths:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    if not uniq:
        raise SystemExit(USAGE)
    return uniq


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="重建索引与 chunks.jsonl（必须 --pdf 或 --docs）",
        epilog=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="PDF 路径，可重复；也可用逗号分隔多个。相对路径相对项目根。",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default="",
        help="文档目录：将该目录下全部 *.pdf 入库（与 --pdf 可同时用）",
    )
    args = parser.parse_args(argv)

    pdfs = resolve_pdfs(args.pdf, args.docs)

    from pdf_split_demo import load_and_split_pdf
    from vector_store import CHUNKS_PATH, create_vector_store

    all_docs = []
    for pdf in pdfs:
        print(f"加载并切块: {pdf}")
        docs = load_and_split_pdf(str(pdf))
        print(f"  -> {len(docs)} 个文本块")
        all_docs.extend(docs)

    if not all_docs:
        raise SystemExit("没有切出任何文本块，请检查 PDF 是否含文本层。")

    print(f"合计 {len(all_docs)} 块，写入向量库与 {CHUNKS_PATH}")
    create_vector_store(all_docs)
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
