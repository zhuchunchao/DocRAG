"""混合检索：向量 + BM25 + RRF，可选 BGE Rerank。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from vector_store import CHUNKS_PATH, get_vector_store, load_chunks

RRF_K = 60
DEFAULT_FETCH_K = 20

# 拒答闸门：问题实词在语料中的覆盖率低于该值就判定为“离题”，直接拒答、不调 LLM。
# 阈值经 20 道语料内问题回归（最低 0.55）与离题问题（最高 0.33）标定，留有余量。
MIN_LEXICAL_RATIO = 0.40

_bm25_cache: dict = {"mtime": None, "bm25": None, "docs": None}
_term_universe_cache: dict = {"mtime": None, "terms": None}
_reranker = None
_reranker_error: Optional[str] = None


def _tokenize(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [t.strip() for t in jieba.lcut_for_search(text) if t.strip()]


def _doc_key(doc: Document) -> str:
    meta = doc.metadata or {}
    cid = meta.get("chunk_id")
    if cid is not None:
        return f"id:{cid}"
    src = meta.get("source") or meta.get("file_path") or ""
    page = meta.get("page", meta.get("page_number", ""))
    return f"c:{hash((doc.page_content or '')[:200])}|{src}|{page}"


def _chunks_to_documents(rows: Sequence[dict]) -> List[Document]:
    docs = []
    for row in rows:
        meta = dict(row.get("metadata") or {})
        if "chunk_id" not in meta:
            meta["chunk_id"] = str(row.get("id", ""))
        docs.append(Document(page_content=row.get("text") or "", metadata=meta))
    return docs


def _get_bm25_index():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"未找到 {CHUNKS_PATH}。请先入库 PDF（Gradio「入库」或运行 "
            "python eval/prepare_index.py）。"
        )
    mtime = CHUNKS_PATH.stat().st_mtime
    if _bm25_cache["bm25"] is not None and _bm25_cache["mtime"] == mtime:
        return _bm25_cache["bm25"], _bm25_cache["docs"]

    rows = load_chunks()
    if not rows:
        raise FileNotFoundError(
            f"{CHUNKS_PATH} 为空。请先对本地文档执行入库。"
        )
    docs = _chunks_to_documents(rows)
    tokenized = [_tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache["mtime"] = mtime
    _bm25_cache["bm25"] = bm25
    _bm25_cache["docs"] = docs
    return bm25, docs


def _content_terms(text: str) -> set:
    """实词集合：分词后保留长度≥2 的词（丢掉单字噪声）。"""
    return {t for t in _tokenize(text) if len(t) >= 2}


def lexical_overlap_ratio(query: str) -> Tuple[float, int]:
    """问题实词在语料中的覆盖率（0~1）与命中词数。

    向量检索/BM25 总会返回“最近的若干块”，即使问题与语料毫不相干。
    这个覆盖率用来判断问题是否离题：接近 0 说明语料里根本没有对应说法。
    """
    terms = _content_terms(query)
    if not terms:
        return 0.0, 0

    _, docs = _get_bm25_index()
    mtime = CHUNKS_PATH.stat().st_mtime
    if _term_universe_cache["terms"] is None or _term_universe_cache["mtime"] != mtime:
        universe = set()
        for d in docs:
            universe |= _content_terms(d.page_content or "")
        _term_universe_cache["mtime"] = mtime
        _term_universe_cache["terms"] = universe

    hit = terms & _term_universe_cache["terms"]
    return len(hit) / len(terms), len(hit)


def bm25_retrieve(query: str, top_k: int = DEFAULT_FETCH_K) -> List[Document]:
    bm25, docs = _get_bm25_index()
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            continue
        out.append(docs[i])
    return out


def vector_retrieve(query: str, top_k: int = DEFAULT_FETCH_K) -> List[Document]:
    # 直接 similarity_search，复用 get_vector_store 单例，避免 as_retriever 反复建通道
    vs = get_vector_store()
    return list(vs.similarity_search(query, k=int(top_k)) or [])


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Document]],
    rrf_k: int = RRF_K,
    top_k: int = DEFAULT_FETCH_K,
) -> List[Document]:
    scores = {}
    keep = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            if key not in keep:
                keep[key] = doc
    ordered = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [keep[k] for k in ordered[:top_k]]


def _local_reranker_dirs() -> List[str]:
    """候选本地模型目录（权重文件齐全才算可用）。"""
    root = Path(__file__).resolve().parent
    candidates = [
        os.environ.get("RERANKER_MODEL_PATH", ""),
        str(root / "models" / "bge-reranker-base"),
        str(root / "models" / "models" / "BAAI--bge-reranker-base" / "snapshots" / "master"),
    ]
    out = []
    for cand in candidates:
        if not cand or not Path(cand).is_dir():
            continue
        has_config = (Path(cand) / "config.json").exists()
        has_weight = any(
            (Path(cand) / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin", "onnx/model.onnx")
        )
        if has_config and has_weight:
            out.append(cand)
    return out


def get_reranker():
    """懒加载 CrossEncoder；失败时记录错误并返回 None（评测可降级为不重排）。

    优先加载本地已有权重（离线可用、零下载），其次才尝试联网拉取；
    下载相关超时设短，失败立即降级，避免评测长时间卡在重试。
    """
    global _reranker, _reranker_error
    if _reranker is not None:
        return _reranker
    if _reranker_error is not None:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except Exception as e:  # noqa: BLE001
        _reranker_error = f"sentence_transformers 不可用: {e}"
        print(f"[retrieval] Reranker 加载失败，将跳过重排: {_reranker_error}")
        return None

    # HF 下载相关环境（仅在真需要联网时生效）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")

    attempts: List[tuple] = []
    for path in _local_reranker_dirs():
        attempts.append((path, True))
    # 本地没有权重时，默认不联网下载（网络受限环境下会长时间卡住）；
    # 需要时显式设置 RERANKER_ALLOW_DOWNLOAD=1 再让它去 HF 镜像拉取。
    if os.environ.get("RERANKER_ALLOW_DOWNLOAD", "").strip() in ("1", "true", "True"):
        attempts.append(("BAAI/bge-reranker-base", False))

    if not attempts:
        _reranker_error = (
            "未找到本地 reranker 权重（可设 RERANKER_MODEL_PATH 指向本地目录，"
            "或设 RERANKER_ALLOW_DOWNLOAD=1 联网下载）"
        )
        print(f"[retrieval] Reranker 加载失败，将跳过重排: {_reranker_error}")
        return None

    errors = []
    for model_ref, local_only in attempts:
        try:
            _reranker = CrossEncoder(
                model_ref,
                device="cpu",
                local_files_only=local_only,
            )
            print(f"[retrieval] Reranker 已加载: {model_ref}")
            return _reranker
        except Exception as e:  # noqa: BLE001
            errors.append(f"{model_ref}: {e}")

    _reranker_error = " | ".join(errors)
    print(f"[retrieval] Reranker 加载失败，将跳过重排: {_reranker_error}")
    return None


def rerank_documents(
    query: str, docs: Sequence[Document], top_k: int
) -> List[Document]:
    if not docs:
        return []
    model = get_reranker()
    if model is None:
        return list(docs)[:top_k]
    pairs = [(query, d.page_content or "") for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked[:top_k]]


def retrieve(
    query: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    fetch_k: int = DEFAULT_FETCH_K,
) -> List[Document]:
    """统一检索入口。

    - 纯向量：use_hybrid=False
    - 混合：向量 + BM25，RRF 融合到 fetch_k
    - Rerank：对候选用 bge-reranker-base 重排后截断 top_k
    """
    query = (query or "").strip()
    if not query:
        return []

    fetch_k = max(int(fetch_k), int(top_k))

    if use_hybrid:
        try:
            vec_docs = vector_retrieve(query, top_k=fetch_k)
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] 向量检索失败，仅用 BM25: {e}")
            vec_docs = []
        try:
            bm25_docs = bm25_retrieve(query, top_k=fetch_k)
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] BM25 失败，仅用向量: {e}")
            bm25_docs = []

        if vec_docs and bm25_docs:
            candidates = rrf_fuse([vec_docs, bm25_docs], top_k=fetch_k)
        elif vec_docs:
            candidates = vec_docs[:fetch_k]
        else:
            candidates = bm25_docs[:fetch_k]
    else:
        candidates = vector_retrieve(query, top_k=fetch_k if use_rerank else top_k)

    if use_rerank:
        return rerank_documents(query, candidates, top_k=top_k)
    return list(candidates)[:top_k]


def retrieval_mode_name(use_hybrid: bool, use_rerank: bool) -> str:
    if use_hybrid and use_rerank:
        return "hybrid+rerank"
    if use_hybrid:
        return "hybrid"
    return "vector"
