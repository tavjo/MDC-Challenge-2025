"""
mdc_retrieval_module.py
------------------------

Offline-friendly retrieval utilities for the MDC Challenge inference phase.

Focus only on:
  • DAS miner (section heuristics + lexicon features)
  • Compiled accession regex bank (DOI/GEO/SRA/...)
  • Hybrid retrieval: BM25↔dense, RRF fuse, then MMR re‑rank
    + Boost chunks in DAS/Methods and those adjacent to regex matches.

Zero external network required. Uses a **pure-Python BM25** implementation by default;
will optionally use scikit-learn TF‑IDF if available, but not required.

Usage (minimal):

    from mdc_retrieval_module import (
        hybrid_retrieve_with_boost,
        build_regex_index,
    )

    # Required maps (you already have these from DuckDB/Chroma)
    id_to_text  = {chunk_id: text, ...}
    id_to_dense = {chunk_id: np.ndarray(D,), ...}   # SAME embedding model used at train

    # Optional metadata
    id_to_section   = {chunk_id: "methods"|"data availability"|...}
    id_to_neighbors = {chunk_id: [neighbor_id1, neighbor_id2, ...]}

    # Prepare regex index once per corpus
    regex_index = build_regex_index(id_to_text)

    # Build a document-level query vector however you prefer (e.g., mean of its chunk vectors
    # or your global super-chunk representation projected to the model space)
    dense_query_vec = np.mean(np.stack(list(id_to_dense.values())), axis=0)

    ranked_ids = hybrid_retrieve_with_boost(
        query_text="data availability deposited in GEO",
        dense_query_vec=dense_query_vec,
        id_to_dense=id_to_dense,
        id_to_text=id_to_text,
        id_to_section=id_to_section,
        id_to_neighbors=id_to_neighbors,
        regex_index=regex_index,
        mmr_top_k=25,
    )

"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
from collections import defaultdict, Counter

import numpy as np

from .helpers import cosine_sim_matrix

# ================================
# DAS miner (lexicon + heuristics)
# ================================

DAS_LEXICON: List[str] = [
    "data availability",
    "data availability statement",
    "availability of data",
    "data are available",
    "data will be available",
    "dataset is available",
    "deposited in",
    "available upon request",
    "upon reasonable request",
    "accession",
    "repository",
    "doi",
    "zenodo",
    "figshare",
    "dryad",
    "geo",
    "gse",
    "sra",
    "ebi",
    "arrayexpress",
    "pdb",
    "uniprot",
]

SECTION_HEADERS_CUES: Dict[str, List[str]] = {
    "data availability": ["data availability", "availability of data"],
    "methods": ["methods", "materials and methods", "experimental procedures"],
}


def _tok_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def das_features(text: str) -> Dict[str, float]:
    """Return simple counts/densities for DAS cues in a text block."""
    t = text.lower()
    total = max(len(t), 1)
    hits = sum(t.count(w) for w in DAS_LEXICON)
    unique = sum(1 for w in DAS_LEXICON if w in t)
    return {
        "das_hits": float(hits),
        "das_unique": float(unique),
        "das_density": float(hits) / float(total),
    }


def guess_section(text: str) -> Optional[str]:
    """Heuristically label block as 'data availability' or 'methods' if cues appear early."""
    head = text[:800].lower()
    for sec, cues in SECTION_HEADERS_CUES.items():
        for cue in cues:
            if cue in head:
                return sec
    return None


# ==================================
# Accession / Identifier regex index
# ==================================

ACCESSION_REGEXES: Dict[str, str] = {
    "doi_url": r"https?://doi\.org/\S+",
    "doi_numeric": r"\b[0-9]{2}\.[0-9]{4,9}/\S+\b",
    "geo_gse": r"\bGSE\d+\b",
    "sra_run": r"\bSRR\d+\b",
    "sra_generic": r"\bSRA\d+\b",
    "ena_bioproject": r"\bPRJ\w+\d+\b",
    "arrayexpress": r"\bE-\w+\-\d+\b",
    "pdb": r"\bPDB\s?[0-9][A-Za-z0-9]{3}\b",
    "uniprot": r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b",  # simple, not exhaustive
    "chembl": r"\bCHEMBL\d+\b",
    "gisaid": r"\bEPI_ISL_\d+\b",
}

COMPILED_REGEXES: Dict[str, re.Pattern] = {
    name: re.compile(rx, flags=re.IGNORECASE) for name, rx in ACCESSION_REGEXES.items()
}


@dataclass
class RegexHit:
    kind: str
    span: Tuple[int, int]
    text: str


def find_accession_hits(text: str) -> List[RegexHit]:
    hits: List[RegexHit] = []
    for kind, pat in COMPILED_REGEXES.items():
        for m in pat.finditer(text):
            hits.append(RegexHit(kind=kind, span=m.span(), text=m.group(0)))
    return hits


def build_regex_index(id_to_text: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    """Precompute regex hit counts per chunk_id by kind and total."""
    out: Dict[str, Dict[str, int]] = {}
    for cid, txt in id_to_text.items():
        counts = {k: 0 for k in ACCESSION_REGEXES}
        total = 0
        for kind, pat in COMPILED_REGEXES.items():
            n = len(pat.findall(txt))
            counts[kind] = n
            total += n
        counts["_total"] = total
        out[cid] = counts
    return out


# ======================
# Pure-Python BM25 (Okapi)
# ======================

class BM25:
    """Tiny Okapi BM25 for offline use. Tokenizer is _tok_words."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [ _tok_words(t) for t in corpus ]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        # term frequencies per doc
        self.tf: List[Counter] = [ Counter(d) for d in self.docs ]
        # document frequencies
        df: Counter = Counter()
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        self.df = df
        # idf with +0.5 correction
        self.idf: Dict[str, float] = {
            w: math.log((self.N - f + 0.5) / (f + 0.5) + 1e-12) for w, f in df.items()
        }

    def _score_doc(self, q_tokens: List[str], i: int) -> float:
        score = 0.0
        dl = len(self.docs[i]) or 1
        K = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        tf_i = self.tf[i]
        for w in q_tokens:
            if w not in tf_i:
                continue
            idf = self.idf.get(w, 0.0)
            tf = tf_i[w]
            score += idf * (tf * (self.k1 + 1)) / (tf + K)
        return score

    def topk(self, query: str, k: int = 50) -> List[int]:
        q = _tok_words(query)
        scores = [ self._score_doc(q, i) for i in range(self.N) ]
        order = np.argsort(-np.asarray(scores))[:k]
        return order.tolist()


# ============================
# RRF fusion + MMR re‑ranking
# ============================

@dataclass
class RRFResult:
    scores: Dict[str, float]
    ranking: List[str]


def reciprocal_rank_fusion(rank_lists: List[List[str]], k: int = 60) -> RRFResult:
    """Return RRF scores and fused ranking.

    RRF score for id at rank r in a list: 1 / (k + r)
    """
    scores: Dict[str, float] = defaultdict(float)
    for rl in rank_lists:
        for r, cid in enumerate(rl, start=1):
            scores[cid] += 1.0 / (k + r)
    ranking = [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]
    return RRFResult(scores=dict(scores), ranking=ranking)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / (na * nb))


def mmr_rerank(candidate_ids: List[str],
               query_vec: np.ndarray,
               id_to_vec: Dict[str, np.ndarray],
               lambda_diversity: float = 0.7,
               top_k: int = 25) -> List[str]:
    selected: List[str] = []
    remaining = candidate_ids[:]
    while remaining and len(selected) < top_k:
        best_id, best_score = None, -1e9
        for cid in remaining:
            v = id_to_vec[cid]
            rel = _cos(query_vec, v)
            div = 0.0 if not selected else max(_cos(v, id_to_vec[sid]) for sid in selected)
            score = lambda_diversity * rel - (1 - lambda_diversity) * div
            if score > best_score:
                best_score, best_id = score, cid
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


# ==============================================
# Hybrid retrieval + boosting (DAS/Methods/regex)
# ==============================================

@dataclass
class BoostConfig:
    das_methods_boost: float = 0.20
    regex_hit_boost: float = 0.50
    regex_neighbor_boost: float = 0.25
    rrf_k: int = 60
    sparse_k: int = 60
    dense_k: int = 60
    mmr_lambda: float = 0.7
    mmr_top_k: int = 25


def _dense_topk(dense_query_vec: np.ndarray,
                id_to_dense: Dict[str, np.ndarray],
                k: int) -> List[str]:
    ids = list(id_to_dense.keys())
    M = np.vstack([id_to_dense[i] for i in ids])
    q = dense_query_vec
    qn = np.linalg.norm(q) + 1e-8
    Mn = np.linalg.norm(M, axis=1) + 1e-8
    sims = (M @ q) / (Mn * qn)
    order = np.argsort(-sims)[:k]
    return [ids[i] for i in order]


def _sparse_topk(query_text: str,
                 id_to_text: Dict[str, str],
                 k: int) -> List[str]:
    ids = list(id_to_text.keys())
    corpus = [id_to_text[i] for i in ids]
    bm25 = BM25(corpus)
    idxs = bm25.topk(query_text, k=k)
    return [ids[i] for i in idxs]


def _is_methods_or_das(section: Optional[str], text: str) -> bool:
    if section is not None:
        s = section.lower()
        if "method" in s or "availability" in s:
            return True
    # Fallback to heuristic
    sec_guess = guess_section(text)
    return sec_guess in {"methods", "data availability"}


def _normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return d
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-12:
        return d
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


# def hybrid_retrieve_with_boost(
#     *,
#     query_text: str,
#     dense_query_vec: np.ndarray,
#     id_to_dense: Dict[str, np.ndarray],
#     id_to_text: Dict[str, str],
#     id_to_section: Optional[Dict[str, Optional[str]]] = None,
#     id_to_neighbors: Optional[Dict[str, List[str]]] = None,
#     regex_index: Optional[Dict[str, Dict[str, int]]] = None,
#     boost_cfg: BoostConfig = BoostConfig(),
# ) -> List[str]:
#     """Run sparse+dense → RRF, then apply DAS/Methods/regex boosts, then MMR diversify.

#     Returns: final ranked list of chunk_ids.
#     """
#     # 1) Retrieve
#     sparse_ids = _sparse_topk(query_text, id_to_text, k=boost_cfg.sparse_k)
#     dense_ids  = _dense_topk(dense_query_vec, id_to_dense, k=boost_cfg.dense_k)

#     rrf = reciprocal_rank_fusion([sparse_ids, dense_ids], k=boost_cfg.rrf_k)
#     base_scores = rrf.scores  # id -> RRF score

#     # 2) Prepare regex counts
#     if regex_index is None:
#         regex_index = build_regex_index(id_to_text)

#     # Neighbor map
#     id_to_neighbors = id_to_neighbors or {}

#     # 3) Apply boosts
#     boosted: Dict[str, float] = dict(base_scores)

#     # Precompute which ids have regex hits
#     regex_positive_ids = {cid for cid, c in regex_index.items() if c.get("_total", 0) > 0}

#     # Direct regex boost
#     for cid in list(boosted.keys()):
#         if cid in regex_positive_ids:
#             n = regex_index[cid].get("_total", 0)
#             boosted[cid] = boosted[cid] + boost_cfg.regex_hit_boost * min(3.0, float(n))

#     # Neighbor regex boost
#     if id_to_neighbors:
#         for cid, neighbors in id_to_neighbors.items():
#             if cid not in boosted:
#                 continue
#             if any(nid in regex_positive_ids for nid in neighbors):
#                 boosted[cid] = boosted[cid] + boost_cfg.regex_neighbor_boost

#     # Section boost (Methods / Data Availability)
#     for cid in list(boosted.keys()):
#         sec = id_to_section[cid] if id_to_section and cid in id_to_section else None
#         txt = id_to_text.get(cid, "")
#         if _is_methods_or_das(sec, txt):
#             boosted[cid] = boosted[cid] + boost_cfg.das_methods_boost

#     # 4) Re-rank by boosted score, keep a pool for MMR
#     boosted = _normalize_scores(boosted)
#     pool_rank = [cid for cid, _ in sorted(boosted.items(), key=lambda x: -x[1])]

#     # 5) MMR diversification on the pool using dense vectors
#     pool_rank = [cid for cid in pool_rank if cid in id_to_dense]
#     final_ids = mmr_rerank(
#         candidate_ids=pool_rank,
#         query_vec=dense_query_vec,
#         id_to_vec=id_to_dense,
#         lambda_diversity=boost_cfg.mmr_lambda,
#         top_k=boost_cfg.mmr_top_k,
#     )
#     return final_ids

# ------------------------------
# Boost configuration (updated)
# ------------------------------
@dataclass
class BoostConfig:
    # Retrieval pools
    sparse_k: int = 30
    dense_k: int = 30
    rrf_k: int = 60

    # Section / regex boosts
    regex_hit_boost: float = 0.25
    regex_neighbor_boost: float = 0.10
    das_methods_boost: float = 0.20

    # Prototype priors (NEW)
    # 1) Add a prototype-based ranked list as another "ranker" in RRF
    proto_add_to_rrf: bool = True
    proto_rrf_k: int = 50         # how many items from the prototype rank list to feed into RRF
    # 2) Add an additive prototype boost post-RRF
    proto_weight: float = 0.20    # weight of prototype affinity boost
    proto_mode: str = "max"       # "max" or "mean_top_m"
    proto_top_m: int = 3          # if mode == "mean_top_m"
    proto_min_sim: float = 0.00   # floor to ignore tiny affinities (on normalized [0,1] scale)

    # Final diversification
    mmr_lambda: float = 0.70
    mmr_top_k: int = 15


# ------------------------------
# Prototype affinity helpers
# ------------------------------


def _prototype_affinity(
    id_to_dense: Dict[str, np.ndarray],
    prototypes,  # Prototypes | np.ndarray of centroids [P, D]
    mode: str = "max",
    top_m: int = 3,
) -> Dict[str, float]:
    """
    Compute a per-chunk prototype affinity in [0,1] via min-max normalization.
    Affinity = max over centroids, or mean of top-M sims to centroids.
    """
    if prototypes is None:
        return {cid: 0.0 for cid in id_to_dense.keys()}

    # Accept either Prototypes dataclass or raw centroids
    C = getattr(prototypes, "centroids", prototypes)
    if C is None or len(C) == 0:
        return {cid: 0.0 for cid in id_to_dense.keys()}

    ids = list(id_to_dense.keys())
    V = np.vstack([id_to_dense[cid] for cid in ids])  # [N, D]
    S = cosine_sim_matrix(V, C)                      # [N, P]

    if mode == "mean_top_m":
        m = max(1, min(top_m, S.shape[1]))
        # Take mean of top-m per row (no full sort: partial partition is faster)
        part = np.partition(S, -m, axis=1)[:, -m:]
        raw = part.mean(axis=1)
    else:  # "max"
        raw = S.max(axis=1)

    # Min-max normalize across all chunks so boosts are comparable to other signals
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-8:
        norm = np.zeros_like(raw, dtype=np.float32)
    else:
        norm = (raw - lo) / (hi - lo)
    return dict(zip(ids, norm))

def _rank_by_prototype_affinity(
    id_to_dense: Dict[str, np.ndarray],
    prototypes,
    mode: str = "max",
    top_m: int = 3,
) -> List[str]:
    """Return chunk_ids ranked by descending prototype affinity (no truncation here)."""
    aff = _prototype_affinity(id_to_dense, prototypes, mode=mode, top_m=top_m)
    return [cid for cid, _ in sorted(aff.items(), key=lambda x: -x[1])]


# ----------------------------------------
# Hybrid retrieval with prototype priors
# ----------------------------------------
def hybrid_retrieve_with_boost(
    *,
    query_text: str,
    dense_query_vec: np.ndarray,
    id_to_dense: Dict[str, np.ndarray],
    id_to_text: Dict[str, str],
    id_to_section: Optional[Dict[str, Optional[str]]] = None,
    id_to_neighbors: Optional[Dict[str, List[str]]] = None,
    regex_index: Optional[Dict[str, Dict[str, int]]] = None,
    boost_cfg: BoostConfig = BoostConfig(),
    prototypes=None,  # Prototypes | np.ndarray of centroids [P, D] | None
) -> List[str]:
    """Run sparse+dense → (optionally + prototype rank) → RRF → boosts (regex/section/prototype) → MMR.

    Returns: final ranked list of chunk_ids.
    """
    # ----------------
    # 1) Retrieve
    # ----------------
    sparse_ids = _sparse_topk(query_text, id_to_text, k=boost_cfg.sparse_k)
    dense_ids  = _dense_topk(dense_query_vec, id_to_dense, k=boost_cfg.dense_k)

    rank_lists = [sparse_ids, dense_ids]

    # Optional: include prototype-based ranking as another signal into RRF
    if boost_cfg.proto_add_to_rrf and prototypes is not None:
        proto_rank = _rank_by_prototype_affinity(
            id_to_dense=id_to_dense,
            prototypes=prototypes,
            mode=boost_cfg.proto_mode,
            top_m=boost_cfg.proto_top_m,
        )
        # Limit how many from this source we feed into RRF to avoid overpowering
        proto_rank = proto_rank[:boost_cfg.proto_rrf_k]
        rank_lists.append(proto_rank)

    rrf = reciprocal_rank_fusion(rank_lists, k=boost_cfg.rrf_k)
    base_scores = rrf.scores  # id -> RRF score (assumes your implementation exposes .scores)

    # ----------------
    # 2) Prep indices
    # ----------------
    if regex_index is None:
        regex_index = build_regex_index(id_to_text)
    id_to_neighbors = id_to_neighbors or {}

    # ----------------
    # 3) Apply boosts
    # ----------------
    boosted: Dict[str, float] = dict(base_scores)

    # (a) Regex hits on the chunk itself
    regex_positive_ids = {cid for cid, c in regex_index.items() if c.get("_total", 0) > 0}
    for cid in list(boosted.keys()):
        if cid in regex_positive_ids:
            n = regex_index[cid].get("_total", 0)
            boosted[cid] = boosted[cid] + boost_cfg.regex_hit_boost * min(3.0, float(n))

    # (b) Neighbor regex proximity
    if id_to_neighbors:
        for cid, neighbors in id_to_neighbors.items():
            if cid not in boosted:
                continue
            if any(nid in regex_positive_ids for nid in neighbors):
                boosted[cid] = boosted[cid] + boost_cfg.regex_neighbor_boost

    # (c) Section prior (Methods / Data Availability)
    for cid in list(boosted.keys()):
        sec = id_to_section[cid] if id_to_section and cid in id_to_section else None
        txt = id_to_text.get(cid, "")
        if _is_methods_or_das(sec, txt):
            boosted[cid] = boosted[cid] + boost_cfg.das_methods_boost

    # (d) Prototype affinity boost (NEW)
    if prototypes is not None and boost_cfg.proto_weight > 0:
        aff = _prototype_affinity(
            id_to_dense=id_to_dense,
            prototypes=prototypes,
            mode=boost_cfg.proto_mode,
            top_m=boost_cfg.proto_top_m,
        )  # normalized [0,1]
        for cid in list(boosted.keys()):
            a = aff.get(cid, 0.0)
            if a > boost_cfg.proto_min_sim:
                boosted[cid] = boosted[cid] + boost_cfg.proto_weight * (a - boost_cfg.proto_min_sim)

    # ----------------
    # 4) Re-rank & MMR
    # ----------------
    boosted = _normalize_scores(boosted)
    pool_rank = [cid for cid, _ in sorted(boosted.items(), key=lambda x: -x[1])]
    pool_rank = [cid for cid in pool_rank if cid in id_to_dense]  # ensure we can MMR with dense vecs

    final_ids = mmr_rerank(
        candidate_ids=pool_rank,
        query_vec=dense_query_vec,
        id_to_vec=id_to_dense,
        lambda_diversity=boost_cfg.mmr_lambda,
        top_k=boost_cfg.mmr_top_k,
    )
    return final_ids


# =====================
# Convenience utilities
# =====================

def explain_scores(
    ids: List[str],
    *,
    rrf_scores: Dict[str, float],
    regex_index: Dict[str, Dict[str, int]],
    id_to_section: Optional[Dict[str, Optional[str]]] = None,
    id_to_text: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    """Return a small table (list of dicts) describing why each id scored well."""
    rows: List[Dict[str, object]] = []
    for cid in ids:
        sec = id_to_section.get(cid) if id_to_section else None
        txt = id_to_text.get(cid, "") if id_to_text else ""
        guess = guess_section(txt)
        rx = regex_index.get(cid, {})
        rows.append({
            "chunk_id": cid,
            "rrf": round(float(rrf_scores.get(cid, 0.0)), 6),
            "regex_total": int(rx.get("_total", 0)),
            "section": sec,
            "section_guess": guess,
            "has_methods_or_das": bool(_is_methods_or_das(sec, txt)),
        })
    return rows
