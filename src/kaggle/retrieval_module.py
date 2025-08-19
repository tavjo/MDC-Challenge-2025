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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
# THIS_DIR = Path(__file__).parent
# if str(THIS_DIR) not in sys.path:
#     sys.path.append(str(THIS_DIR))

import numpy as np

from .helpers import cosine_sim_matrix
from .models import BoostConfig as PydBoostConfig

# --- Retrieval weights (normalized-ish; tuned to avoid dominance) ---
RETRIEVAL_WEIGHTS: Dict[str, float] = {
    "prototype": 0.70,
    "das": 0.10,
    "rrf": 0.10,
    "regex": 0.05,
    "doi_repo": 0.05,
    "ref_penalty": 0.12,  # kept for completeness; unused since no section titles available
}

NEIGHBOR_BOOST_CAP: float = 0.05
REGEX_MAX_HITS_FOR_SCORE: int = 3

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
    # Generic DOI patterns removed from scoring per plan; repository-specific DOIs handled separately
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


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# ------------------------------
# Boost configuration (moved to models.BoostConfig)
# ------------------------------


# ------------------------------
# Prototype affinity helpers
# ------------------------------
def _prototype_topk(proto_mat: np.ndarray, chunk_mat: np.ndarray, k: int, top_m: int = 1):
    """Return (per_chunk_scores, top_indices[:k]) using mean of top_m sims per chunk.

    When top_m == 1, this reduces to max-sim over prototypes.
    """
    def _l2norm(m: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
        return m / denom
    if proto_mat.size == 0 or chunk_mat.size == 0:
        return np.zeros((chunk_mat.shape[0],), dtype=float), np.array([], dtype=int)
    Pn = _l2norm(proto_mat)
    Cn = _l2norm(chunk_mat)
    sims = Pn @ Cn.T  # [P, N]
    m = int(max(1, min(top_m, sims.shape[0])))
    if m == 1:
        per_chunk = np.max(sims, axis=0)
    else:
        # take mean of top-m similarities per column (per chunk)
        # use argpartition for efficiency
        top_m_vals = np.partition(sims, -m, axis=0)[-m:, :]
        per_chunk = np.mean(top_m_vals, axis=0)
    k_eff = min(int(k), per_chunk.shape[0]) if per_chunk.shape[0] > 0 else 0
    if k_eff <= 0:
        return per_chunk, np.array([], dtype=int)
    top_idx = np.argpartition(per_chunk, -k_eff)[-k_eff:]
    top_sorted = top_idx[np.argsort(per_chunk[top_idx])[::-1]]
    return per_chunk, top_sorted


# Specific accession-style patterns (tight)
REGEX_SPECIFIC: Dict[str, str] = {
    "gisaid": r"\bEPI(?:_ISL)?_?\d+\b",
    "geo_gse": r"\bGSE\d+\b",
    "sra_run": r"\bSRR\d+\b",
    "sra_experiment": r"\bSRX\d+\b",
    "sra_study": r"\bSRP\d+\b",
    "sra_sample": r"\bSRS\d+\b",
    "ena_bioproject": r"\bPRJ[A-Z]+\d+\b",
    "ena_run": r"\b[ED]RR\d+\b",
    "ena_experiment": r"\b[ED]RX\d+\b",
    "ena_sample": r"\b[ED]RS\d+\b",
    "ena_biosample": r"\bSAM[A-Z]?\d+\b",
    "arrayexpress": r"\bE-\w+-\d+\b",
    "pdb": r"\b[0-9][A-Za-z0-9]{3}\b",
    "uniprot": r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b",
    "interpro": r"\bIPR\d{6}\b",
    "pfam": r"\bPF\d{5}\b",
    "pfam_clan": r"\bCL\d{4}\b",
    "chembl": r"\bCHEMBL\d+\b",
    "kegg_ko": r"\bK\d{5}\b",
    "ensembl_gene": r"\bENS\w*\d+\b",
    "cellosaurus": r"\bCVCL_\w{4}\b",
    "empiar": r"\bEMPIAR-\d+\b",
    "pride_pxd": r"\bPXD\d+\b",
    "pride_prd": r"\bPRD\d+\b",
    "ncbi_refseq": r"\b[A-Z]{2}_\d+(?:\.\d+)?\b",
    "dbsnp": r"\brs\d+\b",
    "hgnc": r"\bHGNC:\d+\b",
    "kegg_pathway": r"\bmap\d{5}\b",
}
_REGEX_SPECIFIC_COMPILED: Dict[str, re.Pattern] = {k: re.compile(v, flags=re.IGNORECASE) for k, v in REGEX_SPECIFIC.items()}


# Repository DOI patterns (context-gated via DAS language and repo keywords in text)
REGEX_REPO_DOI: Dict[str, str] = {
    "figshare_doi": r"https?://doi\.org/10\.6084/(?:m9\.figshare\.|figshare\.)\S+",
    "dryad_doi": r"https?://doi\.org/10\.5061/dryad\.\S+",
    "zenodo_doi": r"https?://doi\.org/10\.5281/zenodo\.\S+",
    "pangaea_doi": r"https?://doi\.org/10\.1594/pangaea\.\S+",
    "mendeley_doi": r"https?://doi\.org/10\.(?:17632|18150)/\S+",
    "ccdc_doi": r"https?://doi\.org/10\.5517/\S+",
    "tcia_doi": r"https?://doi\.org/10\.7937/tcia\.\S+",
    "pasta_doi": r"https?://doi\.org/10\.6073/pasta-\S+",
}
_REPO_DOI_COMPILED: Dict[str, re.Pattern] = {k: re.compile(v, flags=re.IGNORECASE) for k, v in REGEX_REPO_DOI.items()}
_REPO_KEYWORDS = {"figshare", "dryad", "zenodo", "pangaea", "mendeley", "ccdc", "tcia", "pasta"}


def _das_prior(text: str) -> float:
    t = text.lower()
    hits = 0
    for w in DAS_LEXICON:
        if w in t:
            hits += 1
    return min(1.0, hits / 4.0)


def _regex_specific_score(text: str) -> float:
    n = 0
    for pat in _REGEX_SPECIFIC_COMPILED.values():
        if pat.search(text):
            n += 1
            if n >= REGEX_MAX_HITS_FOR_SCORE:
                break
    return float(n) / float(REGEX_MAX_HITS_FOR_SCORE)


def _doi_repo_prior(text: str) -> float:
    t = text.lower()
    doi_hit = any(pat.search(t) for pat in _REPO_DOI_COMPILED.values())
    if not doi_hit:
        return 0.0
    has_repo_word = any(w in t for w in _REPO_KEYWORDS)
    has_das_lang = _das_prior(text) > 0.0
    base = 0.6
    return 1.0 if has_das_lang or has_repo_word else base


# ----------------------------------------
# Hybrid retrieval with prototype priors
# ----------------------------------------
def hybrid_retrieve_with_boost(
    *,
    query_text: str,
    dense_query_vec: np.ndarray,
    id_to_dense: Dict[str, np.ndarray],
    id_to_text: Dict[str, str],
    boost_cfg: PydBoostConfig = PydBoostConfig(),
    prototypes=None,  # Prototypes | np.ndarray of centroids [P, D] | None
) -> List[str]:
    """Prototype-first hybrid retrieval with bounded priors and RRF/MMR fusion.

    Returns: final ranked list of chunk_ids.
    """
    # Dynamic caps based on requested output size
    sig_mult = getattr(boost_cfg, "signal_k_multiplier", 3)
    try:
        sig_mult = int(sig_mult)
    except Exception:
        sig_mult = 3
    TOPK_PER_SIGNAL = max(1, int(boost_cfg.mmr_top_k) * sig_mult)

    # -----------------------------
    # 1) Prototype-first candidates (primary when provided)
    # -----------------------------
    proto_candidates: List[str] = []
    proto_score_map: Dict[str, float] = {}
    if prototypes is not None and isinstance(prototypes, np.ndarray) and prototypes.size > 0:
        ordered_ids = list(id_to_dense.keys())
        if ordered_ids:
            chunk_mat = np.vstack([id_to_dense[cid] for cid in ordered_ids])
            per_chunk_scores, top_idx = _prototype_topk(prototypes, chunk_mat, k=TOPK_PER_SIGNAL, top_m=getattr(boost_cfg, "prototype_top_m", 1) if isinstance(getattr(boost_cfg, "prototype_top_m", 1), (int, float)) else 1)
            per_chunk_scores = _minmax_norm(per_chunk_scores)
            proto_candidates = [ordered_ids[i] for i in top_idx.tolist()] if top_idx.size > 0 else []
            proto_score_map = {cid: float(per_chunk_scores[i]) for i, cid in enumerate(ordered_ids)}

    # -----------------------------
    # 2) Secondary candidate generation (RRF over sparse + dense)
    # -----------------------------
    # Use prototype-centered query vector for dense if prototypes provided; otherwise fallback to provided dense_query_vec
    if prototypes is not None and isinstance(prototypes, np.ndarray) and prototypes.size > 0:
        dense_query_for_dense = np.mean(prototypes, axis=0)
    else:
        dense_query_for_dense = dense_query_vec

    sparse_ids = _sparse_topk(query_text, id_to_text, k=TOPK_PER_SIGNAL)
    dense_ids = _dense_topk(dense_query_for_dense, id_to_dense, k=TOPK_PER_SIGNAL)
    rrf = reciprocal_rank_fusion([sparse_ids, dense_ids], k=boost_cfg.rrf_k)
    rrf_ranking: List[str] = rrf.ranking
    # Normalize RRF scores across its pool for blending
    rrf_pool = rrf_ranking
    rrf_vals_pool = np.array([float(rrf.scores.get(cid, 0.0)) for cid in rrf_pool], dtype=float)
    rrf_vals_pool = _minmax_norm(rrf_vals_pool)
    rrf_norm_map: Dict[str, float] = {cid: float(rrf_vals_pool[i]) for i, cid in enumerate(rrf_pool)}

    # Union candidates (prototype-first dominance) and deduplicate preserving order
    candidate_ids: List[str] = list(dict.fromkeys(proto_candidates + rrf_ranking))
    if not candidate_ids:
        candidate_ids = rrf_ranking[:TOPK_PER_SIGNAL]

    # -----------------------------
    # 2) Feature computation
    # -----------------------------
    # RRF scores aligned to union
    score_rrf = np.array([rrf_norm_map.get(cid, 0.0) for cid in candidate_ids], dtype=float)
    # Prototype scores aligned to union
    score_proto = np.array([proto_score_map.get(cid, 0.0) for cid in candidate_ids], dtype=float)

    # Priors per candidate
    texts = [id_to_text.get(cid, "") for cid in candidate_ids]
    das_prior = np.array([_das_prior(t) for t in texts], dtype=float)
    regex_specific = np.array([_regex_specific_score(t) for t in texts], dtype=float)
    doi_repo = np.array([_doi_repo_prior(t) for t in texts], dtype=float)

    # Neighbor boost from adjacency in the RRF ranking (window=1), tiny and capped
    id_to_pos = {cid: i for i, cid in enumerate(rrf_ranking)}
    neighbor = np.zeros_like(regex_specific)
    for j, cid in enumerate(candidate_ids):
        pos = id_to_pos.get(cid, None)
        if pos is None:
            neighbor[j] = 0.0
            continue
        s = 0.0
        for off in (-1, 1):
            nb_pos = pos + off
            if nb_pos < 0 or nb_pos >= len(rrf_ranking):
                continue
            nb_id = rrf_ranking[nb_pos]
            s += _regex_specific_score(id_to_text.get(nb_id, ""))
        neighbor[j] = min(s * 0.02, NEIGHBOR_BOOST_CAP)

    # -----------------------------
    # 3) Final weighted score and MMR
    # -----------------------------
    W = RETRIEVAL_WEIGHTS
    final_score = (
        W["prototype"]  * _clip01(score_proto) +
        W["rrf"]        * _clip01(score_rrf) +
        W["das"]        * _clip01(das_prior) +
        W["regex"]      * _clip01(regex_specific) +
        W["doi_repo"]   * _clip01(doi_repo) +
        neighbor
    )

    order = np.argsort(final_score)[::-1]
    pool_rank = [candidate_ids[i] for i in order]
    pool_rank = [cid for cid in pool_rank if cid in id_to_dense]

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
