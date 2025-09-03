"""
Minimal integration of BGE-small (SentenceTransformers) with mdc_retrieval_module.py

What it does:
1) Chunk a new document with your sliding-window helper
2) Create dense embeddings (384-d) for each chunk using BAAI/bge-small-en-v1.5
3) Build sparse retriever (BM25 if available, else TF-IDF) over chunk texts
4) Run hybrid retrieval: sparse + dense -> RRF fuse -> MMR diversify
5) Print the top matched chunks with previews

Requirements:
- sentence-transformers  (for BGE embeddings)
- scikit-learn          (for TF-IDF fallback; optional if rank_bm25 is present)
- rank_bm25             (optional but recommended for stronger sparse retrieval)
- mdc_retrieval_module.py (the module I provided earlier, on your PYTHONPATH)

Notes:
- Uses normalize_embeddings=True (recommended for BGE) and optional query instruction for short queries.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

# --- Import your module utilities ---
# Ensure the path containing mdc_retrieval_module.py is on sys.path
# sys.path.append("/path/to/your/module")  # if needed
# try:
#     from src.kaggle.retrieval_module import hybrid_retrieve_with_boost, DAS_LEXICON
#     from src.kaggle.models import BoostConfig
#     from src.kaggle.helpers import load_bge_model, embed_texts, timer_wrap, initialize_logging
#     from src.kaggle.duckdb_utils import get_duckdb_helper
# except Exception:
from retrieval_module import hybrid_retrieve_with_boost, DAS_LEXICON, retrieval_with_boost, mmr_rerank
from models import BoostConfig
from helpers import load_bge_model, embed_texts, timer_wrap, initialize_logging
from duckdb_utils import get_duckdb_helper

logger = initialize_logging()

base_tmp = "/kaggle/temp/"


artifacts = os.path.join(base_tmp, "artifacts")
DEFAULT_DUCKDB = os.path.join(artifacts, "mdc_challenge.db")


# --- Simple query builder for dataset-citation hunting ---
@timer_wrap
def build_query_text() -> str:
    """
    A compact query string: DAS cues + accession keywords.
    For BGE v1.5, instruction is optional; add for short queries if you like.
    """
    accession_terms = [
        "accession", "GSE", "SRA", "SRR",
        "ArrayExpress", "ENA", "BioProject", "PDB", "GISAID", "Zenodo", "Dryad", "Figshare", "GEO", "Deposition number", "SRA", 
    ]
    return " ".join(sorted(set(DAS_LEXICON + accession_terms)))

@timer_wrap
def maybe_add_bge_instruction(query: str, use_instruction: bool = True) -> str:
    """
    For short queries, BGE suggests prefixing an instruction.
    """
    if not use_instruction:
        return query
    instruction = "Represent this sentence for searching relevant passages: "
    return instruction + query

# --- Helper: collapse neighbors before thresholding ---
def collapse_neighbors_by_rank(
    ranked_ids: List[str],
    id_to_chunk: Dict[str, object],
) -> List[str]:
    """
    Given global fused-order chunk IDs, remove immediate neighbors (prev/next)
    when both appear in the list. Keeps the higher-ranked (earlier) ID.
    Preserves original order of survivors.
    """
    keep: List[str] = []
    banned: set[str] = set()
    in_rank: set[str] = set(ranked_ids)

    for cid in ranked_ids:  # ranked_ids is already in descending fused order
        if cid in banned:
            continue
        keep.append(cid)
        ch = id_to_chunk.get(cid)
        if ch is not None and getattr(ch, "chunk_metadata", None) is not None:
            prev_id = getattr(ch.chunk_metadata, "previous_chunk_id", None)
            next_id = getattr(ch.chunk_metadata, "next_chunk_id", None)
            # Only ban if the neighbor is also present in ranked_ids
            if isinstance(prev_id, str) and prev_id in in_rank:
                banned.add(prev_id)
            if isinstance(next_id, str) and next_id in in_rank:
                banned.add(next_id)
    return keep

# --- Run retrieval on a single new document ---
@timer_wrap
def run_hybrid_retrieval_on_document(
    doc_id: str,
    # model_dir: str | Path,
    model,
    top_k: int = 15,
    use_instruction: bool = True,
    prototypes: Optional[np.ndarray] = None,
    db_path: str | None = None,
    # prototype_top_m: int = 3,
    max_tokens: int = 1000,
    boost_cfg: BoostConfig = BoostConfig(),
) -> Optional[List[Tuple[str, str]]]:
    """
    Returns a list of (chunk_id, chunk_preview) for top_k hits, additionally
    constrained by a cumulative token budget (max_tokens). Chunks are first
    selected via ranked anchors with neighbor expansion; if the cumulative
    token count exceeds max_tokens, items are pruned by repeatedly removing
    neighbors of the lowest-ranked anchor, then the anchor itself, moving
    upward through ranks until under budget.
    """
    print(f"Initializing DB Helper")
    db_helper = get_duckdb_helper(db_path)
    print(f"Retrieving chunks for document {doc_id}")
    chunks = db_helper.get_chunks_by_document_id(doc_id)
    if len(chunks) == 0:
        print(f"Retrieved no chunks for {doc_id}. Returning None.")
        return None
    print(f"Retrieved {len(chunks)} chunks for {doc_id}")
    db_helper.close()
    id_to_text: Dict[str, str] = {ch.chunk_id: ch.text for ch in chunks}
    id_to_chunk = {ch.chunk_id: ch for ch in chunks}

    # 2) Load local BGE-small and embed chunks
    print("Loading BGE small embeddings model...")
    # model = load_bge_model(model_dir)
    chunk_ids = list(id_to_text.keys())
    chunk_texts = [id_to_text[cid] for cid in chunk_ids]
    chunk_vecs = embed_texts(model, chunk_texts)
    id_to_dense: Dict[str, np.ndarray] = {cid: vec for cid, vec in zip(chunk_ids, chunk_vecs)}

    # 3) Build the (short) semantic query and embed it
    print("Build and embed short semantic query")
    query_text = build_query_text()
    query_for_embedding = maybe_add_bge_instruction(query_text, use_instruction=use_instruction)
    query_vec = embed_texts(model, [query_for_embedding])[0]

    # 4) Run hybrid retrieval: sparse (BM25/TF-IDF) + dense -> RRF -> MMR
    print("🔄 Starting hybrid retrieval...")
    ranked_ids = hybrid_retrieve_with_boost(  # TODO: Fix to include the prototype ranker
        query_text=query_text,                # sparse side uses raw query text
        dense_query_vec=query_vec,            # dense side uses BGE embedding
        id_to_dense=id_to_dense,              # {chunk_id: 384-d vec}
        id_to_text=id_to_text,                # {chunk_id: raw text}
        boost_cfg=BoostConfig(
                mmr_top_k=top_k,
                prototype_top_m=int(max(1, boost_cfg.prototype_top_m))
            ),
        prototypes=prototypes,
    )
    if ranked_ids:
        print(f"Retrieved {len(ranked_ids)} for document: {doc_id}")
    else:
        print(f"Retrieved no ranked IDs for document: {doc_id}")
        return None
    
    # Anchor-guided refinement:
    # Take top 3 ranked chunks from hybrid retrieval, find their neighbors, merge text,
    # embed the merged text (no instruction), then run retrieval_with_boost.
    # Downstream, use union: [top3 hybrid anchors] + [top_k from retrieval_with_boost], dedup-preserving order.
    anchors = ranked_ids[:3]
    combined_ids: List[str] = []
    seen_local: set[str] = set()
    for anchor_id in anchors:
        if anchor_id and anchor_id not in seen_local and anchor_id in id_to_chunk:
            combined_ids.append(anchor_id)
            seen_local.add(anchor_id)
            ch = id_to_chunk.get(anchor_id)
            if ch is not None and getattr(ch, "chunk_metadata", None) is not None:
                prev_id = getattr(ch.chunk_metadata, "previous_chunk_id", None)
                next_id = getattr(ch.chunk_metadata, "next_chunk_id", None)
                for nb_id in (prev_id, next_id):
                    if nb_id and (nb_id in id_to_text) and (nb_id not in seen_local):
                        combined_ids.append(nb_id)
                        seen_local.add(nb_id)

    combined_text = " ".join([id_to_text[cid] for cid in combined_ids if cid in id_to_text]).strip()
    if combined_text:
        combined_vec = embed_texts(model, [combined_text])[0]
    else:
        # Fallback to the earlier query embedding if no anchors/neighbors were available
        logger.warning("Error creating combined query vector. Using original query vec.")
        combined_vec = query_vec

    # Run retrieval with boost using the combined query embedding (keep same top_k)
    try:
        retrieved_ids = retrieval_with_boost(
            query_text=combined_text or None,
            dense_query_vec=combined_vec,
            id_to_dense=id_to_dense,
            id_to_text=id_to_text,
            boost_cfg=BoostConfig(
                mmr_top_k=top_k,
                prototype_top_m=int(max(1, boost_cfg.prototype_top_m))
            ),
            prototypes=prototypes,
        )
    except Exception:
        # Be conservative if anything goes wrong; continue with original ranking
        retrieved_ids = []

    # New ranking: top 3 hybrid anchors first, then top_k from retrieval_with_boost; dedup and preserve order
    ordered_union: List[str] = []
    seen_union: set[str] = set()
    for cid in anchors:
        if cid and cid not in seen_union:
            ordered_union.append(cid)
            seen_union.add(cid)
    for cid in (retrieved_ids or [])[:top_k]:
        if cid and cid not in seen_union:
            ordered_union.append(cid)
            seen_union.add(cid)
    if ordered_union:
        ranked_ids = ordered_union

    

    # 5) Expand hits with neighbors (prev/next) while respecting top_k and deduplicating
    final_ids: List[str] = []
    seen_ids: set[str] = set()
    # Track selection groups per anchor to support neighbor-first pruning later
    # Each group: {"anchor": str, "neighbors": List[str], "anchor_added": bool}

    def try_add(chunk_id: Optional[str]) -> bool:
        if chunk_id and (chunk_id in id_to_text) and (chunk_id not in seen_ids):
            final_ids.append(chunk_id)
            seen_ids.add(chunk_id)
            return True
        return False

    i = 0
    selection_groups: List[Dict[str, object]] = []

    while len(final_ids) < top_k and i < len(ranked_ids):
        current_id = ranked_ids[i]
        # Add the primary ranked chunk
        anchor_added = try_add(current_id)
        if len(final_ids) >= top_k:
            selection_groups.append({"anchor": current_id, "neighbors": [], "anchor_added": anchor_added})
            break
        # Add up to two neighbors for the current chunk
        ch = id_to_chunk.get(current_id)
        neighbors_added: List[str] = []
        if ch is not None:
            prev_id = ch.chunk_metadata.previous_chunk_id
            next_id = ch.chunk_metadata.next_chunk_id
            if len(final_ids) < top_k and try_add(prev_id):
                neighbors_added.append(prev_id)  # drop prev before next
            if len(final_ids) < top_k and try_add(next_id):
                neighbors_added.append(next_id)
        selection_groups.append({"anchor": current_id, "neighbors": neighbors_added, "anchor_added": anchor_added})
        i += 1

    # 6) Enforce token budget (max_tokens) with neighbor-first pruning from lowest rank upward
    id_to_tokens: Dict[str, int] = {}
    for cid in final_ids:
        ch = id_to_chunk.get(cid)
        tok = 0
        if ch is not None and getattr(ch, "chunk_metadata", None) is not None:
            # token_count is required, but be defensive
            tok = int(getattr(ch.chunk_metadata, "token_count", 0) or 0)
        id_to_tokens[cid] = tok

    current_tokens = sum(id_to_tokens.get(cid, 0) for cid in final_ids)
    print(f"Token budget check: selected {len(final_ids)} chunks totaling {current_tokens} tokens (max {max_tokens}).")

    if current_tokens > max_tokens and len(final_ids) > 0:
        selected_ids_set: set[str] = set(final_ids)

        def remove_id(candidate_id: Optional[str]) -> bool:
            nonlocal current_tokens
            if candidate_id and candidate_id in selected_ids_set:
                selected_ids_set.remove(candidate_id)
                current_tokens -= id_to_tokens.get(candidate_id, 0)
                return True
            return False

        # Iterate from lowest-ranked anchor group to highest
        stop = False
        for grp in reversed(selection_groups):
            if stop:
                break
            neighbors_list: List[str] = grp.get("neighbors", [])  # type: ignore[arg-type]
            # Drop neighbors first (in the order added: prev, then next)
            for nid in neighbors_list:
                if current_tokens <= max_tokens:
                    stop = True
                    break
                if remove_id(nid):
                    pass
            if current_tokens <= max_tokens:
                break
            # Then drop the anchor itself
            anchor_id: str = grp.get("anchor")  # type: ignore[assignment]
            remove_id(anchor_id)
            if current_tokens <= max_tokens:
                break

        final_ids = [cid for cid in final_ids if cid in selected_ids_set]
        print(f"After pruning: {len(final_ids)} chunks totaling {current_tokens} tokens (max {max_tokens}).")

    # 7) Build previews for the deduped, ordered (and pruned) selection
    results: List[Tuple[str, str]] = []
    for cid in final_ids:
        text = id_to_text[cid]
        preview = (text[:240] + "…") if len(text) > 240 else text
        results.append((cid, preview))
    print(f"✅ Done with Hybrid Retrieval for {doc_id}.")
    return results


# --- Run retrieval across ALL documents with percentile threshold ---
@timer_wrap
def run_hybrid_retrieval_on_docs(
    model,
    use_instruction: bool = True,
    prototypes: Optional[np.ndarray] = None,
    db_path: str | None = None,
    boost_cfg: BoostConfig = BoostConfig(),
    threshold_percentile: float = 70.0,
) -> Optional[List[Tuple[str, str, str]]]:
    """
    Global hybrid retrieval → neighbor collapse (pre-threshold) → percentile gate →
    MMR over true anchors → triplets.

    Returns a list of (pre_chunk_id, anchor_chunk_id, post_chunk_id) tuples in
    rank order. Empty strings are used when a neighbor is missing.
    """
    logger.info(
        "Initializing all-docs hybrid retrieval | use_instruction=%s, threshold_pct=%.2f",
        use_instruction,
        threshold_percentile,
    )
    # Log prototypes metadata without dumping full arrays
    if prototypes is None:
        logger.info("Prototypes: none provided")
    else:
        try:
            logger.info(
                "Prototypes: shape=%s dtype=%s", str(prototypes.shape), str(prototypes.dtype)
            )
        except Exception:
            logger.info("Prototypes: provided (shape unknown)")

    db_helper = get_duckdb_helper(db_path)
    logger.info("Retrieving all chunks across documents")
    chunks = db_helper.get_all_chunks()
    if len(chunks) == 0:
        logger.warning("Retrieved no chunks across documents. Returning None.")
        return None
    logger.info("Retrieved %d total chunks across all documents", len(chunks))
    db_helper.close()

    id_to_text: Dict[str, str] = {ch.chunk_id: ch.text for ch in chunks}
    id_to_chunk = {ch.chunk_id: ch for ch in chunks}

    # Embeddings for all chunks
    logger.info("Encoding %d chunks with BGE-small (all-docs)", len(id_to_text))
    chunk_ids = list(id_to_text.keys())
    chunk_texts = [id_to_text[cid] for cid in chunk_ids]
    chunk_vecs = embed_texts(model, chunk_texts)
    id_to_dense: Dict[str, np.ndarray] = {cid: vec for cid, vec in zip(chunk_ids, chunk_vecs)}
    try:
        logger.info("Chunk embeddings computed: shape=%s dtype=%s", str(chunk_vecs.shape), str(chunk_vecs.dtype))
    except Exception:
        logger.info("Chunk embeddings computed")

    # Build and embed short semantic query
    logger.info("Building and embedding short semantic query (all-docs)")
    query_text = build_query_text()
    query_for_embedding = maybe_add_bge_instruction(query_text, use_instruction=use_instruction)
    query_vec = embed_texts(model, [query_for_embedding])[0]
    try:
        logger.debug(
            "Query text length: raw=%d, embedded_with_instruction=%s",
            len(query_text),
            str(use_instruction),
        )
        logger.info("Query embedding: dim=%d dtype=%s", int(query_vec.shape[0]), str(query_vec.dtype))
    except Exception:
        pass

    # Configure retrieval to cover all chunks; reduce signal_k_multiplier to 1
    total_chunks = len(chunks)
    try:
        cfg = boost_cfg.model_copy(update={
            "mmr_top_k": int(total_chunks),
            "signal_k_multiplier": 1,
            "prototype_top_m": int(max(1, boost_cfg.prototype_top_m)),
        })
    except Exception:
        cfg = BoostConfig(
            rrf_k=boost_cfg.rrf_k,
            signal_k_multiplier=1,
            mmr_lambda=boost_cfg.mmr_lambda,
            mmr_top_k=int(total_chunks),
            prototype_top_m=int(max(1, boost_cfg.prototype_top_m)),
            retrieval_weights=boost_cfg.retrieval_weights,
        )
    try:
        cfg_dump = cfg.model_dump() if hasattr(cfg, "model_dump") else {
            "rrf_k": getattr(cfg, "rrf_k", None),
            "signal_k_multiplier": getattr(cfg, "signal_k_multiplier", None),
            "mmr_lambda": getattr(cfg, "mmr_lambda", None),
            "mmr_top_k": getattr(cfg, "mmr_top_k", None),
            "prototype_top_m": getattr(cfg, "prototype_top_m", None),
        }
        logger.info(
            "Retrieval config | mmr_top_k=%s signal_k_multiplier=%s prototype_top_m=%s rrf_k=%s mmr_lambda=%s",
            str(cfg_dump.get("mmr_top_k")),
            str(cfg_dump.get("signal_k_multiplier")),
            str(cfg_dump.get("prototype_top_m")),
            str(cfg_dump.get("rrf_k")),
            str(cfg_dump.get("mmr_lambda")),
        )
    except Exception:
        logger.info("Retrieval config prepared")

    # Single hybrid retrieval pass using fused score ordering; request scores
    logger.info("Starting hybrid retrieval across all documents (single pass, with scores)")
    ranked_ids_with_scores = hybrid_retrieve_with_boost(
        query_text=query_text,
        dense_query_vec=query_vec,
        id_to_dense=id_to_dense,
        id_to_text=id_to_text,
        boost_cfg=cfg,
        prototypes=prototypes,
        return_scores=True,
    )

    try:
        # Expect (pool_rank_by_fused_score, fused_score_map)
        ranked_ids, fused_score_map = ranked_ids_with_scores  # type: ignore[misc]
    except Exception:
        logger.error("Hybrid retrieval did not return scores; aborting.")
        return None

    if not ranked_ids:
        logger.warning("Retrieved no ranked IDs across all documents.")
        return None
    else:
        logger.info("Hybrid retrieval returned %d ranked candidates", len(ranked_ids))

    # (A) Neighbor collapse BEFORE threshold
    collapsed_ids = collapse_neighbors_by_rank(
        ranked_ids=ranked_ids,
        id_to_chunk=id_to_chunk,
    )
    logger.info("Anchors after pre-threshold neighbor-collapse: %d", len(collapsed_ids))

    # (B) Percentile threshold on collapsed anchors
    collapsed_scores = np.array([float(fused_score_map.get(cid, 0.0)) for cid in collapsed_ids], dtype=float)
    try:
        threshold_value = float(np.percentile(collapsed_scores, float(threshold_percentile)))
    except Exception:
        threshold_value = 0.0

    try:
        fs_min = float(np.min(collapsed_scores)) if collapsed_scores.size else 0.0
        fs_max = float(np.max(collapsed_scores)) if collapsed_scores.size else 0.0
        fs_mean = float(np.mean(collapsed_scores)) if collapsed_scores.size else 0.0
        logger.info(
            "Collapsed fused score stats | min=%.6f mean=%.6f max=%.6f threshold(p%.1f)=%.6f",
            fs_min,
            fs_mean,
            fs_max,
            threshold_percentile,
            threshold_value,
        )
        logger.debug(
            "Top-10 collapsed by fused score: %s",
            ", ".join(
                f"{cid}:{fused_score_map.get(cid, 0.0):.6f}"
                for cid in collapsed_ids[:10]
            ),
        )
    except Exception:
        pass

    anchor_ids = [cid for cid in collapsed_ids if float(fused_score_map.get(cid, 0.0)) >= threshold_value]
    logger.info(
        "Anchors after percentile gate p%.1f: %d (of %d collapsed)",
        threshold_percentile,
        len(anchor_ids),
        len(collapsed_ids),
    )
    if not anchor_ids:
        logger.info("No anchors survived threshold; returning empty result.")
        return []

    # (C) MMR over true anchors (bounded set)
    mmr_ranked = mmr_rerank(
        candidate_ids=anchor_ids,
        query_vec=query_vec,
        id_to_vec=id_to_dense,
        lambda_diversity=boost_cfg.mmr_lambda,
        top_k=len(anchor_ids),
    )
    logger.info("Anchors after MMR rerank: %d", len(mmr_ranked))
    if len(anchor_ids) > 5000:
        logger.debug("Large anchor set after threshold (%d) may impact latency.", len(anchor_ids))

    # (D) Build triplets from MMR-ranked anchors
    results: List[Tuple[str, str, str]] = []
    for anchor_id in mmr_ranked:
        pre_id = None
        post_id = None
        ch = id_to_chunk.get(anchor_id)
        if ch is not None and getattr(ch, "chunk_metadata", None) is not None:
            prev_id = getattr(ch.chunk_metadata, "previous_chunk_id", None)
            next_id = getattr(ch.chunk_metadata, "next_chunk_id", None)
            if isinstance(prev_id, str) and (prev_id in id_to_text):
                pre_id = prev_id
            if isinstance(next_id, str) and (next_id in id_to_text):
                post_id = next_id
        results.append((pre_id, anchor_id, post_id))

    logger.info(
        "Done with all-docs hybrid retrieval. Triplets=%d | collapsed=%d | after_threshold=%d | after_mmr=%d",
        len(results),
        len(collapsed_ids),
        len(anchor_ids),
        len(mmr_ranked),
    )
    return results

# --- Example usage ---
if __name__ == "__main__":
    # Point this to your already-downloaded BGE-small v1.5 directory
    LOCAL_BGE_DIR = "offline_models/BAAI/bge-small-en-v1.5"
    model = load_bge_model(LOCAL_BGE_DIR)

    # Expect a document ID as CLI arg; optional top_k as second arg
    if len(sys.argv) < 2:
        print("Usage: python get_citation_context.py <document_id> [top_k]")
        sys.exit(1)

    doc_id = sys.argv[1]
    try:
        tk = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    except Exception:
        tk = 10

    hits = run_hybrid_retrieval_on_document(
        doc_id=doc_id,
        model=model,
        top_k=tk,
        use_instruction=True,  # set False if you prefer no instruction
        prototypes=None,
        db_path=DEFAULT_DUCKDB,
    )

    if hits:
        print("\nTop hits (chunk_id, preview):")
        for cid, preview in hits:
            print(f"- {cid}: {preview}")


