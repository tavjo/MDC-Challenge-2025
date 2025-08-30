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
from retrieval_module import hybrid_retrieve_with_boost, DAS_LEXICON
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
        "ArrayExpress", "ENA", "BioProject", "PDB", "GISAID", "Zenodo", "Dryad", "Figshare", "GEO", "Deposition number", "SRA"
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

# --- Run retrieval on a single new document ---
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
    prototype_top_m: int = 3,
    max_tokens: int = 8000,
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
        boost_cfg=BoostConfig(prototype_top_m=int(max(1, prototype_top_m))),
        prototypes=prototypes,
    )
    if ranked_ids:
        print(f"Retrieved {len(ranked_ids)} for document: {doc_id}")
    else:
        print(f"Retrieved no ranked IDs for document: {doc_id}")
        return None

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

# --- Example usage ---
if __name__ == "__main__":
    # Point this to your already-downloaded BGE-small v1.5 directory
    LOCAL_BGE_DIR = "offline_models/BAAI/bge-small-en-v1.5"
    model = load_bge_model(LOCAL_BGE_DIR)

    # New document text (replace with your actual content)
    doc_text = """
    Data Availability: Sequencing data have been deposited in GEO under accession GSE123456.
    Additional raw reads are available in SRA under SRR987654. The processed matrices are on Zenodo (doi:10.5281/zenodo.9999999).
    Methods... Results... Discussion...
    """

    hits = run_hybrid_retrieval_on_document(
        doc_text=doc_text,
        model=model,
        top_k=10,
        use_instruction=True,  # set False if you prefer no instruction
        prototypes=None,
        db_path=None,
    )

    print("\nTop hits (chunk_id, preview):")
    for cid, preview in hits:
        print(f"- {cid}: {preview}")


