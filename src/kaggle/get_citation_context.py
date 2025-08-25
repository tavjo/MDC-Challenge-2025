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
) -> Optional[List[Tuple[str, str]]]:
    """
    Returns a list of (chunk_id, chunk_preview) for top_k hits.
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

    def try_add(chunk_id: Optional[str]) -> None:
        if chunk_id and (chunk_id in id_to_text) and (chunk_id not in seen_ids):
            final_ids.append(chunk_id)
            seen_ids.add(chunk_id)

    i = 0
    while len(final_ids) < top_k and i < len(ranked_ids):
        current_id = ranked_ids[i]
        # Add the primary ranked chunk
        try_add(current_id)
        if len(final_ids) >= top_k:
            break
        # Add up to two neighbors for the current chunk
        ch = id_to_chunk.get(current_id)
        if ch is not None:
            prev_id = ch.chunk_metadata.previous_chunk_id
            next_id = ch.chunk_metadata.next_chunk_id
            try_add(prev_id)
            if len(final_ids) >= top_k:
                break
            try_add(next_id)
        i += 1

    # Build previews for the deduped, ordered selection
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


