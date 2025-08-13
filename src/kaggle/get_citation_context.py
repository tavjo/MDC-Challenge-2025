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
import os, sys, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mdc_integration_demo")

# --- Import your module utilities ---
# Ensure the path containing mdc_retrieval_module.py is on sys.path
# sys.path.append("/path/to/your/module")  # if needed
from src.kaggle.retrieval_module import hybrid_retrieve_with_boost, DAS_LEXICON
from src.kaggle.helpers import load_bge_model, embed_texts
from src.kaggle.duckdb import get_duckdb_helper


# --- Simple query builder for dataset-citation hunting ---
def build_query_text() -> str:
    """
    A compact query string: DAS cues + accession keywords.
    For BGE v1.5, instruction is optional; add for short queries if you like.
    """
    accession_terms = [
        "DOI", "doi.org", "accession", "GSE", "SRA", "SRR",
        "ArrayExpress", "ENA", "BioProject", "PDB", "GISAID", "Zenodo", "Dryad", "Figshare"
    ]
    return " ".join(sorted(set(DAS_LEXICON + accession_terms)))

def maybe_add_bge_instruction(query: str, use_instruction: bool = True) -> str:
    """
    For short queries, BGE suggests prefixing an instruction.
    """
    if not use_instruction:
        return query
    instruction = "Represent this sentence for searching relevant passages: "
    return instruction + query

# --- Run retrieval on a single new document ---
def run_hybrid_retrieval_on_document(
    doc_id: str,
    model_dir: str | Path,
    top_k: int = 15,
    use_instruction: bool = True,
    prototypes: Optional[np.ndarray] = None,
    db_path: str | None = None,
) -> List[Tuple[str, str]]:
    """
    Returns a list of (chunk_id, chunk_preview) for top_k hits.
    """
    db_helper = get_duckdb_helper(db_path)
    chunks = db_helper.get_chunks_by_document_id(doc_id)
    db_helper.close()
    id_to_text: Dict[str, str] = {f"chunk_{i}": ch.text for i, ch in enumerate(chunks)}

    # 2) Load local BGE-small and embed chunks
    model = load_bge_model(model_dir)
    chunk_ids = list(id_to_text.keys())
    chunk_texts = [id_to_text[cid] for cid in chunk_ids]
    chunk_vecs = embed_texts(model, chunk_texts)
    id_to_dense: Dict[str, np.ndarray] = {cid: vec for cid, vec in zip(chunk_ids, chunk_vecs)}

    # 3) Build the (short) semantic query and embed it
    query_text = build_query_text()
    query_for_embedding = maybe_add_bge_instruction(query_text, use_instruction=use_instruction)
    query_vec = embed_texts(model, [query_for_embedding])[0]

    # 4) Run hybrid retrieval: sparse (BM25/TF-IDF) + dense -> RRF -> MMR
    ranked_ids = hybrid_retrieve_with_boost( # TODO: Fix to include the prototype ranker
        query_text=query_text,               # sparse side uses raw query text
        dense_query_vec=query_vec,           # dense side uses BGE embedding
        id_to_dense=id_to_dense,             # {chunk_id: 384-d vec}
        id_to_text=id_to_text,               # {chunk_id: raw text}
        sparse_k=30, dense_k=30, rrf_k=60,   # tweak as you like
        mmr_lambda=0.7, mmr_top_k=top_k,
        prototypes=prototypes,
    )

    # 5) Return top_k chunks with a short preview
    results: List[Tuple[str, str]] = []
    for cid in ranked_ids[:top_k]:
        text = id_to_text[cid]
        preview = (text[:240] + "…") if len(text) > 240 else text
        results.append((cid, preview))
    return results

# --- Example usage ---
if __name__ == "__main__":
    # Point this to your already-downloaded BGE-small v1.5 directory
    LOCAL_BGE_DIR = "offline_models/BAAI/bge-small-en-v1.5"

    # New document text (replace with your actual content)
    doc_text = """
    Data Availability: Sequencing data have been deposited in GEO under accession GSE123456.
    Additional raw reads are available in SRA under SRR987654. The processed matrices are on Zenodo (doi:10.5281/zenodo.9999999).
    Methods... Results... Discussion...
    """

    hits = run_hybrid_retrieval_on_document(
        doc_text=doc_text,
        model_dir=LOCAL_BGE_DIR,
        top_k=10,
        window_size=300,
        overlap=30,
        use_instruction=True,  # set False if you prefer no instruction
    )

    print("\nTop hits (chunk_id, preview):")
    for cid, preview in hits:
        print(f"- {cid}: {preview}")


