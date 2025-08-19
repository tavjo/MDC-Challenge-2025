"""
Consolidated dataset construction for Kaggle.

Flow:
1) Build query texts from CitationEntity and Chunk (reuse construct_queries adapters)
2) Per dataset_id, restrict candidate chunks to same document_id
3) Compute id_to_dense for chunks on-the-fly using local BGE-small v1.5
4) Call hybrid_retrieve_with_boost to get relevant chunk_ids (ensure target+neighbors)
5) Apply mask_dataset_ids_in_text and construct_datasets_from_retrieval_results
6) Upsert Dataset rows into DuckDB
7) Bulk-embed dataset texts and export /kaggle/temp/dataset_embeddings.parquet
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from src.kaggle.models import CitationEntity, Dataset
from src.kaggle.helpers import load_bge_model, embed_texts
from src.kaggle.duckdb import get_duckdb_helper
from src.kaggle.retrieval_module import hybrid_retrieve_with_boost
from src.kaggle.models import BoostConfig

DEFAULT_DUCKDB_PATH = "/kaggle/temp/mdc.duckdb"


def add_target_and_neighbors_union(ranked_ids: List[str], target_and_neighbors: List[str], top_k: int) -> List[str]:
    """Return union of retrieval results and explicit target+neighbors, preserving order preference."""
    seen = set()
    ordered = []
    # include retrieval-ranked first (top_k)
    for cid in ranked_ids[:top_k]:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    # ensure target + neighbors are present
    for cid in target_and_neighbors:
        if cid and cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def mask_dataset_ids_in_text(text: str, dataset_ids: List[str], mask_token: str = "<DATASET_ID>") -> str:
    """Robust multi-ID masking with regex escaping (ported for Kaggle)."""
    if not dataset_ids:
        return text
    escaped_ids = [re.escape(ds_id) for ds_id in dataset_ids]
    pattern = r"\b(" + r"|".join(escaped_ids) + r")\b"
    return re.sub(pattern, mask_token, text, flags=re.IGNORECASE)


def construct_dataset_from_retrieval_results(
    document_id: str,
    dataset_id: str,
    chunk_ids: List[str],
    id_to_text: Dict[str, str],
) -> Dataset:
    text = "\n\n".join(
        mask_dataset_ids_in_text(id_to_text[cid], [dataset_id]) for cid in chunk_ids if cid in id_to_text
    )
    total_tokens = sum(len(t.split()) for t in text.split())
    avg_tokens_per_chunk = np.mean([len(id_to_text[cid].split()) for cid in chunk_ids if cid in id_to_text]) if chunk_ids else 0.0
    total_char_length = len(text)
    clean_text_length = len(text)
    return Dataset(
        dataset_id=dataset_id,
        document_id=document_id,
        total_tokens=int(total_tokens),
        avg_tokens_per_chunk=float(avg_tokens_per_chunk) if not np.isnan(avg_tokens_per_chunk) else 0.0,
        total_char_length=int(total_char_length),
        clean_text_length=int(clean_text_length),
        cluster=None,
        dataset_type=None,
        text=text,
    )


def build_dataset_embeddings(datasets: List[Dataset], model_dir: str, out_path: str = "/kaggle/temp/dataset_embeddings.parquet") -> Path:
    model = load_bge_model(model_dir)
    texts = [ds.text for ds in datasets]
    embs = embed_texts(model, texts)
    df = pd.DataFrame(embs) # TODO: Add checks here to make sure that this is the right shape: (len(datasets), 384)
    df.insert(0, "dataset_id", [ds.dataset_id for ds in datasets])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out

def get_query_texts(data_citation: CitationEntity, db_helper):
    """
    Get the query text for a given dataset ID
    """
    ce = data_citation
    # Retrieve all chunks for the document
    chunks = db_helper.get_chunks_by_document_id(ce.document_id)
    # Find the chunk containing this dataset citation
    target = None
    for c in chunks:
        ents = c.chunk_metadata.citation_entities or []
        if any(ent.data_citation == ce.data_citation for ent in ents):
            target = c
            break
    if not target:
        print(f"Warning: no chunk found for citation {ce.data_citation} in document {ce.document_id}")
        return "", []

    # Fetch neighbor chunks
    neighbor_ids = [cid for cid in (
        target.chunk_metadata.previous_chunk_id,
        target.chunk_metadata.next_chunk_id
    ) if cid]
    # neighbors = db_helper.get_chunks_by_chunk_ids(neighbor_ids)
    # get neighboring chunks from chunks that are in the same document --> no need to get from DB
    neighbors = [c for c in chunks if c.chunk_id in neighbor_ids]
    query_chunk_ids = neighbor_ids + [target.chunk_id]
    # Construct the query string
    if len(neighbors) > 0:
        parts = [target.text] + [n.text for n in neighbors]
    else:
        parts = [target.text]
    query_text = " ".join(parts)
    return query_text, query_chunk_ids

def construct_datasets_pipeline(
    model_dir: str,
    top_k: int = 30,
    db_path: str = DEFAULT_DUCKDB_PATH
) -> Tuple[List[Dataset], Path]:
    db = get_duckdb_helper(db_path)
    try:
        citations: List[CitationEntity] = db.get_all_citation_entities()

        # Cache per-document chunk text and dense embeddings to avoid recompute
        doc_cache_text: Dict[str, Dict[str, str]] = {}
        doc_cache_dense: Dict[str, Dict[str, np.ndarray]] = {}

        model = load_bge_model(model_dir)
        all_datasets: List[Dataset] = []

        for ce in citations:
            document_id = ce.document_id
            dataset_id = ce.data_citation

            query_text, query_chunk_ids = get_query_texts(ce, db)
            if not query_text or not query_chunk_ids:
                continue

            if document_id not in doc_cache_text:
                chunks = db.get_chunks_by_document_id(document_id)
                id_to_text = {ch.chunk_id: ch.text for ch in chunks}
                doc_cache_text[document_id] = id_to_text
                id_order = list(id_to_text.keys())
                texts = [id_to_text[cid] for cid in id_order]
                vecs = embed_texts(model, texts)
                doc_cache_dense[document_id] = {cid: v for cid, v in zip(id_order, vecs)}

            id_to_text = doc_cache_text[document_id]
            id_to_dense = doc_cache_dense[document_id]

            qvec = embed_texts(model, [query_text])[0]
            ranked_ids = hybrid_retrieve_with_boost(
                query_text=query_text,
                dense_query_vec=qvec,
                id_to_dense=id_to_dense,
                id_to_text=id_to_text,
                boost_cfg=BoostConfig(
                    sparse_k=top_k,
                    dense_k=top_k,
                    rrf_k=2 * top_k,
                    mmr_lambda=0.7,
                    mmr_top_k=top_k,
                ),
                prototypes=None,
            )

            selected_ids = add_target_and_neighbors_union(ranked_ids, query_chunk_ids, top_k)
            ds = construct_dataset_from_retrieval_results(document_id, dataset_id, selected_ids, id_to_text)
            all_datasets.append(ds)

        db.bulk_upsert_datasets(all_datasets)
        out_file = build_dataset_embeddings(all_datasets, model_dir=model_dir)
        return all_datasets, out_file
    finally:
        db.close()


__all__ = [
    "construct_datasets_pipeline",
]


