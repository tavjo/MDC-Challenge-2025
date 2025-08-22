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
from typing import Dict, List, Tuple
import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys, os

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

try:
    from src.kaggle.models import CitationEntity, Dataset
    from src.kaggle.helpers import embed_texts, timer_wrap, initialize_logging
    from src.kaggle.duckdb_utils import get_duckdb_helper
    from src.kaggle.retrieval_module import hybrid_retrieve_with_boost
    from src.kaggle.models import BoostConfig
except Exception:
    from .models import CitationEntity, Dataset
    from .helpers import embed_texts, timer_wrap, initialize_logging
    from .duckdb_utils import get_duckdb_helper
    from .retrieval_module import hybrid_retrieve_with_boost
    from .models import BoostConfig

logger = initialize_logging()

base_tmp = "/kaggle/temp/"

artifacts = os.path.join(base_tmp, "artifacts")
dataset_embeddings_path = os.path.join(base_tmp, "dataset_embeddings.parquet")
DEFAULT_DUCKDB = os.path.join(artifacts, "mdc_challenge.db")

DEFAULT_EMB_PATH = dataset_embeddings_path
TOP_M = int(10)

@timer_wrap
def load_embeddings_parquet(path: str = DEFAULT_EMB_PATH) -> Tuple[List[str], np.ndarray]:
    logger.info(f"Loading embeddings parquet from: {path}")
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as e:
        logger.error(f"Embeddings parquet not found at {path}", exc_info=e)
        raise
    except Exception as e:
        logger.error(f"Failed reading parquet at {path}", exc_info=e)
        raise
    # first column should be dataset_id
    if df.columns[0] != "dataset_id":
        logger.error("Invalid parquet format: first column is not 'dataset_id'")
        raise ValueError("First column of embeddings parquet must be 'dataset_id'")
    dataset_ids = df["dataset_id"].astype(str).tolist()
    X = df.drop(columns=["dataset_id"]).to_numpy(dtype=float, copy=False)
    logger.info(f"Loaded embeddings: {len(dataset_ids)} rows, dim={X.shape[1] if X.ndim==2 else 'NA'}")
    return dataset_ids, X


@timer_wrap
def add_target_and_neighbors_union(ranked_ids: List[str], target_and_neighbors: List[str], top_k: int) -> List[str]:
    """Return union of retrieval results and explicit target+neighbors, preserving order preference."""
    logger.debug(f"Combining ranked_ids(top_k={top_k}) with explicit target+neighbors (n={len(target_and_neighbors)})")
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
    logger.debug(f"Combined selection size: {len(ordered)}")
    return ordered


@timer_wrap
def mask_dataset_ids_in_text(text: str, dataset_ids: List[str], mask_token: str = "<DATASET_ID>") -> str:
    """Robust multi-ID masking with regex escaping (ported for Kaggle)."""
    if not dataset_ids:
        return text
    escaped_ids = [re.escape(ds_id) for ds_id in dataset_ids]
    pattern = r"\b(" + r"|".join(escaped_ids) + r")\b"
    try:
        masked = re.sub(pattern, mask_token, text, flags=re.IGNORECASE)
        return masked
    except re.error as e:
        logger.warning("Regex error while masking dataset ids; returning original text", exc_info=e)
        return text

@timer_wrap
def construct_dataset_from_retrieval_results(
    document_id: str,
    dataset_id: str,
    chunk_ids: List[str],
    id_to_text: Dict[str, str],
) -> Dataset:
    logger.info(f"Constructing Dataset for dataset_id={dataset_id} in document_id={document_id} with {len(chunk_ids)} chunks")
    try:
        text = "\n\n".join(
            mask_dataset_ids_in_text(id_to_text[cid], [dataset_id]) for cid in chunk_ids if cid in id_to_text
        )
    except Exception as e:
        logger.error("Failed to assemble dataset text from chunk ids", exc_info=e)
        text = ""
    try:
        total_tokens = sum(len(t.split()) for t in text.split())
        avg_tokens_per_chunk = np.mean([len(id_to_text[cid].split()) for cid in chunk_ids if cid in id_to_text]) if chunk_ids else 0.0
        if np.isnan(avg_tokens_per_chunk):
            avg_tokens_per_chunk = 0.0
        total_char_length = len(text)
        clean_text_length = len(text)
    except Exception as e:
        logger.warning("Metric computation failed; defaulting to zeros", exc_info=e)
        total_tokens = 0
        avg_tokens_per_chunk = 0.0
        total_char_length = 0
        clean_text_length = 0
    return Dataset(
        dataset_id=dataset_id,
        document_id=document_id,
        total_tokens=int(total_tokens),
        avg_tokens_per_chunk=float(avg_tokens_per_chunk),
        total_char_length=int(total_char_length),
        clean_text_length=int(clean_text_length),
        cluster=None,
        dataset_type=None,
        text=text,
    )

@timer_wrap
def build_dataset_embeddings(datasets: List[Dataset], model, out_path: str = DEFAULT_EMB_PATH) -> Path:
    # model = load_bge_model(model_dir)
    logger.info(f"Building dataset embeddings for {len(datasets)} datasets -> {out_path}")
    texts = [ds.text for ds in datasets]
    try:
        embs = embed_texts(model, texts)
    except Exception as e:
        logger.error("Embedding failed for dataset texts", exc_info=e)
        raise
    if not isinstance(embs, np.ndarray):
        logger.error("embed_texts did not return a numpy array")
        raise ValueError("embed_texts must return a numpy ndarray")
    if embs.shape[0] != len(datasets):
        logger.error(f"Embedding row count mismatch: got {embs.shape[0]}, expected {len(datasets)}")
        raise ValueError("Embedding count does not match number of datasets")
    try:
        df = pd.DataFrame(embs)
        df.insert(0, "dataset_id", [ds.dataset_id for ds in datasets])
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        logger.info(f"Wrote embeddings parquet: {out}")
        return out
    except Exception as e:
        logger.error("Failed to write embeddings parquet", exc_info=e)
        raise

@timer_wrap
def get_query_texts(data_citation: CitationEntity, db_helper):
    """
    Get the query text for a given dataset ID
    """
    ce = data_citation
    logger.info(f"Building query text for dataset_id={ce.data_citation} in document_id={ce.document_id}")
    # Retrieve all chunks for the document
    try:
        chunks = db_helper.get_chunks_by_document_id(ce.document_id)
    except Exception as e:
        logger.error(f"Failed to retrieve chunks for document {ce.document_id}", exc_info=e)
        return "", []
    # Find the chunk containing this dataset citation
    target = None
    for c in chunks:
        ents = c.chunk_metadata.citation_entities or []
        if any(ent.data_citation == ce.data_citation for ent in ents):
            target = c
            break
    if not target:
        logger.warning(f"No chunk found for citation {ce.data_citation} in document {ce.document_id}")
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
    logger.debug(f"Query text length: {len(query_text)}; neighbor_ids={neighbor_ids}")
    return query_text, query_chunk_ids

@timer_wrap
def construct_datasets_pipeline(
    model,
    top_k: int = 5,
    db_path: str = DEFAULT_DUCKDB,
    prototype_top_m: int = TOP_M,
    prototypes: np.ndarray = None,
) -> Tuple[List[Dataset], Path]:
    logger.info(f"Starting construct_datasets_pipeline(db_path={db_path}, top_k={top_k}, prototype_top_m={prototype_top_m})")
    db = get_duckdb_helper(db_path)
    try:
        try:
            citations: List[CitationEntity] = db.get_all_citation_entities()
        except Exception as e:
            logger.error("Failed to load citation entities from DuckDB", exc_info=e)
            raise
        logger.info(f"Loaded {len(citations)} citation entities")

        # Cache per-document chunk text and dense embeddings to avoid recompute
        doc_cache_text: Dict[str, Dict[str, str]] = {}
        doc_cache_dense: Dict[str, Dict[str, np.ndarray]] = {}

        # model = load_bge_model(model_dir)
        all_datasets: List[Dataset] = []

        for ce in citations:
            document_id = ce.document_id
            dataset_id = ce.data_citation

            try:
                query_text, query_chunk_ids = get_query_texts(ce, db)
            except Exception as e:
                logger.warning(f"Failed to build query text for dataset_id={dataset_id}", exc_info=e)
                continue
            if not query_text or not query_chunk_ids:
                logger.warning(f"Skipping dataset_id={dataset_id}: missing query text or chunk ids")
                continue

            if document_id not in doc_cache_text:
                try:
                    chunks = db.get_chunks_by_document_id(document_id)
                    id_to_text = {ch.chunk_id: ch.text for ch in chunks}
                    doc_cache_text[document_id] = id_to_text
                    id_order = list(id_to_text.keys())
                    texts = [id_to_text[cid] for cid in id_order]
                    vecs = embed_texts(model, texts)
                    if not isinstance(vecs, np.ndarray) or vecs.shape[0] != len(id_order):
                        raise ValueError("Chunk embeddings shape mismatch for document cache")
                    doc_cache_dense[document_id] = {cid: v for cid, v in zip(id_order, vecs)}
                    logger.debug(f"Cached {len(id_order)} chunks for document {document_id}")
                except Exception as e:
                    logger.error(f"Failed to cache chunks/embeddings for document {document_id}", exc_info=e)
                    continue

            id_to_text = doc_cache_text[document_id]
            id_to_dense = doc_cache_dense[document_id]

            try:
                qvec = embed_texts(model, [query_text])[0]
            except Exception as e:
                logger.warning(f"Failed to embed query text for dataset_id={dataset_id}", exc_info=e)
                continue
            # Safely combine prototypes with query vector when provided
            prototypes_input = None
            try:
                if isinstance(prototypes, np.ndarray) and prototypes.size > 0:
                    qrow = qvec.reshape(1, -1)
                    if prototypes.ndim == 1:
                        prototypes_input = np.vstack([prototypes.reshape(1, -1), qrow])
                    elif prototypes.ndim == 2:
                        prototypes_input = np.vstack([prototypes, qrow])
                    else:
                        prototypes_input = qrow
            except Exception as e:
                logger.warning("Could not combine prototypes with query vector; proceeding without prototypes", exc_info=e)
                prototypes_input = None

            try:
                ranked_ids = hybrid_retrieve_with_boost(
                    query_text=query_text,
                    dense_query_vec=qvec,
                    id_to_dense=id_to_dense,
                    id_to_text=id_to_text,
                    boost_cfg=BoostConfig(
                        mmr_top_k=top_k,
                        prototype_top_m=int(max(1, prototype_top_m))
                    ),
                    prototypes=prototypes_input,
                )
            except Exception as e:
                logger.warning(f"Retrieval failed for dataset_id={dataset_id}", exc_info=e)
                continue

            selected_ids = add_target_and_neighbors_union(ranked_ids, query_chunk_ids, top_k)
            try:
                ds = construct_dataset_from_retrieval_results(document_id, dataset_id, selected_ids, id_to_text)
                all_datasets.append(ds)
            except Exception as e:
                logger.warning(f"Failed to construct dataset object for dataset_id={dataset_id}", exc_info=e)
                continue

        logger.info(f"Upserting {len(all_datasets)} datasets into DuckDB")
        try:
            db.bulk_upsert_datasets(all_datasets)
        except Exception as e:
            logger.error("Failed to upsert datasets into DuckDB", exc_info=e)
            raise
        out_file = build_dataset_embeddings(all_datasets, model=model)
        logger.info(f"Pipeline completed. Embeddings at: {out_file}")
        return all_datasets, out_file
    finally:
        db.close()


__all__ = [
    "construct_datasets_pipeline",
]


