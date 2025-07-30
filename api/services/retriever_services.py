# -*- coding: utf-8 -*-
"""
ChromaDB + DuckDB Hybrid Retrieval System
==========================================

Retrieves top-K chunks using ChromaDB for similarity search and DuckDB for data storage.
Implements fusion scoring with symbolic boosting for biomedical data citation content.

Key Features:
- ChromaDB similarity search for chunk IDs
- DuckDB storage for full Chunk object retrieval  
- Symbolic boosting for data citation keywords
- Configurable fusion scoring
- Integration with existing project infrastructure

Usage:
------
```python
from api.duckdb_utils import DuckDBHelper
from src.retriever import retrieve_top_chunks

# Initialize DuckDB helper
db_helper = DuckDBHelper("artifacts/mdc_challenge.db")

# Retrieve chunks
chunks = retrieve_top_chunks(
    query_texts=["proteomics dataset repository"],
    collection_name="document_123", 
    k=5,
    db_helper=db_helper
)
```
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
import chromadb
# Add heap queue for bounding candidate list
import heapq
# Use numpy for embedding arrays
import numpy as np
# from llama_index.embeddings.openai import OpenAIEmbedding

# For offline embeddings (fallback)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.models import Chunk
from src.helpers import initialize_logging, timer_wrap, preprocess_text
from api.utils.duckdb_utils import get_duckdb_helper
import threading
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.models import RetrievalResult, BatchRetrievalResult

# Thread-local retriever to avoid multiple ChromaDB initializations per thread
_thread_local = threading.local()
# Shared cache of Chroma collections
_chroma_cache: dict[str, chromadb.Collection] = {}
# Lock to guard shared Chroma collection cache
_chroma_cache_lock = threading.Lock()

def get_retriever(cfg_path, collection_name, symbolic_boost, use_fusion_scoring):
    if not hasattr(_thread_local, "retriever"):
        cfg = _load_cfg(cfg_path)
        _thread_local.retriever = ChromaRetriever(
            cfg,
            collection_name,
            symbolic_boost=symbolic_boost,
            use_fusion_scoring=use_fusion_scoring
        )
    return _thread_local.retriever

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(log_file=filename)

DEFAULT_CACHE_DIR = Path(
    os.path.join(project_root, "offline_models")
)
# Lazy-load embedder to avoid repeated heavy loads
_embedder_lock = threading.Lock()
_EMBEDDER: Optional[SentenceTransformer] = None

def _get_embedder(model_path: str) -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        with _embedder_lock:
            if _EMBEDDER is None:
                _EMBEDDER = SentenceTransformer(model_path)
    return _EMBEDDER

# ---------------------------------------------------------------------------
# Configuration and Setup
# ---------------------------------------------------------------------------

@timer_wrap
def _load_cfg(cfg_path: os.PathLike | None = None) -> Dict[str, Any]:
    """Load YAML config matching existing semantic_chunking.py pattern."""
    default_path = Path("configs/chunking.yaml")
    path = Path(cfg_path or default_path).expanduser()
    
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
        
    with path.open("r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}
    
    if "vector_store" not in cfg:
        raise KeyError("YAML must define a 'vector_store' section")
    return cfg


@timer_wrap
def _get_chroma_collection(cfg: Dict[str, Any], collection_name: str):
    """Return a *shared* Collection instance (thread-safe)"""
    # Guard shared cache with a lock
    with _chroma_cache_lock:
        if collection_name in _chroma_cache:
            return _chroma_cache[collection_name]

        chroma_path = (Path(__file__).resolve().parents[2] / cfg["vector_store"].get("path", "./local_chroma")).expanduser()
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        _chroma_cache[collection_name] = client.get_or_create_collection(collection_name)
        return _chroma_cache[collection_name]


# ---------------------------------------------------------------------------
# Embedding Functions (Adapted from semantic_chunking.py)
# ---------------------------------------------------------------------------

@timer_wrap
def _embed_text(texts: List[str], model_name: Optional[str] = None, batch_size: int = 100) -> np.ndarray:
    embedder = _get_embedder(str(model_name or DEFAULT_CACHE_DIR))
    embeddings = embedder.encode(texts, convert_to_numpy=True, batch_size=batch_size)
    # Ensure embeddings array is not empty
    if embeddings is None or (isinstance(embeddings, np.ndarray) and embeddings.size == 0):
        logger.error("No embeddings generated for %d texts", len(texts))
        return []
    return embeddings


# ---------------------------------------------------------------------------
# Data Citation Entity Detection
# ---------------------------------------------------------------------------

# Data citation keywords focused on repository access and data usage patterns
DATA_CITATION_KEYWORDS = {
    # Data access verbs
    "access_verbs": [
        "deposited", "downloaded", "accessed", "retrieved", "obtained", 
        "collected", "gathered", "sourced", "extracted", "acquired"
    ],
    
    # Repository and database terms
    "repositories": [
        "repository", "database", "databank", "archive", "portal",
        "ncbi", "uniprot", "ensembl", "genbank", "embl", "ddbj",
        "geo", "arrayexpress", "sra", "ena", "pride", "metabolights",
        "figshare", "zenodo", "dryad", "mendeley", "osf"
    ],
    
    # Data identifiers and accessions
    "identifiers": [
        "accession", "identifier", "doi", "pmid", "pmcid", "gse", 
        "srp", "srr", "prj", "sam", "biosample", "bioproject",
        "accession number", "dataset id", "study id"
    ],
    
    # Dataset and data terms
    "data_terms": [
        "dataset", "data", "supplementary data", "raw data", "processed data",
        "transcriptome", "proteome", "genome", "metabolome", "microarray",
        "rna-seq", "chip-seq", "mass spectrometry", "sequencing data"
    ]
}

@timer_wrap
def _detect_data_citation_entities(meta: Dict[str, Any], chunk_text: str = None) -> Dict[str, Any]:
    """
    Detect data citation indicators in chunk metadata and text.
    
    Args:
        meta: ChromaDB metadata dictionary
        chunk_text: Optional chunk text content
        
    Returns:
        Dictionary with detection results and boost score
    """
    detection_results = {
        "has_citation_entities": False,
        "has_access_verbs": False, 
        "has_repositories": False,
        "has_identifiers": False,
        "has_data_terms": False,
        "boost_score": 0.0,
        "matched_keywords": []
    }
    
    # Check existing citation entities in metadata
    if meta.get("citation_entities"):
        detection_results["has_citation_entities"] = True
        detection_results["boost_score"] += 0.2  # Strong boost for existing citations
    
    # Analyze chunk text if available
    if chunk_text:
        text_lower = preprocess_text(chunk_text)
        
        # Check each keyword category
        for category, keywords in DATA_CITATION_KEYWORDS.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                detection_results[f"has_{category}"] = True
                detection_results["matched_keywords"].extend(found_keywords)
                
                # Add category-specific boost scores
                if category == "access_verbs":
                    detection_results["boost_score"] += 0.15
                elif category == "repositories": 
                    detection_results["boost_score"] += 0.2
                elif category == "identifiers":
                    detection_results["boost_score"] += 0.15
                elif category == "data_terms":
                    detection_results["boost_score"] += 0.1
    
    # Cap maximum boost to prevent over-boosting
    detection_results["boost_score"] = min(detection_results["boost_score"], 0.5)
    
    return detection_results


# ---------------------------------------------------------------------------
# Core Retriever Class
# ---------------------------------------------------------------------------

class ChromaRetriever:
    """
    Hybrid retriever using ChromaDB for similarity search and DuckDB for data storage.
    """
    
    def __init__(
        self, 
        cfg: Dict[str, Any], 
        collection_name: str, 
        symbolic_boost: float = 0.15,
        use_fusion_scoring: bool = True,
        model_name: str = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            cfg: Configuration dictionary
            collection_name: ChromaDB collection name
            db_helper: DuckDB helper instance for chunk retrieval
            symbolic_boost: Base symbolic boost value
            use_fusion_scoring: Whether to enable fusion scoring
        """
        self.cfg = cfg
        self.collection = _get_chroma_collection(cfg, collection_name)
        self.model_name = model_name
        self.db_helper = get_duckdb_helper(os.path.join(project_root, "artifacts", "mdc_challenge.db"))
        
        # Fusion scoring parameters
        self.symbolic_boost = symbolic_boost
        self.use_fusion_scoring = use_fusion_scoring
        
        logger.info(
            "Initialized ChromaRetriever: collection=%s, model=%s, fusion_scoring=%s",
            collection_name, self.model_name, use_fusion_scoring
        )
    
    def retrieve_chunks(self, query_texts: List[str], query_embeddings: Optional[np.ndarray] = None, k: int = 3, doc_id_filter: Optional[str] = None) -> List[Chunk]:
        """
        Retrieve top-K chunks using ChromaDB search + DuckDB data retrieval.
        
        Args:
            query_texts: List of query strings
            k: Number of chunks to retrieve
            
        Returns:
            List of Chunk objects with retrieval scores
        """
        if not query_texts and query_embeddings is None:
            logger.warning("No query texts provided")
            return []
            
        logger.info("Retrieving top-%d chunks for %d queries", k, len(query_texts))
        
        # Step 1: Generate embeddings for queries
        logger.info("Generating embeddings for queries")
        if query_embeddings is None:
            query_embeddings = _embed_text(query_texts)

        # Step 2: Search ChromaDB for chunk IDs and metadata
        heap: list[tuple[float, str]] = []   # (score, chunk_id) min-heap
        seen: dict[str, float] = {}          # chunk_id -> best_score
        
        def try_push(score: float, cid: str, k: int):
            """Push chunk to heap only if new or has better score, maintaining uniqueness."""
            # Only push if this chunk is new *or* beats its previous best score
            if cid not in seen or score > seen[cid]:
                seen[cid] = score
                heapq.heappush(heap, (score, cid))
                if len(heap) > k:
                    # pop until heap size is k *and* remove any stale entries
                    while heap and seen[heap[0][1]] != heap[0][0]:
                        heapq.heappop(heap)
                    if len(heap) > k:
                        popped_score, popped_id = heapq.heappop(heap)
                        # make sure the popped element is truly discarded
                        if seen.get(popped_id) == popped_score:
                            del seen[popped_id]
        
        for i, q_emb in enumerate(query_embeddings):
            logger.info("Processing query %d/%d", i+1, len(query_embeddings))
            try:
                # Build query parameters, including an optional document_id metadata filter
                query_params = {
                    "query_embeddings": [q_emb.tolist()],
                    "n_results": k * 5,  # Fetch extra for deduplication
                    "include": ["distances", "metadatas"],
                }
                if doc_id_filter:
                    query_params["where"] = {"document_id": doc_id_filter}
                results = self.collection.query(**query_params)
                
                for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
                    logger.info("Calculating score for chunk %s", meta["chunk_id"])
                    score = self._fuse_score(dist, meta) if self.use_fusion_scoring else 1.0 - dist
                    cid = meta["chunk_id"]
                    # Push if new or better, maintaining uniqueness
                    try_push(score, cid, k)
                    
            except Exception as e:
                logger.error("Error querying ChromaDB for embedding %d: %s", i, e)
                continue
        
        if not heap:
            logger.warning("No results from ChromaDB queries")
            return []

        top_chunk_data = heapq.nlargest(k, heap)  # Returns [(score, chunk_id), ...]
        
        # Step 4: Retrieve full Chunk objects from DuckDB
        chunk_objects = []
        
        if not self.db_helper:
            logger.error("No DuckDB helper provided - cannot retrieve chunk data")
            return []
        
        for score, chunk_id in top_chunk_data:
            try:
                chunk = self.db_helper.get_chunks_by_chunk_ids(chunk_id)
                if chunk:
                    chunk.score = score  # Set retrieval score
                    chunk_objects.append(chunk)
                    logger.info("Retrieved chunk %s with score %.3f", chunk_id, score)
                else:
                    logger.warning("Chunk %s not found in DuckDB", chunk_id)
            except Exception as e:
                logger.error("Error retrieving chunk %s from DuckDB: %s", chunk_id, e)
        
        logger.info("Successfully retrieved %d chunks via ChromaDB->DuckDB", len(chunk_objects))
        return chunk_objects
    
    def _fuse_score(self, cosine_dist: float, meta: Dict[str, Any]) -> float:
        """
        Combine semantic similarity with symbolic boosting for data citations.
        
        Args:
            cosine_dist: Cosine distance from ChromaDB
            meta: Chunk metadata from ChromaDB
            
        Returns:
            Fused similarity score
        """
        # Base semantic similarity
        semantic_score = 1.0 - cosine_dist
        
        # Detect data citation entities (without chunk text for now)
        detection = _detect_data_citation_entities(meta)
        symbolic_boost = detection["boost_score"] * self.symbolic_boost
        
        # Log significant boosts
        if symbolic_boost > 0.1:
            logger.debug(
                "Chunk %s: semantic=%.3f, symbolic=+%.3f, keywords=%s",
                meta.get("chunk_id", "unknown"),
                semantic_score,
                symbolic_boost, 
                detection["matched_keywords"][:3]  # Show first 3 keywords
            )
        
        return semantic_score + symbolic_boost
    
    def retrieve_chunks_with_text_analysis(
        self,
        query_texts: List[str], 
        k: int = 4,
        analyze_text: bool = True,
        doc_id_filter: Optional[str] = None,
        query_embeddings: Optional[np.ndarray] = None,
    ) -> List[Chunk]:
        """
        Enhanced retrieval with full chunk text analysis for better entity detection.
        
        Args:
            query_texts: List of query strings
            k: Number of chunks to retrieve
            analyze_text: Whether to perform text-based entity detection
            
        Returns:
            List of Chunk objects with enhanced scoring
        """
        if not analyze_text:
            return self.retrieve_chunks(query_texts, k, doc_id_filter=doc_id_filter, query_embeddings=query_embeddings)
        
        # First get initial candidates (more than k)
        initial_candidates = self.retrieve_chunks(query_texts, k * 2, doc_id_filter=doc_id_filter, query_embeddings=query_embeddings)
        
        if not initial_candidates:
            return []
        
        # Re-score with full text analysis
        rescored_chunks = []
        
        for chunk in initial_candidates:
            # Analyze full chunk text for better entity detection
            detection = _detect_data_citation_entities({}, chunk.text)
            
            # Recalculate score with text-based boost
            base_score = chunk.score or 0.0
            text_boost = detection["boost_score"] * self.symbolic_boost
            
            chunk.score = base_score + text_boost
            rescored_chunks.append(chunk)
        
        # Sort by new scores and return top-k
        rescored_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        logger.info("Re-scored %d chunks with text analysis", len(rescored_chunks))
        return rescored_chunks[:k]


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def retrieve_top_chunks(
    query_texts: List[str],
    collection_name: str,
    k: int = 4,
    cfg_path: os.PathLike | None = Path(os.path.join(project_root, "configs", "chunking.yaml")),
    symbolic_boost: float = 0.15,
    use_fusion_scoring: bool = True,
    analyze_chunk_text: bool = False,
    doc_id_filter: Optional[str] = None,
    query_embeddings: Optional[np.ndarray] = None
) -> List[Chunk]:
    """
    Retrieve top-K chunks using ChromaDB similarity search + DuckDB storage.
    
    Args:
        query_texts: List of query strings to search for
        collection_name: ChromaDB collection name (usually document ID)
        k: Number of chunks to retrieve
        db_helper: DuckDB helper instance for chunk retrieval
        cfg_path: Path to chunking config file (defaults to configs/chunking.yaml)
        symbolic_boost: Multiplier for symbolic boosting (0.0 to disable)
        use_fusion_scoring: Whether to enable fusion scoring with entity boosting
        analyze_chunk_text: Whether to perform enhanced text analysis (slower but more accurate)
        doc_id_filter: Optional document ID to filter chunks by
    Returns:
        List of Chunk objects with populated similarity scores
        
    Example:
        ```python
        from api.duckdb_utils import DuckDBHelper
        
        db_helper = DuckDBHelper("artifacts/mdc_challenge.db")
        
        chunks = retrieve_top_chunks(
            query_texts=["proteomics mass spectrometry dataset"],
            collection_name="PMC123456",
            k=5,
            db_helper=db_helper,
            use_fusion_scoring=True
        )
        
        for chunk in chunks:
            print(f"Chunk {chunk.chunk_id}: score={chunk.score:.3f}")
        ```
    """
    if not query_texts:
        logger.warning("No query texts provided to retrieve_top_chunks")
        return []
    
    try:
        # Load configuration
        cfg = _load_cfg(cfg_path)
        
        # Initialize retriever
        retriever = ChromaRetriever(
            cfg=cfg,
            collection_name=collection_name,
            symbolic_boost=symbolic_boost,
            use_fusion_scoring=use_fusion_scoring
        )
        
        # Retrieve chunks
        if analyze_chunk_text:
            logger.info("Retrieving chunks with text analysis")
            chunks = retriever.retrieve_chunks_with_text_analysis(query_texts, k, doc_id_filter=doc_id_filter, query_embeddings=query_embeddings)
        else:
            logger.info("Retrieving chunks without text analysis")
            chunks = retriever.retrieve_chunks(query_texts, k, doc_id_filter=doc_id_filter, query_embeddings=query_embeddings)
        
        logger.info("Retrieved %d chunks for collection %s", len(chunks), collection_name)
        return chunks
        
    except Exception as e:
        logger.error("Error in retrieve_top_chunks: %s", e)
        return []

# def batched(seq, n=100):
#     for i in range(0, len(seq), n):
#         yield seq[i:i+n]

@timer_wrap
def batch_retrieve_top_chunks(
    query_texts: Dict[str, List[str]],
    max_workers: int = 1,
    collection_name: str = "mdc_training_data",
    k: int = 3,
    cfg_path: os.PathLike | None = Path(os.path.join(project_root, "configs", "chunking.yaml")),
    symbolic_boost: float = 0.15,
    use_fusion_scoring: bool = True,
    analyze_chunk_text: bool = False,
    doc_id_map: Dict[str, str] = {}
) -> BatchRetrievalResult:
    """
    Batch retrieve top-k chunks in parallel.
    Args:
        query_texts: Dictionary mapping identifier to list of query texts
        max_workers: Number of parallel worker threads
        collection_name: ChromaDB collection name
        k: Number of chunks to retrieve
        cfg_path: Path to chunking config file (defaults to configs/chunking.yaml)
        symbolic_boost: Multiplier for symbolic boosting (0.0 to disable)
        use_fusion_scoring: Whether to enable fusion scoring with entity boosting
        analyze_chunk_text: Whether to perform enhanced text analysis (slower but more accurate)
    """
    logger.info(f"---Starting Batch retrieval of top {k} chunks for {len(query_texts)} dataset IDs across {len(set(doc_id_map.values()))} documents---")
    if len(doc_id_map) < 1:
        logger.warning("No document IDs mapped to Dataset IDs. Retrieval will be performed without document ID filtering.")
    else:
        logger.info("Retrieving top %d chunks for %d dataset IDs across %d documents", k, len(doc_id_map), len(set(doc_id_map.values())))
    
    # Phase 1: Generate embeddings for queries in bulk
    flat_queries = [q for qs in query_texts.values() for q in qs]
    # sanity check since number of flat queries should be the same as the length of the query_texts dictionary
    if len(flat_queries) != len(query_texts):
        logger.error("Number of flat queries (%d) does not match the number of query texts (%d)", len(flat_queries), len(query_texts))
        raise ValueError("Number of flat queries does not match the number of query texts")
    embeddings = _embed_text(flat_queries)
    # assumes that the order of the queries is the same as the order of the embeddings
    query_embeddings_map = {}
    for row_idx, (identifier, _) in enumerate(query_texts.items()):
        query_embeddings_map[identifier] = embeddings[row_idx]

    # Phase 2: parallel retrieval per identifier
    def _retrieve_single(identifier: str, queries: List[str], doc_id_filter: Optional[str] = None, query_embeddings: Optional[List[List[float]]] = None) -> RetrievalResult:
        start = time.time()
        try:
            logger.info(f"Retrieving top {k} chunks for {identifier}")
            retriever = get_retriever(cfg_path, collection_name, symbolic_boost, use_fusion_scoring)
            # Use the thread-local retriever instance to avoid re-initialization
            if analyze_chunk_text:
                # perform text-analysis enhanced retrieval
                chunks = retriever.retrieve_chunks_with_text_analysis(queries, k, analyze_text=True, doc_id_filter=doc_id_filter, query_embeddings=query_embeddings)
            else:
                chunks = retriever.retrieve_chunks(queries, k, doc_id_filter=doc_id_filter, query_embeddings=query_embeddings)
            if len(chunks) == 0:
                elapsed = time.time() - start
                logger.warning(f"No chunks retrieved for {identifier}")
                return RetrievalResult(
                    collection_name=collection_name,
                    success=False,
                    error=f"No chunks retrieved for {identifier}",
                    k=k,
                    chunk_ids=[],
                    median_score=0,
                    max_score=0,
                    retrieval_time=elapsed,
                )
            scores = [c.score or 0.0 for c in chunks]
            elapsed = time.time() - start
            logger.info(f"Retrieved {len(chunks)} chunks for {identifier} in {elapsed:.2f} seconds")
            return RetrievalResult(
                collection_name=collection_name,
                success=True,
                error=None,
                k=k,
                chunk_ids=[c.chunk_id for c in chunks],
                median_score=statistics.median(scores) if scores else None,
                max_score=max(scores) if scores else None,
                retrieval_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            return RetrievalResult(
                collection_name=collection_name,
                success=False,
                error=str(e),
                k=k,
                chunk_ids=[],
                median_score=0,
                max_score=0,
                retrieval_time=elapsed,
            )

    workers = min(max_workers, len(query_texts)) if query_texts else 1
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(_retrieve_single, identifier, queries, doc_id_filter=doc_id_map.get(identifier, None), query_embeddings=query_embeddings_map.get(identifier, None)): identifier
            for identifier, queries in query_texts.items()
        }
        results = [fut.result() for fut in as_completed(futures)]

    # Phase 2: aggregate results into BatchRetrievalResult
    total = len(results)
    failed = sum(1 for r in results if not r.success)
    times = [r.retrieval_time or 0.0 for r in results]
    chunk_ids_map = {id_: r.chunk_ids for id_, r in zip(query_texts.keys(), results)}
    # Calculate batch-level statistics, handling empty score lists
    valid_median_scores = [r.median_score for r in results if r.median_score is not None and r.median_score > 0]
    valid_max_scores = [r.max_score for r in results if r.max_score is not None and r.max_score > 0]
    
    batch_result = BatchRetrievalResult(
        collection_name=collection_name,
        total_queries=total,
        success=(failed == 0),
        error=None,
        k=k,
        chunk_ids=chunk_ids_map,
        median_score=statistics.median(valid_median_scores) if valid_median_scores else 0.0,
        max_score=max(valid_max_scores) if valid_max_scores else 0.0,
        avg_retrieval_time=sum(times)/total if total else None,
        max_retrieval_time=max(times) if times else None,
        total_failed_queries=failed,
    )
    return batch_result


# ---------------------------------------------------------------------------
# Demo and Testing Functions
# ---------------------------------------------------------------------------

def demo_retrieval(
    collection_name: str = "test_collection",
    db_path: str = "artifacts/mdc_challenge.db"
) -> List[Chunk]:
    """
    Demo function to test retrieval system (pending until actually run).
    
    Args:
        collection_name: ChromaDB collection to search
        db_path: Path to DuckDB database
        
    Returns:
        List of retrieved chunks
    """
    logger.info("=== ChromaDB + DuckDB Retrieval Demo ===")
    
    try:
        
        # Example queries focused on data citations
        demo_queries = [
            "proteomics mass spectrometry dataset repository",
            "RNA sequencing data deposited NCBI GEO", 
            "supplementary data downloaded accessed",
            "gene expression dataset accession number"
        ]
        
        logger.info("Demo queries: %s", demo_queries)
        
        # Test basic retrieval
        chunks = retrieve_top_chunks(
            query_texts=demo_queries,
            collection_name=collection_name,
            k=5,
            use_fusion_scoring=True,
            analyze_chunk_text=False
        )
        
        # Display results
        logger.info("Retrieved %d chunks:", len(chunks))
        for i, chunk in enumerate(chunks, 1):
            logger.info(
                "%d. Chunk %s (doc: %s): score=%.3f, tokens=%d",
                i, chunk.chunk_id, chunk.document_id, 
                chunk.score or 0.0, 
                chunk.chunk_metadata.token_count if chunk.chunk_metadata else 0
            )
        
        # Test enhanced text analysis
        if chunks:
            logger.info("\n=== Testing Enhanced Text Analysis ===")
            enhanced_chunks = retrieve_top_chunks(
                query_texts=demo_queries[:2],  # Use fewer queries for enhanced analysis
                collection_name=collection_name,
                k=3,
                analyze_chunk_text=True
            )
            
            logger.info("Enhanced analysis retrieved %d chunks", len(enhanced_chunks))
        
        return chunks
        
    except Exception as e:
        logger.error("Demo retrieval failed: %s", e)
        return []


def analyze_collection_entities(
    collection_name: str,
    sample_size: int = 50
) -> Dict[str, Any]:
    """
    Analyze a ChromaDB collection for data citation entity distribution.
    
    Args:
        collection_name: Collection to analyze
        db_helper: DuckDB helper instance
        sample_size: Number of chunks to sample for analysis
        
    Returns:
        Analysis results dictionary
    """
    logger.info("Analyzing collection '%s' for entity distribution", collection_name)
    
    try:
        db_helper = get_duckdb_helper(os.path.join(project_root, "artifacts", "mdc_challenge.db"))
        cfg = _load_cfg(os.path.join(project_root, "configs", "chunking.yaml"))
        _, collection = _get_chroma_collection(cfg, collection_name)
        
        # Get sample of chunks
        results = collection.get(
            limit=sample_size,
            include=["metadatas"]
        )
        
        if not results["metadatas"]:
            return {"error": "No metadata found in collection"}
        
        # Analyze entity indicators
        analysis = {
            "total_chunks_analyzed": len(results["metadatas"]),
            "chunks_with_citations": 0,
            "keyword_matches": {category: 0 for category in DATA_CITATION_KEYWORDS.keys()},
            "common_keywords": [],
            "boost_score_distribution": []
        }
        
        all_keywords = []
        
        for meta in results["metadatas"]:
            # Check citation entities
            if meta.get("citation_entities"):
                analysis["chunks_with_citations"] += 1
            
            # Get chunk text for keyword analysis if available
            chunk_id = meta.get("chunk_id")
            if chunk_id and db_helper:
                try:
                    chunk = db_helper.get_chunks_by_chunk_ids(chunk_id)
                    if chunk[0] and chunk[0].text:
                        detection = _detect_data_citation_entities(meta, chunk[0].text)
                        analysis["boost_score_distribution"].append(detection["boost_score"])
                        all_keywords.extend(detection["matched_keywords"])
                        
                        # Count keyword categories
                        for category in DATA_CITATION_KEYWORDS.keys():
                            if detection.get(f"has_{category}", False):
                                analysis["keyword_matches"][category] += 1
                                
                except Exception as e:
                    logger.debug("Error analyzing chunk %s: %s", chunk_id, e)
        
        # Find most common keywords
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        analysis["common_keywords"] = keyword_counts.most_common(10)
        
        logger.info("Collection analysis complete: %d/%d chunks have citations", 
                   analysis["chunks_with_citations"], analysis["total_chunks_analyzed"])
        
        return analysis
        
    except Exception as e:
        logger.error("Error analyzing collection: %s", e)
        return {"error": str(e)}
    finally:
        db_helper.close()


if __name__ == "__main__":
    """Run demo if script is executed directly."""
    demo_retrieval()
