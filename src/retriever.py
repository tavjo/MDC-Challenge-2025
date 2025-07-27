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
from typing import Any, Dict, List, Sequence, Optional, Set

import yaml
import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding

# For offline embeddings (fallback)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models import Chunk, ChunkMetadata, CitationEntity
from src.helpers import initialize_logging
from api.utils.duckdb_utils import get_duckdb_helper

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(log_file=filename)

# ---------------------------------------------------------------------------
# Configuration and Setup
# ---------------------------------------------------------------------------

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


def _get_chroma_collection(cfg: Dict[str, Any], collection_name: str):
    """Get ChromaDB collection using existing project pattern."""
    chroma_path = Path(cfg["vector_store"].get("path", "./local_chroma")).expanduser()
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


# ---------------------------------------------------------------------------
# Embedding Functions (Adapted from semantic_chunking.py)
# ---------------------------------------------------------------------------

def _build_embedder(model_name: str, cfg: dict = None):
    """
    Return an embedding instance (OpenAI or offline) based on model name.
    Matches the pattern from semantic_chunking.py.

    Args:
        model_name: Model name. If starts with 'offline:' or is in offline models list,
                   uses SentenceTransformer. Otherwise uses OpenAI.
        cfg: Configuration dictionary with offline model settings.
    
    Returns:
        Embedding instance compatible with llama_index interface.
    """
    # List of known offline models
    offline_models = [
        "bge-small-en-v1.5",
        "all-MiniLM-L6-v2", 
        "all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ]
    
    # Get offline model configuration
    offline_cfg = cfg.get("offline_model", {}) if cfg else {}
    cache_dir = offline_cfg.get("cache_dir", "./offline_models")
    
    # Check if this is an offline model
    if model_name.startswith("offline:"):
        # Remove offline: prefix
        actual_model = model_name[8:]
        return _create_offline_embedder(actual_model, cache_dir)
    elif model_name in offline_models:
        return _create_offline_embedder(model_name, cache_dir)
    else:
        # Use OpenAI embedding
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found – please set it before calling this helper."
            )

        logger.info("▸ Loading OpenAI embedding model %s", model_name)
        return OpenAIEmbedding(model=model_name)


def _create_offline_embedder(model_name: str, cache_dir: str):
    """Create offline embedder using SentenceTransformer."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "SentenceTransformers not available. Install with: pip install sentence-transformers"
        )
    
    logger.info("▸ Loading offline embedding model %s", model_name)
    
    # Create a wrapper to match llama_index interface
    class OfflineEmbedder:
        def __init__(self, model_name: str, cache_dir: str):
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu")
        
        def get_text_embedding(self, text: str) -> List[float]:
            """Get embedding for a single text (matches OpenAIEmbedding interface)."""
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            return embedding.tolist()
        
        def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            """Get embeddings for multiple texts."""
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
    
    return OfflineEmbedder(model_name, cache_dir)


# @functools.lru_cache(maxsize=1)
def _load_embedder(model_name: str, cfg: dict = None):
    """Load embedding model with caching."""
    logger.info("Loading embedding model: %s", model_name)
    return _build_embedder(model_name, cfg)


def _embed_text(texts: Sequence[str], model_name: str, cfg: dict = None) -> List[List[float]]:
    """
    Generate embeddings for text queries using OpenAI or offline models.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the embedding model
        cfg: Configuration dictionary
        
    Returns:
        List of embedding vectors
    """
    embedder = _load_embedder(model_name, cfg)
    
    # Handle both OpenAI and offline embedders
    if hasattr(embedder, 'get_text_embeddings'):
        # Offline embedder with batch method
        return embedder.get_text_embeddings(list(texts))
    else:
        # OpenAI embedder - process one by one
        embeddings = []
        for text in texts:
            embedding = embedder.get_text_embedding(text)
            embeddings.append(embedding)
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
        text_lower = chunk_text.lower()
        
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
        model_name: str = "offline:bge-small-en-v1.5"
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
        _, self.collection = _get_chroma_collection(cfg, collection_name)
        self.model_name = model_name
        self.db_helper = get_duckdb_helper(os.path.join(project_root, "artifacts", "mdc_challenge.db"))
        
        # Fusion scoring parameters
        self.symbolic_boost = symbolic_boost
        self.use_fusion_scoring = use_fusion_scoring
        
        logger.info(
            "Initialized ChromaRetriever: collection=%s, model=%s, fusion_scoring=%s",
            collection_name, self.model_name, use_fusion_scoring
        )
    
    def retrieve_chunks(self, query_texts: List[str], k: int = 4) -> List[Chunk]:
        """
        Retrieve top-K chunks using ChromaDB search + DuckDB data retrieval.
        
        Args:
            query_texts: List of query strings
            k: Number of chunks to retrieve
            
        Returns:
            List of Chunk objects with retrieval scores
        """
        if not query_texts:
            logger.warning("No query texts provided")
            return []
            
        logger.info("Retrieving top-%d chunks for %d queries", k, len(query_texts))
        
        # Step 1: Generate embeddings for queries
        logger.info("Generating embeddings for queries")
        query_embeddings = _embed_text(query_texts, self.model_name, self.cfg)
        
        # Step 2: Search ChromaDB for chunk IDs and metadata
        candidate_results = []
        
        for i, q_emb in enumerate(query_embeddings):
            logger.info("Processing query %d/%d", i+1, len(query_embeddings))
            
            try:
                results = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=k * 3,  # Fetch extra for deduplication
                    include=["distances", "metadatas"]
                )
                
                for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
                    logger.info("Calculating score for chunk %s", meta["chunk_id"])
                    # Apply fusion scoring if enabled
                    if self.use_fusion_scoring:
                        fused_score = self._fuse_score(dist, meta)
                    else:
                        fused_score = 1.0 - dist  # Simple cosine similarity
                        
                    candidate_results.append((fused_score, meta["chunk_id"]))
                    
            except Exception as e:
                logger.error("Error querying ChromaDB for embedding %d: %s", i, e)
                continue
        
        if not candidate_results:
            logger.warning("No results from ChromaDB queries")
            return []
        
        # Step 3: Deduplicate and select top-K chunk IDs
        seen_chunks: Set[str] = set()
        top_chunk_data = []
        
        # Sort by fused score (descending)
        for score, chunk_id in sorted(candidate_results, key=lambda x: x[0], reverse=True):
            if chunk_id not in seen_chunks:
                top_chunk_data.append((chunk_id, score))
                seen_chunks.add(chunk_id)
            
            if len(top_chunk_data) >= k:
                break
        
        logger.info("Selected %d unique chunks from ChromaDB results", len(top_chunk_data))
        
        # Step 4: Retrieve full Chunk objects from DuckDB
        chunk_objects = []
        
        if not self.db_helper:
            logger.error("No DuckDB helper provided - cannot retrieve chunk data")
            return []
        
        for chunk_id, score in top_chunk_data:
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
        analyze_text: bool = True
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
            return self.retrieve_chunks(query_texts, k)
        
        # First get initial candidates (more than k)
        initial_candidates = self.retrieve_chunks(query_texts, k * 2)
        
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
    analyze_chunk_text: bool = False
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
    
    # if not db_helper:
    #     logger.error("DuckDB helper is required for chunk retrieval")
    #     return []
    
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
            chunks = retriever.retrieve_chunks_with_text_analysis(query_texts, k)
        else:
            logger.info("Retrieving chunks without text analysis")
            chunks = retriever.retrieve_chunks(query_texts, k)
        
        logger.info("Retrieved %d chunks for collection %s", len(chunks), collection_name)
        return chunks
        
    except Exception as e:
        logger.error("Error in retrieve_top_chunks: %s", e)
        return []


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
        # Initialize DuckDB helper
        # from api.duckdb_utils import DuckDBHelper
        # db_helper = DuckDBHelper(db_path)
        
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
