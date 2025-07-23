"""
src/tools/semantic_chunking.py
------------------------------
Semantic chunking + OpenAI embeddings + ChromaDB persistence.

This now uses OpenAI’s `text-embedding-3-small` (or any other model you set in *conf/chunking.yaml*)
directly via the OpenAI library, instead of llama_index embeddings.

Prerequisites
~~~~~~~~~~~~~
1. `pip install openai chromadb`   # plus your other deps
2. Export OPENAI_API_KEY=sk-...

Usage example
~~~~~~~~~~~~~
>>> from semantic_chunking import semantic_chunk_text
>>> chunks = semantic_chunk_text(page_text)
>>> save_chunks_to_chroma(chunks, collection_name="my_doc")
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import yaml
import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding
from src.helpers import get_embedding, CustomOpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from dotenv import load_dotenv

load_dotenv()

# For offline embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.helpers import initialize_logging, sliding_window_chunks  # noqa: E402
from src.models import Chunk, ChunkMetadata, CitationEntity

filename = os.path.basename(__file__)
logger = initialize_logging(log_file=filename)

try:
    # Re-use the timer decorator that already exists in your helpers.
    from src.helpers import timer_wrap  # type: ignore
except Exception:  # pragma: no cover – stand-alone fallback
    import functools, time

    def timer_wrap(func):  # pylint: disable=missing-function-docstring
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            logging.getLogger(__name__).info(
                "%s finished in %.2f s", func.__name__, time.time() - start
            )
            return result

        return wrapper

# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULT_CFG_PATH = Path(
    os.getenv("CHUNKING_CONFIG", os.path.join(project_root, "configs/chunking.yaml"))
)


def _load_cfg(cfg_path: Optional[os.PathLike] | None = None) -> dict:
    """Load YAML config and sanity-check required keys."""
    path = Path(cfg_path or DEFAULT_CFG_PATH).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Chunking config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    if cfg.get("splitter", "semantic") != "semantic":
        raise ValueError("cfg['splitter'] must be 'semantic' for this helper")
    return cfg


# ---------------------------------------------------------------------------
# Offline Embedder Class
# ---------------------------------------------------------------------------

class OfflineEmbedder:
    """
    Offline embedding class using SentenceTransformers for local embeddings.
    Compatible with llama_index embedding interface.
    """
    
    def __init__(self, model_name: str = "bge-small-en-v1.5", cache_dir: str = "./offline_models"):
        """Initialize the offline embedder with a SentenceTransformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("▸ Loading offline embedding model %s (cache: %s)", model_name, cache_dir)
        
        # Download and cache model for offline use
        self.model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir))
        
        # Cache the model to ensure it's available offline
        self._cache_model()
    
    def _cache_model(self):
        """Cache the model for offline use."""
        try:
            # Test embedding to ensure model is fully loaded
            test_embedding = self.model.encode("test")
            logger.info("✅ Model %s cached successfully (embedding dim: %d)", 
                       self.model_name, len(test_embedding))
            
            # Save model info to cache directory
            model_info = {
                "model_name": self.model_name,
                "embedding_dim": len(test_embedding),
                "cached_at": str(datetime.now()),
                "cache_dir": str(self.cache_dir)
            }
            
            info_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_info.json"
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
                
        except Exception as e:
            logger.error("❌ Failed to cache model %s: %s", self.model_name, str(e))
            raise
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string."""
        if not text:
            return []
        
        # Clean text for embedding
        text = text.replace("\n", " ").strip()
        
        # Generate embedding
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple text strings."""
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
        
        # Generate embeddings in batch
        embeddings = self.model.encode(cleaned_texts)
        return embeddings.tolist()

    @staticmethod
    def download_model(model_name: str, cache_dir: str = "./offline_models") -> bool:
        """Download and cache a model for offline use.
        
        Args:
            model_name: Name of the SentenceTransformer model to download
            cache_dir: Directory to cache the model
            
        Returns:
            True if successful, False otherwise
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available for model download")
            return False
            
        try:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("▸ Downloading model %s to %s", model_name, cache_dir)
            
            # Download the model
            model = SentenceTransformer(model_name, cache_folder=str(cache_path))
            
            # Test the model
            test_embedding = model.encode("test")
            logger.info("✅ Model %s downloaded successfully (embedding dim: %d)", 
                       model_name, len(test_embedding))
            
            # Save model info
            model_info = {
                "model_name": model_name,
                "embedding_dim": len(test_embedding),
                "downloaded_at": str(datetime.now()),
                "cache_dir": str(cache_path)
            }
            
            info_file = cache_path / f"{model_name.replace('/', '_')}_info.json"
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error("❌ Failed to download model %s: %s", model_name, str(e))
            return False

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_embedder(model_name: str, cfg: dict = None):
    """Return an embedding instance (OpenAI or offline) based on model name.

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
        return OfflineEmbedder(actual_model, cache_dir=cache_dir)
    elif model_name in offline_models:
        return OfflineEmbedder(model_name, cache_dir=cache_dir)
    else:
        # Use OpenAI embedding
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found – please set it before calling this helper."
            )

        logger.info("▸ Loading OpenAI embedding model %s", model_name)
        return OpenAIEmbedding(model=model_name)


def _build_splitter(cfg: dict, embedder) -> SemanticSplitterNodeParser:
    """Instantiate SemanticSplitterNodeParser with config-driven hyper-params."""
    return SemanticSplitterNodeParser(
        embed_model=embedder,
        similarity_threshold=cfg.get("similarity_threshold", 0.75),
        chunk_overlap=cfg.get("overlap_sentences", 2),
        chunk_size=cfg.get("max_tokens", 300),
        min_chunk_size=cfg.get("min_tokens", 80),
    )


def _get_chroma_collection(cfg: dict, collection_name: str):
    """Return (client, collection) tuple for the given YAML settings."""
    chroma_path = Path(cfg["vector_store"].get("path", "./local_chroma")).expanduser()
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@timer_wrap
def semantic_chunk_text(
    text: str,
    cfg_path: Optional[os.PathLike] | None = None,
    model_name: str = "text-embedding-3-small",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[str]:
    """Return a list of semantic chunks for *text* (page or whole document)."""
    if not text:
        return []

    # Load YAML config and allow overrides
    cfg = _load_cfg(cfg_path)
    # Override token-based chunk size if provided
    if chunk_size is not None:
        cfg["max_tokens"] = chunk_size
    # Override sentence-based overlap if provided
    if chunk_overlap is not None:
        cfg["overlap_sentences"] = chunk_overlap

    embedder = _build_embedder(cfg.get("embed_model", model_name))
    splitter = _build_splitter(cfg, embedder)

    logger.info(
        "▸ Splitting text with SemanticSplitterNodeParser (chunk_size=%s, τ=%s)",
        cfg.get("max_tokens", 300),
        cfg.get("similarity_threshold", 0.75),
    )

    # Wrap raw text in LlamaIndex Document for splitting
    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    # Extract text from each split node
    chunks = [node.text for node in nodes]

    logger.info("▸ Created %d semantic chunks", len(chunks))
    return chunks


@staticmethod
def _chunk_obj_to_metadata(chunk_obj: "Chunk") -> dict:
    """Flatten a Chunk instance to a Chroma-friendly metadata dict."""
    cm = chunk_obj.chunk_metadata
    
    # Handle citation_entities (not candidate_entities)
    citation_entities_str = (
        "/".join([ce.to_string() for ce in cm.citation_entities])
        if cm.citation_entities
        else ""
    )

    return {
        "chunk_id": cm.chunk_id,
        "document_id": chunk_obj.document_id,
        "token_count": cm.token_count,
        "previous_chunk_id": cm.previous_chunk_id or "",
        "next_chunk_id": cm.next_chunk_id or "",
        "citation_entities": citation_entities_str,
    }

@timer_wrap
def save_chunk_obj_to_chroma(
    chunk_obj: "Chunk",
    cfg_path: Optional[os.PathLike] | None = None,
    collection_name: str | None = None,
    model_name: str = "text-embedding-3-small"
):
    """Persist a list of Chunk objects (text + metadata) to ChromaDB."""
    if not chunk_obj:
        logger.warning("No chunk objects supplied – nothing to write to Chroma")
        return

    cfg = _load_cfg(cfg_path)
    embedder = _build_embedder(cfg.get("embed_model", model_name))
    _, collection = _get_chroma_collection(
        cfg, collection_name or chunk_obj.document_id
    )

    logger.info("▸ Embedding %d chunks for Chroma upsert")
    documents =chunk_obj.text
    embeddings = embedder.get_text_embedding(documents)
    metadatas = _chunk_obj_to_metadata(chunk_obj)
    ids = chunk_obj.chunk_id

    collection.add(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )
    logger.info(
        "✅ Persisted 1 chunk to Chroma collection '%s'", collection.name
    )

@timer_wrap
def save_chunk_objs_to_chroma(
    chunk_objs: List["Chunk"],
    cfg_path: Optional[os.PathLike] | None = None,
    collection_name: str | None = None,
    model_name: str = "text-embedding-3-small",
    chunk_size: int = 300,
    chunk_overlap: int = 2
):
    """Persist a list of Chunk objects (text + metadata) to ChromaDB."""
    if not chunk_objs:
        logger.warning("No chunk objects supplied – nothing to write to Chroma")
        return

    cfg = _load_cfg(cfg_path)
    embedder = _build_embedder(cfg.get("embed_model", model_name))
    _, collection = _get_chroma_collection(
        cfg, collection_name or chunk_objs[0].document_id
    )

    logger.info("▸ Embedding %d chunks for Chroma upsert", len(chunk_objs))
    documents = [c.text for c in chunk_objs]
    embeddings = [embedder.get_text_embedding(txt) for txt in documents]
    # embeddings = []
    # for txt in documents:
    #     import tiktoken
    #     tok = tiktoken.get_encoding("cl100k_base")
    #     token_count = len(tok.encode(txt))
    #     if token_count > chunk_size:
    #         processed_chunks = sliding_window_chunks(txt, chunk_size, chunk_overlap)
    #         for chunk in processed_chunks:
    #             embeddings.append(embedder.get_text_embedding(chunk))
    #     else:
    #         embeddings.append(embedder.get_text_embedding(txt))
    metadatas = [_chunk_obj_to_metadata(c) for c in chunk_objs]
    ids = [c.chunk_id for c in chunk_objs]

    collection.add(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )
    logger.info(
        "✅ Persisted %d chunks to Chroma collection '%s'", len(ids), collection.name
    )


@timer_wrap
def fetch_chunk_objs_from_chroma(
    doc_id: str,
    cfg_path: Optional[os.PathLike] | None = None,
    collection_name: str | None = None,
) -> List["Chunk"]:
    """Re-hydrate Chunk objects from Chroma by `document_id`."""
    # from src.models import ChunkMetadata, CitationEntity  # Updated import

    cfg = _load_cfg(cfg_path)
    _, collection = _get_chroma_collection(cfg, collection_name or doc_id)

    logger.info("▸ Querying Chroma for document_id=%s", doc_id)
    res = collection.get(
        where={"document_id": doc_id}, include=["documents", "metadatas"]
    )

    chunks: List[Chunk] = []
    for doc, meta in zip(res["documents"], res["metadatas"]):
        # Handle citation_entities
        citation_entities = []
        if meta.get("citation_entities"):
            for ce_str in meta["citation_entities"].split("/"):
                if ce_str:
                    citation_entities.append(CitationEntity.from_string(ce_str))

        chunk_meta = ChunkMetadata(
            chunk_id=meta["chunk_id"],
            token_count=meta.get("token_count", 0),
            previous_chunk_id=meta.get("previous_chunk_id"),
            next_chunk_id=meta.get("next_chunk_id"),
            citation_entities=citation_entities,
        )
        chunks.append(
            Chunk(
                chunk_id=meta["chunk_id"], 
                document_id=meta["document_id"],
                text=doc, 
                score=None, 
                chunk_metadata=chunk_meta
            )
        )

    logger.info("▸ Fetched %d chunks from Chroma", len(chunks))
    return chunks

@timer_wrap
def save_chunk_to_chroma(
    chunk: str,
    cfg_path: Optional[os.PathLike] | None = None,
    collection_name: str = "chunks",
    metadata: Optional[List[dict]] = None,
    model_name: str = "text-embedding-3-small"
):
    """Embed raw *chunks* and upsert them into a ChromaDB collection."""
    if not chunk:
        logger.warning("save_chunks_to_chroma() called with an empty chunk list")
        return

    cfg = _load_cfg(cfg_path)
    embedder = _build_embedder(cfg.get("embed_model", model_name))

    logger.info("▸ Embedding %d chunks for Chroma upsert")
    embeddings = embedder.get_text_embedding(chunk)
    if metadata is None:
        metadata = {"chunk_index": 0}

    client, collection = _get_chroma_collection(cfg, collection_name)
    id = str(uuid.uuid4())

    logger.info(
        "▸ Upserting 1 item into collection '%s' (path=%s)",
        collection_name,
        cfg["vector_store"]["path"],
    )
    collection.add(ids=id, documents=chunk, metadatas=metadata, embeddings=embeddings)
    logger.info("✅ Persisted collection to disk (%s)", cfg["vector_store"]["path"])
    return {
        "id": id,
        "metadata": metadata,
        "embeddings": embeddings
    }

@timer_wrap
def save_chunks_to_chroma(
    chunks: List[str],
    cfg_path: Optional[os.PathLike] | None = None,
    collection_name: str = "chunks",
    metadata: Optional[List[dict]] = None,
    model_name: str = "text-embedding-3-small"
):
    """Embed raw *chunks* and upsert them into a ChromaDB collection."""
    if not chunks:
        logger.warning("save_chunks_to_chroma() called with an empty chunk list")
        return

    cfg = _load_cfg(cfg_path)
    embedder = _build_embedder(cfg.get("embed_model", model_name))

    logger.info("▸ Embedding %d chunks for Chroma upsert", len(chunks))
    embeddings = [embedder.get_text_embedding(ch) for ch in chunks]

    if metadata is None:
        metadata = [{"chunk_index": i} for i in range(len(chunks))]
    elif len(metadata) != len(chunks):
        raise ValueError("length of metadata list must match chunks list")

    client, collection = _get_chroma_collection(cfg, collection_name)
    ids = [str(uuid.uuid4()) for _ in chunks]

    logger.info(
        "▸ Upserting %d items into collection '%s' (path=%s)",
        len(chunks),
        collection_name,
        cfg["vector_store"]["path"],
    )
    collection.add(ids=ids, documents=chunks, metadatas=metadata, embeddings=embeddings)
    logger.info("✅ Persisted collection to disk (%s)", cfg["vector_store"]["path"])
    return [
        {
            "id": id,
            "metadata": metadata,
            "embeddings": embeddings
        } for id, metadata, embeddings in zip(ids, metadata, embeddings)
    ]


# ---------------------------------------------------------------------------
# CLI helper (quick test)
# ---------------------------------------------------------------------------

def download_offline_model(model_name: str = "bge-small-en-v1.5"):
    """Download and cache an offline model for use."""
    success = OfflineEmbedder.download_model(model_name)
    if success:
        logger.info("✅ Offline model %s ready for use", model_name)
    else:
        logger.error("❌ Failed to download offline model %s", model_name)
    return success


if __name__ == "__main__":  # pragma: no cover
    import sys
    import textwrap

    # Check if we're being asked to download a model
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        model_name = sys.argv[2] if len(sys.argv) > 2 else "bge-small-en-v1.5"
        download_offline_model(model_name)
    else:
        # Default demo
        sample_text = textwrap.dedent(
            """
            Semantic chunking splits text along **topic changes** instead of raw
            token counts.  This standalone demo embeds the chunks via OpenAI and
            stores them in a local ChromaDB collection named 'test'.
            """
        ).strip()

        cks = semantic_chunk_text(sample_text)
        save_chunks_to_chroma(cks, collection_name="test")
