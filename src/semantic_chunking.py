"""
src/tools/semantic_chunking.py
------------------------------
Semantic chunking + OpenAI embeddings + ChromaDB persistence.

This replaces the previous MiniLM/HuggingFace encoder with OpenAI’s
`text-embedding-3-small` (or any other model you set in *conf/chunking.yaml*).

Prerequisites
~~~~~~~~~~~~~
1. `pip install llama-index-embeddings-openai chromadb`   # plus your other deps
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
from pathlib import Path
from typing import List, Optional

import yaml
import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from dotenv import load_dotenv

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.helpers import initialize_logging  # noqa: E402
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

logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

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
# Builders
# ---------------------------------------------------------------------------


def _build_embedder(model_name: str):
    """Return an OpenAIEmbedding instance.

    The OPENAI_API_KEY environment variable must be set, or you can create
    the embedder with an explicit `api_key` argument.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found – please set it before calling this helper."
        )

    logger.info("▸ Loading OpenAI embedding model %s", model_name)
    return OpenAIEmbedding(model=model_name, api_key=api_key)


def _build_splitter(cfg: dict, embedder: OpenAIEmbedding) -> SemanticSplitterNodeParser:
    """Instantiate SemanticSplitterNodeParser with config-driven hyper-params."""
    return SemanticSplitterNodeParser(
        embed_model=embedder,
        similarity_threshold=cfg.get("similarity_threshold", 0.75),
        chunk_overlap=cfg.get("overlap_sentences", 2),
        chunk_size=cfg.get("max_tokens", 400),
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
    text: str, cfg_path: Optional[os.PathLike] | None = None
) -> List[str]:
    """Return a list of semantic chunks for *text* (page or whole document)."""
    if not text:
        return []

    cfg = _load_cfg(cfg_path)
    embedder = _build_embedder(cfg.get("embed_model", "text-embedding-3-small"))
    splitter = _build_splitter(cfg, embedder)

    logger.info(
        "▸ Splitting text with SemanticSplitterNodeParser (chunk_size=%s, τ=%s)",
        cfg.get("max_tokens", 400),
        cfg.get("similarity_threshold", 0.75),
    )

    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    chunks = [node.text for node in nodes]

    logger.info("▸ Created %d semantic chunks", len(chunks))
    return chunks


@staticmethod
def _chunk_obj_to_metadata(chunk_obj: "Chunk") -> dict:
    """Flatten a Chunk instance to a Chroma-friendly metadata dict."""
    cm = chunk_obj.chunk_metadata
    
    # Handle citation_entities (not candidate_entities)
    citation_entities_str = (
        "/".join([f"{ce.data_citation}|{ce.doc_id}" for ce in cm.citation_entities])
        if cm.citation_entities
        else ""
    )

    return {
        "chunk_id": cm.chunk_id,
        "document_id": cm.document_id,
        "token_count": cm.token_count,
        "previous_chunk_id": cm.previous_chunk_id or "",
        "next_chunk_id": cm.next_chunk_id or "",
        "citation_entities": citation_entities_str,
    }


@timer_wrap
def save_chunk_objs_to_chroma(
    chunk_objs: List["Chunk"],
    cfg_path: Optional[os.PathLike] | None = None,
    collection_name: str | None = None,
):
    """Persist a list of Chunk objects (text + metadata) to ChromaDB."""
    if not chunk_objs:
        logger.warning("No chunk objects supplied – nothing to write to Chroma")
        return

    cfg = _load_cfg(cfg_path)
    embedder = _build_embedder(cfg.get("embed_model", "text-embedding-3-small"))
    _, collection = _get_chroma_collection(
        cfg, collection_name or chunk_objs[0].chunk_metadata.document_id
    )

    logger.info("▸ Embedding %d chunks for Chroma upsert", len(chunk_objs))
    documents = [c.text for c in chunk_objs]
    embeddings = [embedder.get_text_embedding(txt) for txt in documents]
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
    from src.models import ChunkMetadata, CitationEntity  # Updated import

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
                if "|" in ce_str:
                    data_citation, doc_id = ce_str.split("|", 1)
                    citation_entities.append(
                        CitationEntity(
                            data_citation=data_citation,
                            doc_id=doc_id,
                            pages=None,  # Not stored in metadata
                            evidence=None  # Not stored in metadata
                        )
                    )

        chunk_meta = ChunkMetadata(
            chunk_id=meta["chunk_id"],
            document_id=meta["document_id"],
            token_count=meta.get("token_count", 0),
            previous_chunk_id=meta.get("previous_chunk_id"),
            next_chunk_id=meta.get("next_chunk_id"),
            citation_entities=citation_entities,
        )
        chunks.append(
            Chunk(
                chunk_id=meta["chunk_id"], 
                text=doc, 
                score=None, 
                chunk_metadata=chunk_meta
            )
        )

    logger.info("▸ Fetched %d chunks from Chroma", len(chunks))
    return chunks


@timer_wrap
def save_chunks_to_chroma(
    chunks: List[str],
    cfg_path: Optional[os.PathLike] | None = None,
    collection_name: str = "chunks",
    metadata: Optional[List[dict]] = None,
):
    """Embed raw *chunks* and upsert them into a ChromaDB collection."""
    if not chunks:
        logger.warning("save_chunks_to_chroma() called with an empty chunk list")
        return

    cfg = _load_cfg(cfg_path)
    embedder = _build_embedder(cfg.get("embed_model", "text-embedding-3-small"))

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


# ---------------------------------------------------------------------------
# CLI helper (quick test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import textwrap

    sample_text = textwrap.dedent(
        """
        Semantic chunking splits text along **topic changes** instead of raw
        token counts.  This standalone demo embeds the chunks via OpenAI and
        stores them in a local ChromaDB collection named 'test'.
        """
    ).strip()

    cks = semantic_chunk_text(sample_text)
    save_chunks_to_chroma(cks, collection_name="test")
