#!/usr/bin/env python3
"""
Semantic Chunking Pipeline for MDC-Challenge-2025
Creates chunks from Document objects using semantic chunking + ChromaDB storage
"""

import os
import sys
from typing import List


# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.helpers import initialize_logging
from src.semantic_chunking import save_chunk_to_chroma, save_chunks_to_chroma
from src.models import EmbeddingResult

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_CHROMA_CONFIG = os.path.join(project_root, "configs", "chunking.yaml")

def embed_chunk(chunk: str, collection_name: str = "text_embeddings", cfg_path: str = DEFAULT_CHROMA_CONFIG, local_model: bool = False) -> EmbeddingResult:
    logger.info(f"Embedding chunk")
    try:
        if local_model:
            model_name = "bge-small-en-v1.5"
        else:
            model_name = "text-embedding-3-small"
        response = save_chunk_to_chroma(chunk, cfg_path, collection_name, model_name=model_name)
        if response["id"] is not None:
            logger.info(f"Created and saved chunk {response["id"]} embeddings to ChromaDB collection {collection_name} using model {model_name}")
            return EmbeddingResult(
                success=True,
                error=None,
                embeddings=response.get("embeddings", None),
                model_name=model_name,
                collection_name=collection_name,
                id=response.get("id", "")
            )
        else:
            logger.error(f"Failed to create and save chunk {response["id"]} embeddings to ChromaDB collection {collection_name} using model {model_name}")
            return EmbeddingResult(
                success=False,
                error=f"Failed to create and save chunk {response["id"]} embeddings to ChromaDB collection {collection_name} using model {model_name}",
                embeddings=response.get("embeddings", None),
                model_name=model_name,
                collection_name=collection_name,
                id=response.get("id", "")
            )
    except Exception as e:
        logger.error(f"Error embedding chunk: {str(e)}")
        return EmbeddingResult(
                success=False,
                error=f"Error embedding chunk: {str(e)}",
                embeddings=response.get("embeddings", None),
                model_name=model_name,
                collection_name=collection_name,
                id=response.get("id", "")
            )

def embed_chunks(chunks: List[str], collection_name: str = "text_embeddings", cfg_path: str = DEFAULT_CHROMA_CONFIG, local_model: bool = False) -> List[EmbeddingResult]:
    logger.info(f"Embedding {len(chunks)} chunks")
    try:
        if local_model:
            model_name = "bge-small-en-v1.5"
        else:
            model_name = "text-embedding-3-small"
        response = save_chunks_to_chroma(chunks, cfg_path, collection_name, model_name=model_name)
        if response is not None:
            logger.info(f"Created and saved {len(response)} chunks to ChromaDB collection {collection_name} using model {model_name}")
            return [
                EmbeddingResult(
                    success=True,
                    error=None,
                    embeddings=r.get("embeddings", None),
                    model_name=model_name,
                    collection_name=collection_name,
                    id=r.get("id", "")
                ) for r in response
            ]
        else:
            logger.error(f"Failed to create and save {len(chunks)} chunks to ChromaDB collection {collection_name} using model {model_name}")
            return [
                EmbeddingResult(
                    success=False,
                    error=f"Failed to create and save {len(chunks)} chunks to ChromaDB collection {collection_name} using model {model_name}",
                    embeddings=None,
                    model_name=model_name,
                    collection_name=collection_name,
                    id=r.get("id", "")
                ) for r in response
            ]
    except Exception as e:
        logger.error(f"Error embedding chunks: {str(e)}")
        return [
            EmbeddingResult(
                success=False,
                error=f"Error embedding chunks: {str(e)}",
                embeddings=None,
                model_name=model_name,
                collection_name=collection_name,
                id=r.get("id", "")
            ) for r in response
        ]