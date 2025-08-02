#!/usr/bin/env python3
"""
Dataset Construction Service for MDC-Challenge-2025

This service handles the construction of Dataset objects from retrieval results,
including text concatenation, masking, embedding computation, and ChromaDB storage.
"""

import os
import sys
import json
import re
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Local imports
from src.models import Dataset, Chunk
from src.helpers import initialize_logging, timer_wrap
from api.utils.duckdb_utils import DuckDBHelper
from src.semantic_chunking import save_chunks_to_chroma
import chromadb

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)

# API endpoints for embeddings service
API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "create_chunks": "/create_chunks",
    "batch_create_chunks": "/batch_create_chunks",
    "embed_chunks": "/embed_chunks",
    "run_semantic_chunking": "/run_semantic_chunking",
    "chunk_specific_documents": "/chunk/documents"
}


@timer_wrap
def construct_datasets_from_retrieval_results(
    retrieval_results_path: str = "reports/retrieval/retrieval_results.json",
    db_path: str = "artifacts/mdc_challenge.db",
    collection_name: str = "mdc_training_data",
    k_neighbors: int = 5,
    mask_token: str = "<DATASET_ID>"
) -> List[Dataset]:
    """
    For each dataset ID in retrieval results:
    1. Fetch all associated chunk IDs and retrieve full Chunk objects from DuckDB
    2. Concatenate chunk texts, mask dataset ID tokens
    3. Compute aggregation statistics (total_tokens, avg_tokens_per_chunk, etc.)
    4. Re-embed masked concatenated text
    5. Collect k-nearest neighbor embeddings and compute neighborhood stats
    6. Construct Dataset Pydantic objects
    
    Args:
        retrieval_results_path: Path to retrieval results JSON file
        db_path: Path to DuckDB database
        collection_name: ChromaDB collection name
        k_neighbors: Number of neighbors for neighborhood statistics
        mask_token: Token to use for masking dataset IDs
        
    Returns:
        List of constructed Dataset objects
    """
    logger.info(f"Loading retrieval results from {retrieval_results_path}")
    
    # Load retrieval results
    with open(retrieval_results_path, 'r') as f:
        retrieval_data = json.load(f)
    
    chunk_ids_by_dataset = retrieval_data.get("chunk_ids", {})
    logger.info(f"Found {len(chunk_ids_by_dataset)} datasets in retrieval results")
    
    # Initialize DuckDB helper
    db_helper = DuckDBHelper(db_path)
    
    datasets = []
    
    for dataset_id, chunk_ids in chunk_ids_by_dataset.items():
        logger.info(f"Processing dataset {dataset_id} with {len(chunk_ids)} chunks")
        
        try:
            # 1. Fetch chunk objects from DuckDB
            chunks = db_helper.get_chunks_by_chunk_ids(chunk_ids)
            
            if not chunks:
                logger.warning(f"No chunks found for dataset {dataset_id}")
                continue
            
            # 2. Concatenate chunk texts
            concatenated_text = " ".join([chunk.chunk_text for chunk in chunks])
            
            # Extract document_id from the first chunk (assuming all chunks from same document)
            document_id = chunks[0].document_id if chunks else ""
            
            # 3. Mask dataset IDs in text
            masked_text = mask_dataset_ids_in_text(concatenated_text, [dataset_id], mask_token)
            
            # 4. Compute aggregation statistics
            total_tokens = sum(len(chunk.chunk_text.split()) for chunk in chunks)
            avg_tokens_per_chunk = total_tokens / len(chunks) if chunks else 0
            total_char_length = len(concatenated_text)
            clean_text_length = len(masked_text)
            
            # 5. Create Dataset object
            dataset = Dataset(
                dataset_id=dataset_id,
                document_id=document_id,
                total_tokens=total_tokens,
                avg_tokens_per_chunk=avg_tokens_per_chunk,
                total_char_length=total_char_length,
                clean_text_length=clean_text_length,
                text=masked_text
            )
            
            datasets.append(dataset)
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
            continue
    
    logger.info(f"Successfully constructed {len(datasets)} Dataset objects")
    return datasets


@timer_wrap 
def mask_dataset_ids_in_text(text: str, dataset_ids: List[str], mask_token: str = "<DATASET_ID>") -> str:
    """
    Robust multi-ID masking with regex escaping:
    - Handles multiple dataset IDs per text
    - Case-insensitive matching with word boundaries  
    - Prevents leakage from compound citations
    
    Args:
        text: Input text to mask
        dataset_ids: List of dataset IDs to mask
        mask_token: Token to replace dataset IDs with
        
    Returns:
        Text with dataset IDs masked
    """
    if not dataset_ids:
        return text
    
    # Create pattern with word boundaries and case-insensitive matching
    escaped_ids = [re.escape(str(id_)) for id_ in dataset_ids]
    pattern = r'\b(' + r'|'.join(escaped_ids) + r')\b'
    
    masked_text = re.sub(pattern, mask_token, text, flags=re.IGNORECASE)
    
    logger.debug(f"Masked {len(dataset_ids)} dataset IDs in text")
    return masked_text


@timer_wrap
def compute_dataset_embeddings(
    datasets: List[Dataset],
    collection_name: str = "dataset-aggregates-train",
    cfg_path: str = "configs/chunking.yaml",
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, np.ndarray]:
    """
    Compute dataset-level embeddings and store in ChromaDB:
    - Re-embed concatenated masked text 
    - Store in dedicated ChromaDB collection for dataset-level vectors
    
    Args:
        datasets: List of Dataset objects to embed
        collection_name: ChromaDB collection name for dataset embeddings
        cfg_path: Path to chunking configuration
        model_name: Name of the embedding model to use
        
    Returns:
        Dictionary mapping dataset_id to embedding arrays
    """
    logger.info(f"Computing embeddings for {len(datasets)} datasets")
    
    if not datasets:
        return {}
    
    # Initialize embedding model
    model = SentenceTransformer(model_name)
    
    # Initialize ChromaDB client
    try:
        client = chromadb.PersistentClient(path="local_chroma")
        
        # Get or create collection for dataset embeddings  
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Dataset-level embeddings for aggregated chunks"}
        )
        
        logger.info(f"Connected to ChromaDB collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}")
        raise
    
    embeddings_dict = {}
    
    # Prepare batch data for ChromaDB
    texts_to_embed = []
    dataset_ids = []
    metadata_list = []
    
    for dataset in datasets:
        texts_to_embed.append(dataset.text)
        dataset_ids.append(dataset.dataset_id)
        
        # Prepare metadata according to memory specification
        metadata = {
            "id": dataset.dataset_id,  # dataset citation ID
            "document_id": dataset.document_id,  # document citation location
            "total_tokens": dataset.total_tokens,
            "avg_tokens_per_chunk": dataset.avg_tokens_per_chunk,
            "total_char_length": dataset.total_char_length,
            "clean_text_length": dataset.clean_text_length
        }
        metadata_list.append(metadata)
    
    try:
        # Generate embeddings in batch
        logger.info("Generating embeddings for dataset texts...")
        embeddings = model.encode(texts_to_embed, convert_to_numpy=True)
        
        # Store in ChromaDB
        logger.info(f"Storing {len(embeddings)} embeddings in ChromaDB...")
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts_to_embed,
            metadatas=metadata_list,
            ids=dataset_ids
        )
        
        # Build return dictionary
        for i, dataset_id in enumerate(dataset_ids):
            embeddings_dict[dataset_id] = embeddings[i]
        
        logger.info(f"Successfully stored {len(embeddings)} dataset embeddings in ChromaDB")
        
    except Exception as e:
        logger.error(f"Error computing/storing dataset embeddings: {str(e)}")
        raise
    
    return embeddings_dict


@timer_wrap
def get_neighborhood_statistics(
    dataset_embeddings: Dict[str, np.ndarray],
    collection_name: str = "dataset-aggregates-train",
    k_neighbors: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Compute k-nearest neighbor statistics for each dataset embedding.
    
    Args:
        dataset_embeddings: Dictionary mapping dataset_id to embeddings
        collection_name: ChromaDB collection name
        k_neighbors: Number of neighbors to consider
        
    Returns:
        Dictionary mapping dataset_id to neighborhood statistics
    """
    logger.info(f"Computing neighborhood statistics for {len(dataset_embeddings)} datasets")
    
    try:
        client = chromadb.PersistentClient(path="local_chroma")
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB collection: {str(e)}")
        return {}
    
    neighborhood_stats = {}
    
    for dataset_id, embedding in dataset_embeddings.items():
        try:
            # Query for k+1 neighbors (including self)
            results = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=k_neighbors + 1,
                include=['distances', 'embeddings']
            )
            
            if results['distances'] and len(results['distances'][0]) > 1:
                # Exclude self (first result) and get neighbor distances
                neighbor_distances = results['distances'][0][1:]  # Skip self
                neighbor_embeddings = results['embeddings'][0][1:]  # Skip self
                
                # Convert distances to similarities (1 - distance for cosine distance)
                similarities = [1 - dist for dist in neighbor_distances]
                
                # Compute embedding norms
                neighbor_norms = [np.linalg.norm(emb) for emb in neighbor_embeddings]
                
                # Compute statistics
                stats = {
                    'neighbor_mean_similarity': float(np.mean(similarities)),
                    'neighbor_max_similarity': float(np.max(similarities)),
                    'neighbor_var_similarity': float(np.var(similarities)),
                    'neighbor_mean_norm': float(np.mean(neighbor_norms)),
                    'neighbor_max_norm': float(np.max(neighbor_norms))
                }
                
                neighborhood_stats[dataset_id] = stats
                
        except Exception as e:
            logger.error(f"Error computing neighborhood stats for {dataset_id}: {str(e)}")
            continue
    
    logger.info(f"Computed neighborhood statistics for {len(neighborhood_stats)} datasets")
    return neighborhood_stats


@timer_wrap
def export_dataset_embedding_stats(
    datasets: List[Dataset],
    neighborhood_stats: Dict[str, Dict[str, float]],
    output_path: str = "dataset_embedding_stats.csv"
) -> str:
    """
    Export dataset statistics and embedding neighborhood data to CSV.
    
    Args:
        datasets: List of Dataset objects
        neighborhood_stats: Neighborhood statistics for each dataset
        output_path: Path for output CSV file
        
    Returns:
        Path to exported CSV file
    """
    logger.info(f"Exporting dataset embedding statistics to {output_path}")
    
    # Prepare data for CSV export
    export_data = []
    
    for dataset in datasets:
        row = {
            'dataset_id': dataset.dataset_id,
            'document_id': dataset.document_id,
            'total_tokens': dataset.total_tokens,
            'avg_tokens_per_chunk': dataset.avg_tokens_per_chunk,
            'total_char_length': dataset.total_char_length,
            'clean_text_length': dataset.clean_text_length,
        }
        
        # Add neighborhood statistics if available
        if dataset.dataset_id in neighborhood_stats:
            row.update(neighborhood_stats[dataset.dataset_id])
        else:
            # Add null values for missing neighborhood stats
            row.update({
                'neighbor_mean_similarity': None,
                'neighbor_max_similarity': None,
                'neighbor_var_similarity': None,
                'neighbor_mean_norm': None,
                'neighbor_max_norm': None
            })
        
        export_data.append(row)
    
    # Create DataFrame and export to CSV
    df = pd.DataFrame(export_data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Successfully exported {len(export_data)} dataset records to {output_path}")
    return output_path