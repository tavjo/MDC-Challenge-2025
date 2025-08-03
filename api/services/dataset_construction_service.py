"""
Dataset Construction Service for MDC Challenge 2025

This module provides functions for constructing Dataset objects from retrieval results,
including masking dataset IDs in text and storing the results in DuckDB.

Functions:
- mask_dataset_ids_in_text: Masks dataset IDs in text with configurable tokens
- construct_datasets_from_retrieval_results: Main pipeline function for dataset construction
"""

import os
import sys
import json
import re
from typing import List

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.models import Dataset, DatasetConstructionResult
from src.helpers import timer_wrap, initialize_logging
from api.utils.duckdb_utils import DuckDBHelper

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)

# Default database path
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"


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
        str: Text with dataset IDs masked
    """
    if not dataset_ids:
        return text
    
    # Escape special regex characters and create pattern with word boundaries
    escaped_ids = [re.escape(dataset_id) for dataset_id in dataset_ids]
    pattern = r'\b(' + r'|'.join(escaped_ids) + r')\b'
    
    # Perform case-insensitive replacement
    masked_text = re.sub(pattern, mask_token, text, flags=re.IGNORECASE)
    
    logger.debug(f"Masked {len(dataset_ids)} dataset IDs in text of length {len(text)}")
    return masked_text


# Note: Using existing bulk_upsert_datasets method from DuckDBHelper


@timer_wrap
def construct_datasets_from_retrieval_results(
    retrieval_results_path: str = "reports/retrieval/retrieval_results.json",
    db_path: str = DEFAULT_DUCKDB_PATH,
    mask_token: str = "<DATASET_ID>"
) -> DatasetConstructionResult:
    """
    For each dataset ID in retrieval results:
    1. Fetch all associated chunk IDs and retrieve full Chunk objects from DuckDB
    2. Concatenate chunk texts, mask dataset ID tokens
    3. Compute aggregation statistics (total_tokens, avg_tokens_per_chunk, etc.)
    4. Construct Dataset Pydantic objects
    5. Save Dataset Objects to DuckDB
    
    Args:
        retrieval_results_path: Path to retrieval results JSON file
        db_path: Path to DuckDB database
        mask_token: Token to replace dataset IDs with
        
    Returns:
        List[Dataset]: List of constructed Dataset objects
    """
    logger.info(f"Starting dataset construction from {retrieval_results_path}")
    
    # 1. Load retrieval results
    try:
        with open(retrieval_results_path, 'r') as f:
            retrieval_data = json.load(f)
        
        chunk_ids_by_dataset = retrieval_data.get("chunk_ids", {})
        logger.info(f"Loaded retrieval results with {len(chunk_ids_by_dataset)} dataset IDs")
        
    except Exception as e:
        logger.error(f"Failed to load retrieval results: {str(e)}")
        raise ValueError(f"Could not load retrieval results from {retrieval_results_path}: {str(e)}")
    
    # 2. Initialize database helper
    db_helper = DuckDBHelper(db_path)
    
    datasets = []
    
    try:
        # 3. Process each dataset ID
        for dataset_id, chunk_ids in chunk_ids_by_dataset.items():
            logger.info(f"Processing dataset ID: {dataset_id} with {len(chunk_ids)} chunks")
            
            # 3.1. Retrieve chunks from DuckDB
            try:
                chunks = db_helper.get_chunks_by_chunk_ids(chunk_ids)
                logger.debug(f"Retrieved {len(chunks)} chunks for dataset {dataset_id}")
                
                if not chunks:
                    logger.warning(f"No chunks found for dataset {dataset_id}, skipping")
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to retrieve chunks for dataset {dataset_id}: {str(e)}")
                continue
            
            # 3.2. Concatenate chunk texts
            combined_text = " ".join([chunk.text for chunk in chunks])
            
            # 3.3. Mask dataset ID in the combined text
            masked_text = mask_dataset_ids_in_text(combined_text, [dataset_id], mask_token)
            
            # 3.4. Compute aggregation statistics
            total_tokens = sum(chunk.chunk_metadata.token_count for chunk in chunks)
            avg_tokens_per_chunk = total_tokens / len(chunks) if chunks else 0.0
            total_char_length = len(combined_text)
            clean_text_length = len(masked_text)
            
            # Get document_id from the first chunk (assuming all chunks from same dataset belong to same doc)
            document_id = chunks[0].document_id if chunks else ""
            
            # 3.5. Create Dataset object
            dataset = Dataset(
                dataset_id=dataset_id,
                document_id=document_id,
                total_tokens=total_tokens,
                avg_tokens_per_chunk=avg_tokens_per_chunk,
                total_char_length=total_char_length,
                clean_text_length=clean_text_length,
                cluster=None,  # Will be populated later in clustering phase
                dataset_type=None,  # Will be populated later
                text=masked_text
            )
            
            datasets.append(dataset)
            logger.debug(f"Created dataset object for {dataset_id}")
        
        # 4. Store datasets in DuckDB using existing bulk_upsert_datasets method
        if datasets:
            logger.info(f"Storing {len(datasets)} datasets in DuckDB")
            result = db_helper.bulk_upsert_datasets(datasets)
            if not result.get("success", False):
                logger.error("Failed to store datasets in DuckDB")
                raise RuntimeError("Dataset storage failed")
            logger.info(f"Successfully stored {result.get('total_datasets_upserted', len(datasets))} datasets in DuckDB")
        else:
            logger.warning("No datasets were constructed")
        
    finally:
        # 5. Close database connection
        db_helper.close()
    
    logger.info(f"Dataset construction completed. Created {len(datasets)} datasets")
    return DatasetConstructionResult(
        success=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Test masking function
    test_text = "This dataset https://doi.org/10.5281/zenodo.8014150 contains important data."
    test_ids = ["https://doi.org/10.5281/zenodo.8014150"]
    masked = mask_dataset_ids_in_text(test_text, test_ids)
    print(f"Original: {test_text}")
    print(f"Masked: {masked}")
    
    # Test full pipeline (uncomment to run)
    # datasets = construct_datasets_from_retrieval_results()
    # print(f"Constructed {len(datasets)} datasets")