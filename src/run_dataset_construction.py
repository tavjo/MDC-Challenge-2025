#!/usr/bin/env python3
"""
Dataset Construction Orchestration Script

This script orchestrates the complete dataset construction process:
1. Load retrieval results JSON
2. Call dataset construction service
3. Bulk upload datasets to DuckDB using DBHelper
4. Mask Citations
5. Embeddings & storage to ChromaDB
6. Export dataset_embedding_stats.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.helpers import initialize_logging, timer_wrap
from api.services.dataset_construction_service import (
    construct_datasets_from_retrieval_results,
    compute_dataset_embeddings,
    get_neighborhood_statistics,
    export_dataset_embedding_stats
)
from api.utils.duckdb_utils import DuckDBHelper

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)


@timer_wrap
def main(
    retrieval_results_path: str = "reports/retrieval/retrieval_results.json",
    db_path: str = "artifacts/mdc_challenge.db",
    collection_name: str = "dataset-aggregates-train",
    output_dir: str = "reports/dataset_construction",
    k_neighbors: int = 5,
    mask_token: str = "<DATASET_ID>",
    model_name: str = "all-MiniLM-L6-v2"
):
    """
    Orchestrate the complete Phase 6 dataset construction process.
    
    Args:
        retrieval_results_path: Path to retrieval results JSON
        db_path: Path to DuckDB database
        collection_name: ChromaDB collection name for dataset embeddings
        output_dir: Directory for output files
        k_neighbors: Number of neighbors for statistics
        mask_token: Token for masking dataset IDs
        model_name: Embedding model name
    """
    logger.info("=== Starting Dataset Object Construction & Aggregation ===")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Construct Dataset objects from retrieval results
        logger.info("Step 1: Constructing Dataset objects from retrieval results")
        datasets = construct_datasets_from_retrieval_results(
            retrieval_results_path=retrieval_results_path,
            db_path=db_path,
            collection_name="mdc_training_data",  # Source collection for chunk retrieval
            k_neighbors=k_neighbors,
            mask_token=mask_token
        )
        
        if not datasets:
            logger.error("No datasets were constructed. Exiting.")
            return
        
        logger.info(f"Successfully constructed {len(datasets)} Dataset objects")
        
        # Step 2: Store Dataset objects in DuckDB
        logger.info("Step 2: Storing Dataset objects in DuckDB")
        db_helper = DuckDBHelper(db_path)
        
        storage_result = db_helper.bulk_upsert_datasets(datasets)
        logger.info(f"Storage result: {storage_result}")
        
        # Step 3: Compute and store dataset embeddings in ChromaDB
        logger.info("Step 3: Computing and storing dataset embeddings")
        dataset_embeddings = compute_dataset_embeddings(
            datasets=datasets,
            collection_name=collection_name,
            model_name=model_name
        )
        
        logger.info(f"Computed embeddings for {len(dataset_embeddings)} datasets")
        
        # Step 4: Compute neighborhood statistics
        logger.info("Step 4: Computing neighborhood statistics")
        neighborhood_stats = get_neighborhood_statistics(
            dataset_embeddings=dataset_embeddings,
            collection_name=collection_name,
            k_neighbors=k_neighbors
        )
        
        logger.info(f"Computed neighborhood stats for {len(neighborhood_stats)} datasets")
        
        # Step 5: Export dataset embedding statistics CSV
        logger.info("Step 5: Exporting dataset embedding statistics")
        stats_csv_path = output_path / "dataset_embedding_stats.csv"
        
        exported_path = export_dataset_embedding_stats(
            datasets=datasets,
            neighborhood_stats=neighborhood_stats,
            output_path=str(stats_csv_path)
        )
        
        logger.info(f"Exported statistics to: {exported_path}")
        
        # Step 6: Generate summary report
        logger.info("Step 6: Generating summary report")
        summary_report = generate_summary_report(
            datasets=datasets,
            dataset_embeddings=dataset_embeddings,
            neighborhood_stats=neighborhood_stats,
            storage_result=storage_result
        )
        
        # Save summary report
        summary_path = output_path / "dataset_construction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to: {summary_path}")
        
        logger.info("=== Phase 6 Dataset Construction Completed Successfully ===")
        
        # Print summary
        print("\n" + "="*60)
        print("PHASE 6 DATASET CONSTRUCTION SUMMARY")
        print("="*60)
        print(f"Total datasets processed: {len(datasets)}")
        print(f"Datasets stored in DuckDB: {storage_result.get('total_datasets_upserted', 0)}")
        print(f"Embeddings computed: {len(dataset_embeddings)}")
        print(f"Neighborhood stats computed: {len(neighborhood_stats)}")
        print(f"Statistics exported to: {exported_path}")
        print(f"Summary report: {summary_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Phase 6 dataset construction failed: {str(e)}")
        raise


def generate_summary_report(
    datasets: List,
    dataset_embeddings: Dict[str, Any],
    neighborhood_stats: Dict[str, Dict[str, float]],
    storage_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a summary report of the dataset construction process.
    
    Args:
        datasets: List of Dataset objects
        dataset_embeddings: Dictionary of dataset embeddings
        neighborhood_stats: Neighborhood statistics
        storage_result: Database storage results
        
    Returns:
        Summary report dictionary
    """
    # Compute aggregate statistics
    total_tokens = sum(d.total_tokens for d in datasets)
    avg_tokens_per_dataset = total_tokens / len(datasets) if datasets else 0
    avg_chunks_per_dataset = sum(d.avg_tokens_per_chunk for d in datasets) / len(datasets) if datasets else 0
    
    # Character length statistics
    total_chars = sum(d.total_char_length for d in datasets)
    total_clean_chars = sum(d.clean_text_length for d in datasets)
    masking_ratio = (total_chars - total_clean_chars) / total_chars if total_chars > 0 else 0
    
    # Neighborhood statistics aggregates (if available)
    neighborhood_summary = {}
    if neighborhood_stats:
        all_similarities = []
        all_norms = []
        
        for stats in neighborhood_stats.values():
            if 'neighbor_mean_similarity' in stats:
                all_similarities.append(stats['neighbor_mean_similarity'])
            if 'neighbor_mean_norm' in stats:
                all_norms.append(stats['neighbor_mean_norm'])
        
        if all_similarities:
            neighborhood_summary = {
                'avg_neighborhood_similarity': sum(all_similarities) / len(all_similarities),
                'max_neighborhood_similarity': max(all_similarities),
                'min_neighborhood_similarity': min(all_similarities)
            }
        
        if all_norms:
            neighborhood_summary.update({
                'avg_embedding_norm': sum(all_norms) / len(all_norms),
                'max_embedding_norm': max(all_norms),
                'min_embedding_norm': min(all_norms)
            })
    
    return {
        'phase': 'Phase 6: Dataset Object Construction & Aggregation',
        'timestamp': Path().cwd().name,  # Current timestamp placeholder
        'totals': {
            'datasets_constructed': len(datasets),
            'datasets_stored': storage_result.get('total_datasets_upserted', 0),
            'embeddings_computed': len(dataset_embeddings),
            'neighborhood_stats_computed': len(neighborhood_stats)
        },
        'aggregates': {
            'total_tokens': total_tokens,
            'avg_tokens_per_dataset': avg_tokens_per_dataset,
            'avg_chunks_per_dataset': avg_chunks_per_dataset,
            'total_characters': total_chars,
            'total_clean_characters': total_clean_chars,
            'masking_ratio': masking_ratio
        },
        'neighborhood_summary': neighborhood_summary,
        'storage_result': storage_result
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 6 Dataset Construction pipeline"
    )
    
    parser.add_argument(
        "--retrieval-results",
        default="reports/retrieval/retrieval_results.json",
        help="Path to retrieval results JSON file"
    )
    
    parser.add_argument(
        "--db-path",
        default="artifacts/mdc_challenge.db",
        help="Path to DuckDB database"
    )
    
    parser.add_argument(
        "--collection-name",
        default="dataset-aggregates-train",
        help="ChromaDB collection name for dataset embeddings"
    )
    
    parser.add_argument(
        "--output-dir",
        default="reports/dataset_construction",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for statistics computation"
    )
    
    parser.add_argument(
        "--mask-token",
        default="<DATASET_ID>",
        help="Token for masking dataset IDs"
    )
    
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    main(
        retrieval_results_path=args.retrieval_results,
        db_path=args.db_path,
        collection_name=args.collection_name,
        output_dir=args.output_dir,
        k_neighbors=args.k_neighbors,
        mask_token=args.mask_token,
        model_name=args.model_name
    )