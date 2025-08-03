"""
src/clustering.py
-----------------
Network-based clustering using k-NN similarity graphs and Leiden algorithm.

This module provides functions for:
1. Computing neighborhood embedding statistics from ChromaDB
2. Building k-NN similarity graphs with memory efficiency
3. Applying Leiden clustering algorithm
4. Updating dataset records and engineered features in DuckDB
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import igraph as ig
from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics.pairwise import cosine_similarity
import leidenalg
import chromadb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.helpers import initialize_logging, timer_wrap
from src.models import EngineeredFeatures, Dataset
from api.utils.duckdb_utils import DuckDBHelper

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)


@timer_wrap
def compute_neighborhood_embedding_stats(
    dataset_embedding: np.ndarray,
    chroma_client,
    collection_name: str,
    k: int = 5
) -> Dict[str, float]:
    """
    Find k-nearest neighbors to dataset embedding and compute:
    - mean, max, variance of neighbor similarities
    - mean, max, variance of neighbor embedding norms
    
    Args:
        dataset_embedding: The embedding vector for the dataset
        chroma_client: ChromaDB client instance
        collection_name: Name of the ChromaDB collection
        k: Number of nearest neighbors to find
        
    Returns:
        Dictionary with neighborhood statistics
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        
        # Query for k nearest neighbors
        results = collection.query(
            query_embeddings=[dataset_embedding.tolist()],
            n_results=k,
            include=["embeddings", "distances"]
        )
        
        if not results["embeddings"] or not results["embeddings"][0]:
            logger.warning("No neighbors found for dataset embedding")
            return {}
        
        neighbor_embeddings = np.array(results["embeddings"][0])
        distances = np.array(results["distances"][0])
        
        # Convert distances to similarities (assuming cosine distance)
        similarities = 1 - distances
        
        # Compute neighbor embedding norms
        neighbor_norms = np.linalg.norm(neighbor_embeddings, axis=1)
        
        stats = {
            "neighbor_similarity_mean": float(np.mean(similarities)),
            "neighbor_similarity_max": float(np.max(similarities)),
            "neighbor_similarity_var": float(np.var(similarities)),
            "neighbor_norm_mean": float(np.mean(neighbor_norms)),
            "neighbor_norm_max": float(np.max(neighbor_norms)),
            "neighbor_norm_var": float(np.var(neighbor_norms))
        }
        
        logger.info(f"Computed neighborhood stats for {k} neighbors")
        return stats
        
    except Exception as e:
        logger.error(f"Error computing neighborhood embedding stats: {e}")
        return {}


@timer_wrap
def build_knn_similarity_graph(
    dataset_embeddings: Dict[str, np.ndarray], 
    k_neighbors: int = 30,
    similarity_threshold: float = None,
    threshold_method: str = "degree_target"
) -> ig.Graph:
    """
    Memory-efficient k-NN similarity graph construction:
    1. Use sklearn.neighbors.NearestNeighbors for O(N*k) memory complexity
    2. Apply similarity threshold to filter edges
    3. Build igraph directly (no NetworkX conversion)
    4. Default to degree-target heuristic for deterministic thresholding
    
    Args:
        dataset_embeddings: Dictionary mapping dataset_id to embedding vector
        k_neighbors: Number of nearest neighbors to consider
        similarity_threshold: Minimum similarity for edge creation
        threshold_method: Method for determining similarity threshold
        
    Returns:
        igraph.Graph object with similarity edges
    """
    logger.info(f"Building k-NN graph for {len(dataset_embeddings)} datasets")
    
    # Convert to arrays for sklearn
    dataset_ids = list(dataset_embeddings.keys())
    embeddings_matrix = np.array([dataset_embeddings[id] for id in dataset_ids])
    
    # Build k-NN index
    logger.info(f"Computing {k_neighbors}-NN with sklearn")
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(dataset_ids)), 
                           metric='cosine', algorithm='brute')
    nbrs.fit(embeddings_matrix)
    
    # Get distances and indices
    distances, indices = nbrs.kneighbors(embeddings_matrix)
    
    # Determine similarity threshold if not provided
    if similarity_threshold is None:
        similarity_threshold = determine_similarity_threshold(
            distances, method=threshold_method, target_degree=15
        )
        logger.info(f"Auto-determined similarity threshold: {similarity_threshold}")
    
    # Build edge list with similarities
    edges = []
    edge_weights = []
    
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for j, (dist, neighbor_idx) in enumerate(zip(dist_row, idx_row)):
            if i != neighbor_idx:  # Skip self-loops
                similarity = 1 - dist  # Convert cosine distance to similarity
                if similarity >= similarity_threshold:
                    edges.append((i, neighbor_idx))
                    edge_weights.append(similarity)
    
    logger.info(f"Created {len(edges)} edges above threshold {similarity_threshold}")
    
    # Create igraph
    g = ig.Graph(n=len(dataset_ids), edges=edges, directed=False)
    g.es['weight'] = edge_weights
    g.vs['dataset_id'] = dataset_ids
    
    # Add basic graph stats
    logger.info(f"Graph stats: {g.vcount()} vertices, {g.ecount()} edges, "
                f"avg degree: {2*g.ecount()/g.vcount():.2f}")
    
    return g


@timer_wrap
def determine_similarity_threshold(
    distances: np.ndarray, 
    method: str = "degree_target",
    target_degree: int = 15
) -> float:
    """
    Dynamically determine similarity threshold:
    - degree_target: Target average node degree (default, deterministic)
    - percentile_90: Use 90th percentile of similarities
    - elbow_method: Find largest gap in sorted similarities
    
    Args:
        distances: Distance matrix from k-NN computation
        method: Method for threshold determination
        target_degree: Target average degree for degree_target method
        
    Returns:
        Similarity threshold value
    """
    # Convert distances to similarities (assuming cosine distance)
    similarities = 1 - distances.flatten()
    similarities = similarities[similarities < 1.0]  # Remove self-similarities
    
    if method == "degree_target":
        # Sort similarities in descending order
        sorted_sims = np.sort(similarities)[::-1]
        
        # Estimate threshold that would give target degree
        n_nodes = distances.shape[0]
        target_edges = (target_degree * n_nodes) // 2
        
        if target_edges >= len(sorted_sims):
            threshold = sorted_sims[-1]
        else:
            threshold = sorted_sims[target_edges]
            
        logger.info(f"Degree-target threshold: {threshold:.4f} "
                   f"(targeting {target_degree} avg degree)")
        return float(threshold)
    
    elif method == "percentile_90":
        threshold = np.percentile(similarities, 90)
        logger.info(f"90th percentile threshold: {threshold:.4f}")
        return float(threshold)
    
    elif method == "elbow_method":
        sorted_sims = np.sort(similarities)[::-1]
        
        # Find largest gap in sorted similarities
        gaps = np.diff(sorted_sims)
        max_gap_idx = np.argmax(np.abs(gaps))
        threshold = sorted_sims[max_gap_idx]
        
        logger.info(f"Elbow method threshold: {threshold:.4f}")
        return float(threshold)
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")


@timer_wrap
def run_leiden_clustering(
    graph: ig.Graph, 
    resolution: float = 1.0, 
    min_cluster_size: int = 2,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Apply Leiden clustering with safeguards:
    1. Use igraph directly (no NetworkX conversion)
    2. Set random seed for reproducibility
    3. Filter out clusters smaller than min_cluster_size
    4. Return {dataset_id: cluster_label} mapping
    
    Args:
        graph: igraph.Graph object with similarity edges
        resolution: Resolution parameter for Leiden algorithm
        min_cluster_size: Minimum size for valid clusters
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping dataset_id to cluster_label
    """
    logger.info(f"Running Leiden clustering with resolution={resolution}")
    
    if graph.ecount() == 0:
        logger.warning("Graph has no edges, assigning all nodes to singleton clusters")
        return {graph.vs[i]['dataset_id']: f"cluster_{i}" 
                for i in range(graph.vcount())}
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Run Leiden clustering
    partition = leidenalg.find_partition(
        graph, 
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_seed
    )
    
    logger.info(f"Initial clustering: {len(partition)} clusters, "
                f"modularity: {partition.modularity:.4f}")
    
    # Create cluster assignments
    cluster_assignments = {}
    cluster_sizes = {}
    
    for i, cluster_id in enumerate(partition.membership):
        dataset_id = graph.vs[i]['dataset_id']
        cluster_label = f"cluster_{cluster_id}"
        cluster_assignments[dataset_id] = cluster_label
        cluster_sizes[cluster_label] = cluster_sizes.get(cluster_label, 0) + 1
    
    # Filter small clusters (reassign to largest cluster or mark as noise)
    large_clusters = {k: v for k, v in cluster_sizes.items() if v >= min_cluster_size}
    
    if large_clusters:
        largest_cluster = max(large_clusters.keys(), key=lambda x: large_clusters[x])
        
        # Reassign small clusters to largest cluster
        for dataset_id, cluster_label in cluster_assignments.items():
            if cluster_label not in large_clusters:
                cluster_assignments[dataset_id] = largest_cluster
                logger.debug(f"Reassigned {dataset_id} from {cluster_label} to {largest_cluster}")
    
    # Final cluster stats
    final_cluster_sizes = {}
    for cluster_label in cluster_assignments.values():
        final_cluster_sizes[cluster_label] = final_cluster_sizes.get(cluster_label, 0) + 1
    
    logger.info(f"Final clustering: {len(final_cluster_sizes)} clusters")
    for cluster_label, size in sorted(final_cluster_sizes.items()):
        logger.info(f"  {cluster_label}: {size} datasets")
    
    return cluster_assignments


@timer_wrap
def update_dataset_clusters_in_duckdb(
    cluster_assignments: Dict[str, str],
    db_helper: DuckDBHelper
) -> bool:
    """
    Update Dataset objects in DuckDB by adding cluster values.
    
    Args:
        cluster_assignments: Dictionary mapping dataset_id to cluster_label
        db_helper: DuckDBHelper instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Updating {len(cluster_assignments)} dataset cluster assignments")
        
        # Get all datasets
        datasets = db_helper.get_all_datasets()
        
        # Update cluster assignments
        updated_datasets = []
        updated_count = 0
        
        for dataset in datasets:
            if dataset.dataset_id in cluster_assignments:
                dataset.cluster = cluster_assignments[dataset.dataset_id]
                dataset = Dataset.model_validate(dataset.model_dump())
                updated_datasets.append(dataset)
                updated_count += 1
        
        if updated_datasets:
            success = db_helper.bulk_upsert_datasets(updated_datasets)
            if success:
                logger.info(f"Successfully updated {updated_count} dataset cluster assignments")
                return True
            else:
                logger.error("Failed to update dataset cluster assignments")
                return False
        else:
            logger.warning("No datasets found to update with cluster assignments")
            return True
            
    except Exception as e:
        logger.error(f"Error updating dataset clusters: {e}")
        return False


@timer_wrap
def add_neighborhood_features_to_duckdb(
    dataset_neighborhood_stats: Dict[str, Dict[str, float]],
    db_helper: DuckDBHelper
) -> bool:
    """
    Add new feature values from neighborhood embeddings stats to 
    engineered_feature_values table in DuckDB.
    
    Args:
        dataset_neighborhood_stats: Dict mapping dataset_id to neighborhood stats
        db_helper: DuckDBHelper instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Adding neighborhood features for {len(dataset_neighborhood_stats)} datasets")
        
        # Get all datasets to map dataset_id to document_id
        datasets = db_helper.get_all_datasets()
        dataset_to_doc = {ds.dataset_id: ds.document_id for ds in datasets}
        
        # Create EngineeredFeatures objects
        features_list = []
        
        for dataset_id, stats in dataset_neighborhood_stats.items():
            if dataset_id in dataset_to_doc:
                # Create EngineeredFeatures object with neighborhood stats
                features = EngineeredFeatures(
                    dataset_id=dataset_id,
                    document_id=dataset_to_doc[dataset_id],
                    **stats  # Unpack all the neighborhood statistics
                )
                features_list.append(features)
        
        if features_list:
            success = db_helper.upsert_engineered_features_batch(features_list)
            if success:
                logger.info(f"Successfully added neighborhood features for {len(features_list)} datasets")
                return True
            else:
                logger.error("Failed to add neighborhood features")
                return False
        else:
            logger.warning("No datasets found to add neighborhood features")
            return True
            
    except Exception as e:
        logger.error(f"Error adding neighborhood features: {e}")
        return False


@timer_wrap
def export_clustering_report(
    cluster_assignments: Dict[str, str],
    dataset_neighborhood_stats: Dict[str, Dict[str, float]],
    graph_stats: Dict[str, Any],
    output_dir: str = "reports/clustering"
) -> bool:
    """
    Export summary report in JSON to reports/clustering directory.
    
    Args:
        cluster_assignments: Dictionary mapping dataset_id to cluster_label
        dataset_neighborhood_stats: Neighborhood statistics for each dataset
        graph_stats: Graph construction statistics
        output_dir: Output directory for the report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"clustering_report_{timestamp}.json"
        
        # Compute cluster statistics
        cluster_sizes = {}
        for cluster_label in cluster_assignments.values():
            cluster_sizes[cluster_label] = cluster_sizes.get(cluster_label, 0) + 1
        
        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_datasets": len(cluster_assignments),
                "total_clusters": len(cluster_sizes),
                "avg_cluster_size": sum(cluster_sizes.values()) / len(cluster_sizes) if cluster_sizes else 0,
                "largest_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
                "smallest_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0
            },
            "graph_stats": graph_stats,
            "cluster_sizes": cluster_sizes,
            "cluster_assignments": cluster_assignments,
            "neighborhood_stats_summary": {
                "datasets_with_stats": len(dataset_neighborhood_stats),
                "avg_neighbor_similarity_mean": np.mean([
                    stats.get("neighbor_similarity_mean", 0) 
                    for stats in dataset_neighborhood_stats.values()
                ]) if dataset_neighborhood_stats else 0
            }
        }
        
        # Write report to file
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Clustering report exported to {report_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting clustering report: {e}")
        return False


def main():
    """
    Main function to demonstrate the clustering pipeline.
    This is primarily for testing purposes.
    """
    logger.info("Starting clustering pipeline demonstration")
    
    # This would typically be called from another script that loads
    # actual dataset embeddings from ChromaDB
    logger.info("Clustering pipeline functions are ready for use")
    logger.info("Use the individual functions to build your clustering workflow")


if __name__ == "__main__":
    main()