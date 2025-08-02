#!/usr/bin/env python3
"""
Network-Based Clustering for MDC-Challenge-2025
Implements k-NN similarity graph construction and Leiden clustering for dataset embeddings
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import leidenalg
import chromadb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.helpers import initialize_logging, timer_wrap
from src.models import Dataset
from api.utils.duckdb_utils import DuckDBHelper

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
        dataset_embedding: The embedding vector for a dataset
        chroma_client: ChromaDB client instance
        collection_name: Name of the ChromaDB collection
        k: Number of nearest neighbors to find
        
    Returns:
        Dictionary with neighborhood statistics
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        
        # Query for k+1 neighbors (includes self)
        results = collection.query(
            query_embeddings=[dataset_embedding.tolist()],
            n_results=k + 1
        )
        
        if not results['embeddings'] or len(results['embeddings'][0]) < 2:
            logger.warning("Insufficient neighbors found for statistics")
            return {}
        
        # Get neighbor embeddings (exclude self - first result)
        neighbor_embeddings = np.array(results['embeddings'][0][1:k+1])
        
        # Compute similarities with dataset embedding
        similarities = cosine_similarity([dataset_embedding], neighbor_embeddings)[0]
        
        # Compute embedding norms
        neighbor_norms = np.linalg.norm(neighbor_embeddings, axis=1)
        
        stats = {
            'mean_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'var_similarity': float(np.var(similarities)),
            'mean_neighbor_norm': float(np.mean(neighbor_norms)),
            'max_neighbor_norm': float(np.max(neighbor_norms)),
            'var_neighbor_norm': float(np.var(neighbor_norms)),
            'num_neighbors': len(neighbor_embeddings)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error computing neighborhood stats: {str(e)}")
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
        dataset_embeddings: Dictionary mapping dataset_id to embedding arrays
        k_neighbors: Number of neighbors to consider for each node
        similarity_threshold: Minimum similarity for edge creation (optional)
        threshold_method: Method for determining threshold if not provided
        
    Returns:
        igraph.Graph object with similarity edges
    """
    logger.info(f"Building k-NN similarity graph for {len(dataset_embeddings)} datasets")
    
    if len(dataset_embeddings) < 2:
        logger.warning("Need at least 2 datasets to build similarity graph")
        return ig.Graph()
    
    # Convert to arrays for sklearn
    dataset_ids = list(dataset_embeddings.keys())
    embeddings_matrix = np.array([dataset_embeddings[dataset_id] for dataset_id in dataset_ids])
    
    logger.info(f"Embeddings matrix shape: {embeddings_matrix.shape}")
    
    # Build k-NN index
    k_effective = min(k_neighbors, len(dataset_ids) - 1)
    nn_model = NearestNeighbors(n_neighbors=k_effective + 1, metric='cosine')
    nn_model.fit(embeddings_matrix)
    
    # Find k-nearest neighbors
    distances, indices = nn_model.kneighbors(embeddings_matrix)
    
    # Convert distances to similarities (cosine distance -> cosine similarity)
    similarities = 1 - distances
    
    # Determine similarity threshold if not provided
    if similarity_threshold is None:
        similarity_threshold = determine_similarity_threshold(
            similarities.flatten(), 
            method=threshold_method,
            target_degree=15
        )
        logger.info(f"Automatically determined similarity threshold: {similarity_threshold:.4f}")
    
    # Build edge list
    edges = []
    edge_weights = []
    
    for i in range(len(dataset_ids)):
        for j_idx in range(1, len(indices[i])):  # Skip self (index 0)
            j = indices[i][j_idx]
            similarity = similarities[i][j_idx]
            
            if similarity >= similarity_threshold:
                # Add undirected edge (avoid duplicates by ensuring i < j)
                if i < j:
                    edges.append((i, j))
                    edge_weights.append(similarity)
    
    logger.info(f"Created {len(edges)} edges above similarity threshold {similarity_threshold:.4f}")
    
    # Create igraph
    graph = ig.Graph()
    graph.add_vertices(len(dataset_ids))
    
    # Add dataset_id as vertex attribute
    graph.vs['dataset_id'] = dataset_ids
    
    if edges:
        graph.add_edges(edges)
        graph.es['weight'] = edge_weights
        graph.es['similarity'] = edge_weights
    
    logger.info(f"Built graph with {graph.vcount()} vertices and {graph.ecount()} edges")
    return graph


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
        distances: Array of similarity/distance values
        method: Thresholding method to use
        target_degree: Target average degree for degree_target method
        
    Returns:
        Computed similarity threshold
    """
    # Remove self-similarities (typically 1.0 or very high values)
    valid_distances = distances[distances < 0.99]
    
    if len(valid_distances) == 0:
        logger.warning("No valid distances found, using default threshold")
        return 0.7
    
    if method == "degree_target":
        # Sort similarities in descending order
        sorted_sims = np.sort(valid_distances)[::-1]
        
        # Estimate number of nodes from distances array
        n_comparisons = len(distances)
        estimated_nodes = int(np.sqrt(n_comparisons * 2))  # Rough estimate
        
        # Calculate target number of edges for desired average degree
        target_edges = (estimated_nodes * target_degree) // 2
        
        if target_edges >= len(sorted_sims):
            threshold = sorted_sims[-1] if len(sorted_sims) > 0 else 0.5
        else:
            threshold = sorted_sims[target_edges - 1]
            
        logger.info(f"Degree target method: threshold={threshold:.4f}, estimated_nodes={estimated_nodes}")
        return threshold
        
    elif method == "percentile_90":
        threshold = np.percentile(valid_distances, 90)
        logger.info(f"Percentile 90 method: threshold={threshold:.4f}")
        return threshold
        
    elif method == "elbow_method":
        sorted_sims = np.sort(valid_distances)[::-1]
        
        # Find largest gap in similarities
        gaps = np.diff(sorted_sims)
        largest_gap_idx = np.argmax(np.abs(gaps))
        threshold = sorted_sims[largest_gap_idx]
        
        logger.info(f"Elbow method: threshold={threshold:.4f}, gap_position={largest_gap_idx}")
        return threshold
        
    else:
        logger.warning(f"Unknown threshold method '{method}', using default")
        return 0.7


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
        graph: igraph.Graph object to cluster
        resolution: Resolution parameter for Leiden algorithm
        min_cluster_size: Minimum size for valid clusters
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping dataset_id to cluster_label
    """
    logger.info(f"Running Leiden clustering on graph with {graph.vcount()} vertices, {graph.ecount()} edges")
    
    if graph.vcount() == 0:
        logger.warning("Empty graph provided for clustering")
        return {}
    
    if graph.ecount() == 0:
        logger.warning("Graph has no edges, assigning each node to its own cluster")
        dataset_ids = graph.vs['dataset_id']
        return {dataset_id: f"cluster_{i}" for i, dataset_id in enumerate(dataset_ids)}
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    try:
        # Run Leiden clustering
        partition = leidenalg.find_partition(
            graph, 
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=random_seed
        )
        
        logger.info(f"Initial clustering found {len(partition)} clusters")
        
        # Get cluster assignments
        cluster_assignments = partition.membership
        dataset_ids = graph.vs['dataset_id']
        
        # Count cluster sizes
        cluster_sizes = {}
        for cluster_id in cluster_assignments:
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        
        # Filter clusters by minimum size and reassign small clusters
        valid_clusters = {cluster_id for cluster_id, size in cluster_sizes.items() if size >= min_cluster_size}
        
        logger.info(f"Clusters meeting minimum size ({min_cluster_size}): {len(valid_clusters)}")
        
        # Create final cluster mapping
        cluster_mapping = {}
        cluster_counter = 0
        
        for i, (dataset_id, cluster_id) in enumerate(zip(dataset_ids, cluster_assignments)):
            if cluster_id in valid_clusters:
                # Use valid cluster
                if cluster_id not in cluster_mapping:
                    cluster_mapping[cluster_id] = f"cluster_{cluster_counter}"
                    cluster_counter += 1
                final_cluster = cluster_mapping[cluster_id]
            else:
                # Assign small clusters to singleton clusters
                final_cluster = f"singleton_{dataset_id}"
            
            cluster_mapping[dataset_id] = final_cluster
        
        # Report final statistics
        final_cluster_counts = {}
        for cluster_label in cluster_mapping.values():
            final_cluster_counts[cluster_label] = final_cluster_counts.get(cluster_label, 0) + 1
        
        logger.info(f"Final clustering: {len(set(cluster_mapping.values()))} clusters")
        logger.info(f"Cluster size distribution: {dict(sorted(final_cluster_counts.items()))}")
        
        return cluster_mapping
        
    except Exception as e:
        logger.error(f"Error in Leiden clustering: {str(e)}")
        # Fallback: assign each node to its own cluster
        dataset_ids = graph.vs['dataset_id']
        return {dataset_id: f"fallback_cluster_{i}" for i, dataset_id in enumerate(dataset_ids)}


@timer_wrap
def update_datasets_with_clusters(
    cluster_mapping: Dict[str, str],
    db_path: str = "artifacts/mdc_challenge.db"
) -> Dict[str, Any]:
    """
    Update Dataset objects in DuckDB by adding cluster membership.
    
    Args:
        cluster_mapping: Dictionary mapping dataset_id to cluster_label
        db_path: Path to DuckDB database
        
    Returns:
        Update result summary
    """
    if not cluster_mapping:
        logger.warning("No cluster mapping provided")
        return {"updated": 0, "errors": 0}
    
    logger.info(f"Updating {len(cluster_mapping)} datasets with cluster assignments")
    
    try:
        db_helper = DuckDBHelper(db_path)
        
        # Prepare update data
        update_data = []
        for dataset_id, cluster_label in cluster_mapping.items():
            update_data.append({
                'dataset_id': dataset_id,
                'cluster': cluster_label
            })
        
        # Convert to DataFrame for bulk update
        df = pd.DataFrame(update_data)
        
        # Execute bulk update using DuckDB SQL
        update_query = """
        UPDATE datasets 
        SET cluster = df.cluster
        FROM df
        WHERE datasets.dataset_id = df.dataset_id
        """
        
        connection = db_helper.engine
        connection.execute("CREATE TEMPORARY TABLE df AS SELECT * FROM df")
        result = connection.execute(update_query)
        
        updated_count = len(cluster_mapping)
        logger.info(f"Successfully updated {updated_count} dataset records with cluster assignments")
        
        return {
            "updated": updated_count,
            "errors": 0,
            "clusters_created": len(set(cluster_mapping.values()))
        }
        
    except Exception as e:
        logger.error(f"Error updating datasets with clusters: {str(e)}")
        return {"updated": 0, "errors": 1, "error_message": str(e)}


@timer_wrap
def export_clustering_report(
    cluster_mapping: Dict[str, str],
    graph_stats: Dict[str, Any],
    neighborhood_stats: Dict[str, Dict[str, float]],
    output_dir: str = "reports/clustering"
) -> str:
    """
    Export comprehensive clustering analysis report to JSON.
    
    Args:
        cluster_mapping: Dictionary mapping dataset_id to cluster_label
        graph_stats: Graph construction statistics
        neighborhood_stats: Neighborhood statistics for each dataset
        output_dir: Output directory for reports
        
    Returns:
        Path to the generated report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / f"clustering_report_{timestamp}.json"
    
    # Compute cluster statistics
    cluster_sizes = {}
    for cluster_label in cluster_mapping.values():
        cluster_sizes[cluster_label] = cluster_sizes.get(cluster_label, 0) + 1
    
    # Aggregate neighborhood statistics
    aggregated_neighborhood_stats = {}
    if neighborhood_stats:
        all_stats = list(neighborhood_stats.values())
        if all_stats:
            stat_keys = all_stats[0].keys()
            for key in stat_keys:
                values = [stats[key] for stats in all_stats if key in stats]
                if values:
                    aggregated_neighborhood_stats[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
    
    report_data = {
        "timestamp": timestamp,
        "clustering_summary": {
            "total_datasets": len(cluster_mapping),
            "total_clusters": len(set(cluster_mapping.values())),
            "cluster_size_distribution": cluster_sizes,
            "largest_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "smallest_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
            "singleton_clusters": sum(1 for size in cluster_sizes.values() if size == 1)
        },
        "graph_construction": graph_stats,
        "neighborhood_statistics": {
            "individual_stats": neighborhood_stats,
            "aggregated_stats": aggregated_neighborhood_stats
        },
        "cluster_assignments": cluster_mapping
    }
    
    # Write report
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Clustering report exported to: {report_file}")
    return str(report_file)


@timer_wrap
def run_full_clustering_pipeline(
    collection_name: str = "dataset-aggregates-train",
    db_path: str = "artifacts/mdc_challenge.db",
    k_neighbors: int = 30,
    similarity_threshold: float = None,
    resolution: float = 1.0,
    min_cluster_size: int = 2,
    output_dir: str = "reports/clustering"
) -> Dict[str, Any]:
    """
    Run the complete clustering pipeline:
    1. Load dataset embeddings from ChromaDB
    2. Compute neighborhood statistics
    3. Build k-NN similarity graph
    4. Run Leiden clustering
    5. Update DuckDB with cluster assignments
    6. Export comprehensive report
    
    Args:
        collection_name: ChromaDB collection name for dataset embeddings
        db_path: Path to DuckDB database
        k_neighbors: Number of neighbors for graph construction
        similarity_threshold: Similarity threshold for edge creation
        resolution: Leiden clustering resolution parameter
        min_cluster_size: Minimum cluster size
        output_dir: Output directory for reports
        
    Returns:
        Pipeline execution summary
    """
    logger.info("Starting full clustering pipeline")
    pipeline_start = datetime.now()
    
    try:
        # Step 1: Load dataset embeddings from ChromaDB
        logger.info("Step 1: Loading dataset embeddings from ChromaDB")
        client = chromadb.PersistentClient(path="local_chroma")
        collection = client.get_collection(collection_name)
        
        # Get all embeddings
        results = collection.get(include=['embeddings', 'metadatas'])
        
        if not results['ids']:
            logger.error("No embeddings found in ChromaDB collection")
            return {"status": "error", "message": "No embeddings found"}
        
        # Build embeddings dictionary
        dataset_embeddings = {}
        for i, dataset_id in enumerate(results['ids']):
            dataset_embeddings[dataset_id] = np.array(results['embeddings'][i])
        
        logger.info(f"Loaded {len(dataset_embeddings)} dataset embeddings")
        
        # Step 2: Compute neighborhood statistics
        logger.info("Step 2: Computing neighborhood statistics")
        neighborhood_stats = {}
        for dataset_id, embedding in dataset_embeddings.items():
            stats = compute_neighborhood_embedding_stats(
                embedding, client, collection_name, k=5
            )
            if stats:
                neighborhood_stats[dataset_id] = stats
        
        logger.info(f"Computed neighborhood stats for {len(neighborhood_stats)} datasets")
        
        # Step 3: Build k-NN similarity graph
        logger.info("Step 3: Building k-NN similarity graph")
        graph = build_knn_similarity_graph(
            dataset_embeddings=dataset_embeddings,
            k_neighbors=k_neighbors,
            similarity_threshold=similarity_threshold
        )
        
        graph_stats = {
            "num_vertices": graph.vcount(),
            "num_edges": graph.ecount(),
            "density": (2 * graph.ecount()) / (graph.vcount() * (graph.vcount() - 1)) if graph.vcount() > 1 else 0,
            "k_neighbors": k_neighbors,
            "similarity_threshold": similarity_threshold
        }
        
        # Step 4: Run Leiden clustering
        logger.info("Step 4: Running Leiden clustering")
        cluster_mapping = run_leiden_clustering(
            graph=graph,
            resolution=resolution,
            min_cluster_size=min_cluster_size
        )
        
        # Step 5: Update DuckDB with cluster assignments
        logger.info("Step 5: Updating DuckDB with cluster assignments")
        update_result = update_datasets_with_clusters(cluster_mapping, db_path)
        
        # Step 6: Export comprehensive report
        logger.info("Step 6: Exporting clustering report")
        report_file = export_clustering_report(
            cluster_mapping=cluster_mapping,
            graph_stats=graph_stats,
            neighborhood_stats=neighborhood_stats,
            output_dir=output_dir
        )
        
        pipeline_end = datetime.now()
        execution_time = (pipeline_end - pipeline_start).total_seconds()
        
        summary = {
            "status": "success",
            "execution_time_seconds": execution_time,
            "datasets_processed": len(dataset_embeddings),
            "clusters_created": len(set(cluster_mapping.values())),
            "graph_stats": graph_stats,
            "update_result": update_result,
            "report_file": report_file,
            "timestamp": pipeline_start.strftime("%Y%m%d_%H%M%S")
        }
        
        logger.info(f"Clustering pipeline completed successfully in {execution_time:.2f} seconds")
        return summary
        
    except Exception as e:
        logger.error(f"Clustering pipeline failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }