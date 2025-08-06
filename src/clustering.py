"""
src/clustering.py
-----------------
Network-based clustering using k-NN similarity graphs and Leiden algorithm.

This module provides functions for:
1. Building k-NN similarity graphs with memory efficiency
2. Applying Leiden clustering algorithm
3. Updating dataset records in DuckDB with cluster assignments
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import igraph as ig
from sklearn.neighbors import NearestNeighbors
import leidenalg

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.helpers import initialize_logging, timer_wrap
from src.models import Dataset
from api.utils.duckdb_utils import DuckDBHelper

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"


@timer_wrap
def build_knn_similarity_graph(
    dataset_embeddings: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k_neighbors: int = 12,
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
        feature_matrix: shape (n_samples, n_features) - embedding matrix
        feature_names: Optional list of feature names, defaults to ["f0", "f1", ...]
        k_neighbors: Number of nearest neighbors to consider
        similarity_threshold: Minimum similarity for edge creation
        threshold_method: Method for determining similarity threshold
        
    Returns:
        igraph.Graph object with similarity edges among features
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(dataset_embeddings.shape[1])]
    
    logger.info(f"Building k-NN graph for {len(feature_names)} features")
    
    # Transpose so rows = features
    feat_vecs = dataset_embeddings.T  # shape: (n_features, n_samples)
    
    # Build k-NN index
    logger.info(f"Computing {k_neighbors}-NN with sklearn")
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, feat_vecs.shape[0]), 
                           metric='cosine', algorithm='brute')
    nbrs.fit(feat_vecs)
    
    # Get distances and indices
    distances, indices = nbrs.kneighbors(feat_vecs)
    
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
    g = ig.Graph(n=len(feature_names), edges=edges, directed=False)
    g.es['weight'] = edge_weights
    g.vs['feature_name'] = feature_names
    
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
    resolution: float = 1.5,
    min_cluster_size: int = 6,
    max_cluster_size: int = 75,
    split_factor: float = 1.3,
    random_seed: int = 42,
    max_iter: int = 20,           # ❶ hard ceiling on passes
    stagnation_patience: int = 3  # ❷ stop if no net change for N passes
) -> Dict[str, str]:
    """
    Size-balanced Leiden clustering with explicit stop criteria.

    *   **Split big → re-cluster** with higher resolution (`resolution*split_factor`).
    *   **Merge small → nearest neighbour** by strongest connectivity.
    *   Break when:  
        – no changes for `stagnation_patience` passes or  
        – `max_iter` reached or  
        – membership pattern repeats (cycle detection).

    Returns
    -------
    {dataset_id: "cluster_<int>"}
    """
    if graph.ecount() == 0:
        logger.warning("Graph is edgeless; returning singletons")
        return {v["feature_name"]: f"cluster_{i}" for i, v in enumerate(graph.vs)}

    np.random.seed(random_seed)
    optimiser = leidenalg.Optimiser()

    def _leiden(g, res):
        part = leidenalg.RBConfigurationVertexPartition(g, resolution_parameter=res)
        optimiser.optimise_partition(part, n_iterations=-1)
        return part

    # ---------- first pass -------------------------------------------------
    part = _leiden(graph, resolution)
    logger.info(f"Initial Leiden: {len(part)} clusters; modularity={part.modularity:.4f}")

    previous_memberships = set()
    stagnation = 0

    for iteration in range(1, max_iter + 1):
        changed = False
        sizes = np.bincount(part.membership)

        # ---- 1) split oversize clusters -----------------------------------
        for cid, size in enumerate(sizes):
            if size > max_cluster_size:
                idx = [v.index for v in graph.vs if part.membership[v.index] == cid]
                sub = graph.subgraph(idx)
                sub_part = _leiden(sub, resolution * split_factor)

                if len(sub_part) > 1:
                    offset = max(part.membership) + 1
                    for sub_v, new_c in zip(sub.vs, sub_part.membership):
                        part.membership[idx[sub_v.index]] = offset + new_c
                    changed = True

        # recompute after splits
        sizes = np.bincount(part.membership)

        # ---- 2) merge undersize clusters ----------------------------------
        for cid, size in enumerate(sizes):
            if size < min_cluster_size:
                small_idx = [v.index for v in graph.vs if part.membership[v.index] == cid]
                nbr_counts = {}
                for vid in small_idx:
                    for nb in graph.vs[vid].neighbors():
                        tgt = part.membership[nb.index]
                        if tgt != cid:
                            nbr_counts[tgt] = nbr_counts.get(tgt, 0) + 1
                if nbr_counts:
                    target = max(nbr_counts, key=nbr_counts.get)
                    for vid in small_idx:
                        part.membership[vid] = target
                    changed = True

        # ---------- stopping logic -----------------------------------------
        mem_tuple = tuple(part.membership)
        if mem_tuple in previous_memberships:
            logger.info("Cycle detected; stopping at iteration %d", iteration)
            break
        previous_memberships.add(mem_tuple)

        if not changed:
            stagnation += 1
            if stagnation >= stagnation_patience:
                logger.info("No changes for %d passes; converged", stagnation_patience)
                break
        else:
            stagnation = 0

    else:
        logger.warning("Reached max_iter=%d without full convergence", max_iter)

    # ---------- build mapping ---------------------------------------------
    assignments = {v["feature_name"]: f"cluster_{part.membership[v.index]}" for v in graph.vs}
    final_sizes = np.bincount(part.membership)
    logger.info("Final: %d clusters (min=%d │ median=%d │ max=%d)",
                len(final_sizes), final_sizes.min(), int(np.median(final_sizes)),
                final_sizes.max())
    return assignments

@timer_wrap
def find_resolution_for_target(
    graph: ig.Graph,
    target_n: int,
    tol: int = 2,               # acceptable ± window
    min_cluster_size: int = 3,  # still guard against singletons
    res_low: float = 0.5,
    res_high: float = 8.0,
    max_steps: int = 10,
    **kwargs
):
    """
    Binary-search the Leiden resolution_parameter so that the
    number of clusters is within `target_n ± tol`.

    Parameters
    ----------
    graph : ig.Graph
    target_n : int
        Desired cluster count (e.g. 48).
    tol : int
        Acceptable deviation (e.g. 2 ⇒ 46–50).
    res_low, res_high : float
        Bracketing resolution values.
    max_steps : int
        Safety cap on iterations.
    kwargs :
        Extra args forwarded to `run_leiden_clustering`
        (min/max size, split_factor, etc.).

    Returns
    -------
    (assignments, resolution) : (Dict[str,str], float)
    """
    for step in range(max_steps):
        res_mid = (res_low + res_high) / 2
        assignments = run_leiden_clustering(
            graph,
            resolution=res_mid,
            min_cluster_size=min_cluster_size,
            **kwargs
        )
        n = len(set(assignments.values()))
        logger.info(
            f"[res search] step {step:02d}  γ={res_mid:.3f} → {n} clusters"
        )

        if abs(n - target_n) <= tol:
            return assignments, res_mid

        if n < target_n:
            # too few clusters → raise γ
            res_low = res_mid
        else:
            # too many clusters → lower γ
            res_high = res_mid

    logger.warning(
        "Resolution search hit max_steps=%d; returning best effort", max_steps
    )
    return assignments, res_mid


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
            success = db_helper.update_datasets(updated_datasets)
            if success:
                logger.info(f"Successfully updated {updated_count} dataset cluster assignments")
                return True
            else:
                logger.error("Failed to update dataset cluster assignments")
                return False
        else:
            logger.warning("No datasets found to update with cluster assignments")
            return False
            
    except Exception as e:
        logger.error(f"Error updating dataset clusters: {e}")
        return False


@timer_wrap
def export_feature_clusters(feat2cluster: Dict[str, str], out_dir: str) -> bool:
    """
    Export feature cluster mapping to JSON file.
    
    Args:
        feat2cluster: Dictionary mapping feature_name to cluster_label
        out_dir: Output directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(out_dir) / "feature_clusters.json"
        
        with open(output_file, "w") as fh:
            json.dump(feat2cluster, fh, indent=2)
            
        logger.info(f"Feature clusters exported to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting feature clusters: {e}")
        return False


@timer_wrap
def export_clustering_report(
    cluster_assignments: Dict[str, str],
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
            # "cluster_assignments": cluster_assignments,
        }
        
        # Write report to file
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Clustering report exported to {report_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting clustering report: {e}")
        return False

# @timer_wrap
# def run_clustering_pipeline(
#     dataset_embeddings: Dict[str, np.ndarray],
#     k_neighbors: int = 30,
#     similarity_threshold: float = None,
#     threshold_method: str = "degree_target",
#     resolution: float = 1.5,
#     min_cluster_size: int = 6,
#     max_cluster_size: Optional[int] = None,
#     split_factor: float = 1.3,
#     random_seed: int = 42,
#     target_n: int = 48,
#     tol: int = 2,
#     db_path: str = DEFAULT_DUCKDB_PATH,
#     output_dir: str = "reports/clustering"
# ) -> bool:
#     """
#     Run the complete clustering pipeline.
#     """
#     logger.info("Running clustering pipeline")
#     db_helper = DuckDBHelper(db_path)
#     try:
#         graph = build_knn_similarity_graph(dataset_embeddings, k_neighbors, similarity_threshold, threshold_method)
#         graph_stats = {
#             "vertices": graph.vcount(),
#             "edges": graph.ecount(),
#             "avg_degree": 2*graph.ecount()/graph.vcount()
#         }
#         # cluster_assignments = run_leiden_clustering(graph, resolution, min_cluster_size,max_cluster_size, split_factor, random_seed)
#         cluster_assignments, gamma = find_resolution_for_target(
#         graph,
#         target_n=target_n,            # aim for ~48 clusters
#         tol=tol,
#         min_cluster_size=min_cluster_size,
#         max_cluster_size=max_cluster_size,  # effectively disable the upper bound
#         split_factor=split_factor,
#         random_seed=random_seed)
#         logger.info("Final γ = %.3f yielded %d clusters", gamma,
#                 len(set(cluster_assignments.values())))
#         update_dataset_clusters_in_duckdb(cluster_assignments, db_helper)
#         return export_clustering_report(cluster_assignments, graph_stats, output_dir)
#     except Exception as e:
#         logger.error(f"Error running clustering pipeline: {e}")
#         return False


@timer_wrap
def run_clustering_pipeline(
    dataset_embeddings: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k_neighbors: int = 12,
    similarity_threshold: float = None,
    threshold_method: str = "degree_target",
    target_n: int = 48,
    tol: int = 2,
    min_cluster_size: int = 3,
    max_cluster_size: Optional[int] = None,
    split_factor: float = 1.3,
    random_seed: int = 42,
    output_dir: str = "reports/clustering"
) -> Dict[str, str]:
    """
    Run feature-level clustering pipeline:
    1. Build k-NN graph among features (columns)
    2. Apply Leiden clustering to group similar features
    3. Compute PCA within each feature cluster
    4. Return reduced feature matrix and cluster mapping
    
    Args:
        feature_matrix: Input matrix (n_samples, n_features)
        feature_names: Optional feature names, defaults to ["f0", "f1", ...]
        k_neighbors: Number of nearest neighbors for graph construction
        similarity_threshold: Minimum similarity for edge creation
        threshold_method: Method for determining similarity threshold
        target_n: Target number of feature clusters
        tol: Acceptable deviation from target_n
        min_cluster_size: Minimum cluster size
        max_cluster_size: Maximum cluster size (None to disable)
        split_factor: Factor for splitting large clusters
        random_seed: Random seed for reproducibility
        n_components: Number of PCA components per cluster
        output_dir: Output directory for reports
        
    Returns:
        Tuple of (reduced_feature_matrix, feature_cluster_map)
    """
    logger.info(f"Running feature clustering pipeline on {dataset_embeddings.shape} matrix")
    
    try:
        # Build k-NN graph among features
        graph = build_knn_similarity_graph(
            dataset_embeddings=dataset_embeddings,
            feature_names=feature_names,
            k_neighbors=k_neighbors,
            similarity_threshold=similarity_threshold,
            threshold_method=threshold_method
        )
        
        # Find optimal resolution for target cluster count
        feature_cluster_map, gamma = find_resolution_for_target(
            graph=graph,
            target_n=target_n,
            tol=tol,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            split_factor=split_factor,
            random_seed=random_seed
        )
        
        logger.info(f"Final γ = {gamma:.3f} yielded {len(set(feature_cluster_map.values()))} clusters")
        
        # Export feature clusters for inspection
        export_feature_clusters(feature_cluster_map, output_dir)
        graph_stats = {
            "vertices": graph.vcount(),
            "edges": graph.ecount(),
            "avg_degree": 2*graph.ecount()/graph.vcount()
        }

        # Export clustering report
        export_clustering_report(feature_cluster_map, graph_stats, output_dir)
        
        logger.info(f"Feature clustering pipeline completed successfully")
        
        return feature_cluster_map
        
    except Exception as e:
        logger.error(f"Error running feature clustering pipeline: {e}")
        raise


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