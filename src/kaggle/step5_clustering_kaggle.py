"""
Kaggle version of clustering utilities, mirroring src/clustering.py.

- Builds k-NN similarity graph over feature (column) space
- Runs Leiden (igraph + leidenalg) with the same logic
- Optionally searches resolution to hit a target number of clusters
- Writes small reports to /kaggle/tmp/reports/clustering

NOTE: This module is intentionally siloed for Kaggle; it does not depend on
api/utils or non-kaggle code paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np
import json
import igraph as ig
from sklearn.neighbors import NearestNeighbors
import leidenalg

from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

# Clustering parameters
DEFAULT_K_NEIGHBORS = 5
DEFAULT_SIMILARITY_THRESHOLD = None
DEFAULT_THRESHOLD_METHOD = "degree_target"
DEFAULT_RESOLUTION = 1
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MAX_CLUSTER_SIZE = 9999
DEFAULT_SPLIT_FACTOR = 1.3
DEFAULT_RANDOM_SEED = 42
DEFAULT_TARGET_N = 50
DEFAULT_TOL = 2


# ------------------------------ Graph build ------------------------------
def build_knn_similarity_graph(
    dataset_embeddings: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k_neighbors: int = 12,
    similarity_threshold: float | None = None,
    threshold_method: str = "degree_target",
) -> ig.Graph:
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(dataset_embeddings.shape[1])]

    # Transpose so rows = features
    feat_vecs = dataset_embeddings.T  # shape: (n_features, n_samples)

    # k-NN in feature space (cosine distance)
    nbrs = NearestNeighbors(
        n_neighbors=min(k_neighbors, feat_vecs.shape[0]), metric="cosine", algorithm="brute"
    )
    nbrs.fit(feat_vecs)
    distances, indices = nbrs.kneighbors(feat_vecs)

    if similarity_threshold is None:
        similarity_threshold = determine_similarity_threshold(
            distances, method=threshold_method, target_degree=15
        )

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for dist, j in zip(dist_row, idx_row):
            if i == j:
                continue
            sim = 1 - float(dist)
            if sim >= similarity_threshold:
                edges.append((i, j))
                weights.append(sim)

    g = ig.Graph(n=len(feature_names), edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["feature_name"] = feature_names
    return g


def determine_similarity_threshold(
    distances: np.ndarray,
    method: str = "degree_target",
    target_degree: int = 15,
) -> float:
    similarities = 1 - distances.flatten()
    similarities = similarities[similarities < 1.0]

    if method == "degree_target":
        sorted_sims = np.sort(similarities)[::-1]
        n_nodes = distances.shape[0]
        target_edges = (target_degree * n_nodes) // 2
        if target_edges >= len(sorted_sims):
            threshold = float(sorted_sims[-1])
        else:
            threshold = float(sorted_sims[target_edges])
        return threshold

    if method == "percentile_90":
        return float(np.percentile(similarities, 90))

    if method == "elbow_method":
        sorted_sims = np.sort(similarities)[::-1]
        gaps = np.diff(sorted_sims)
        max_gap_idx = int(np.argmax(np.abs(gaps)))
        return float(sorted_sims[max_gap_idx])

    raise ValueError(f"Unknown threshold method: {method}")


# ------------------------------ Leiden ------------------------------
def run_leiden_clustering(
    graph: ig.Graph,
    resolution: float = 1.5,
    min_cluster_size: int = 6,
    max_cluster_size: int = 75,
    split_factor: float = 1.3,
    random_seed: int = 42,
    max_iter: int = 20,
    stagnation_patience: int = 3,
) -> Dict[str, str]:
    if graph.ecount() == 0:
        return {v["feature_name"]: f"cluster_{i}" for i, v in enumerate(graph.vs)}

    np.random.seed(random_seed)
    optimiser = leidenalg.Optimiser()

    def _leiden(g: ig.Graph, res: float):
        part = leidenalg.RBConfigurationVertexPartition(g, resolution_parameter=res)
        optimiser.optimise_partition(part, n_iterations=-1)
        return part

    part = _leiden(graph, resolution)
    previous_memberships: set[Tuple[int, ...]] = set()
    stagnation = 0

    for _ in range(1, max_iter + 1):
        changed = False
        sizes = np.bincount(part.membership)

        # split oversize
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

        sizes = np.bincount(part.membership)

        # merge undersize
        for cid, size in enumerate(sizes):
            if size < min_cluster_size:
                small_idx = [v.index for v in graph.vs if part.membership[v.index] == cid]
                nbr_counts: Dict[int, int] = {}
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

        mem_tuple = tuple(part.membership)
        if mem_tuple in previous_memberships:
            break
        previous_memberships.add(mem_tuple)

        if not changed:
            stagnation += 1
            if stagnation >= stagnation_patience:
                break
        else:
            stagnation = 0

    assignments = {v["feature_name"]: f"cluster_{part.membership[v.index]}" for v in graph.vs}
    return assignments


def find_resolution_for_target(
    graph: ig.Graph,
    target_n: int,
    tol: int = 2,
    min_cluster_size: int = 3,
    res_low: float = 0.5,
    res_high: float = 8.0,
    max_steps: int = 10,
    **kwargs,
) -> Tuple[Dict[str, str], float]:
    assignments: Dict[str, str] = {}
    res_mid = res_low
    for _ in range(max_steps):
        res_mid = (res_low + res_high) / 2
        assignments = run_leiden_clustering(
            graph,
            resolution=res_mid,
            min_cluster_size=min_cluster_size,
            **kwargs,
        )
        n = len(set(assignments.values()))
        if abs(n - target_n) <= tol:
            return assignments, res_mid
        if n < target_n:
            res_low = res_mid
        else:
            res_high = res_mid
    return assignments, res_mid


# ------------------------------ Reports ------------------------------
def export_feature_clusters(feat2cluster: Dict[str, str], out_dir: str) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "feature_clusters.json"
    with open(output_file, "w") as fh:
        json.dump(feat2cluster, fh, indent=2)
    return output_file


def export_clustering_report(
    cluster_assignments: Dict[str, str],
    graph_stats: Dict[str, Any],
    output_dir: str = "/kaggle/tmp/reports/clustering",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = out_dir / f"clustering_report_{timestamp}.json"

    cluster_sizes: Dict[str, int] = {}
    for lab in cluster_assignments.values():
        cluster_sizes[lab] = cluster_sizes.get(lab, 0) + 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_features": len(cluster_assignments),
            "total_clusters": len(cluster_sizes),
            "avg_cluster_size": (sum(cluster_sizes.values()) / len(cluster_sizes)) if cluster_sizes else 0,
            "largest_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "smallest_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
        },
        "graph_stats": graph_stats,
        "cluster_sizes": cluster_sizes,
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    return report_file


# ------------------------------ Orchestrator ------------------------------
def run_clustering_pipeline(
    dataset_embeddings: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    similarity_threshold: float | None = DEFAULT_SIMILARITY_THRESHOLD,
    threshold_method: str = DEFAULT_THRESHOLD_METHOD,
    target_n: int = DEFAULT_TARGET_N,
    tol: int = DEFAULT_TOL,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    max_cluster_size: Optional[int] = DEFAULT_MAX_CLUSTER_SIZE,
    split_factor: float = DEFAULT_SPLIT_FACTOR,
    random_seed: int = DEFAULT_RANDOM_SEED,
    report_dir: str = "/kaggle/tmp/reports/clustering",
) -> Dict[str, str]:
    graph = build_knn_similarity_graph(
        dataset_embeddings=dataset_embeddings,
        feature_names=feature_names,
        k_neighbors=k_neighbors,
        similarity_threshold=similarity_threshold,
        threshold_method=threshold_method,
    )
    feature_cluster_map, gamma = find_resolution_for_target(
        graph=graph,
        target_n=target_n,
        tol=tol,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size if max_cluster_size is not None else 10 ** 9,
        split_factor=split_factor,
        random_seed=random_seed,
    )

    export_feature_clusters(feature_cluster_map, report_dir)
    graph_stats = {
        "vertices": graph.vcount(),
        "edges": graph.ecount(),
        "avg_degree": (2 * graph.ecount() / graph.vcount()) if graph.vcount() > 0 else 0,
        "resolution": gamma,
    }
    export_clustering_report(feature_cluster_map, graph_stats, report_dir)
    return feature_cluster_map


__all__ = [
    "build_knn_similarity_graph",
    "determine_similarity_threshold",
    "run_leiden_clustering",
    "find_resolution_for_target",
    "export_feature_clusters",
    "export_clustering_report",
    "run_clustering_pipeline",
]


