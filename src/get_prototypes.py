# Generate Prototypes from Training Set
"""
In this script, we will generate prototype embeddings from the training set.
This script will be a mirror of the clustering + per cluster PCA process, but with the nodes being `Dataset` objects instead of features.
The output of this script will be combined with the existing 4 global prototypes (UMAP1,UMAP2, PCA2, PCA3) to form N clusters + 4 prototypes that can help retrieve relevant snippets of text from new papers.
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

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "load_embeddings": "/load_embeddings"
}

# Default paths and configurations
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DEFAULT_COLLECTION_NAME = "dataset-aggregates-train"
DEFAULT_FEATURE_CLUSTERS_PATH = "reports/clustering/dataset_clusters.json"

# UMAP parameters
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_N_COMPONENTS = 2
DEFAULT_RANDOM_SEED = 42

# Clustering parameters
DEFAULT_K_NEIGHBORS = 3
DEFAULT_SIMILARITY_THRESHOLD = None
DEFAULT_THRESHOLD_METHOD = "degree_target"
DEFAULT_RESOLUTION = 1
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MAX_CLUSTER_SIZE = 9999
DEFAULT_SPLIT_FACTOR = 1.3
DEFAULT_RANDOM_SEED = 42
DEFAULT_TARGET_N = 100
DEFAULT_TOL = 2

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
        feature_names = [f"f{i}" for i in range(dataset_embeddings.shape[0])]
    
    logger.info(f"Building k-NN graph for {len(feature_names)} features")
    
    # Transpose so rows = features
    feat_vecs = dataset_embeddings#.T  # shape: (n_features, n_samples)
    
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
        output_file = Path(out_dir) / "dataset_clusters.json"
        
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

################################################################################
# Per-cluster PCA
################################################################################
"""
Dimensionality Reduction Pipeline for MDC-Challenge-2025
Loads embeddings from ChromaDB, runs UMAP + per-cluster PCA, creates visualizations,
and saves engineered features to DuckDB.
"""

import json
import os
import sys
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.models import LoadChromaDataPayload, LoadChromaDataResult, Dataset
# from src.helpers import initialize_logging, timer_wrap
# from api.utils.duckdb_utils import DuckDBHelper
import requests

# try:
#     import umap as umap_lib
#     UMAP_AVAILABLE = True
# except ImportError:
#     umap_lib = None
#     UMAP_AVAILABLE = False
# import umap
EPS = 1e-8          # threshold for “effectively zero variance”
    
# try:
#     import matplotlib.pyplot as plt
#     MATPLOTLIB_AVAILABLE = True
# except ImportError:
#     MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    # from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor, as_completed


class Reducer:
    """
    Dimensionality reduction pipeline following similar pattern to ClusteringPipeline.
    Loads embeddings via API, runs UMAP + per-cluster PCA, creates visualizations,
    and saves engineered features to DuckDB.
    """
    
    def __init__(self,
                 base_api_url: str = API_ENDPOINTS["base_api_url"],
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 cfg_path: Optional[str] = DEFAULT_CHROMA_CONFIG,
                 db_path: str = DEFAULT_DUCKDB_PATH,
                 n_neighbors: int = DEFAULT_N_NEIGHBORS,
                 min_dist: float = DEFAULT_MIN_DIST,
                 n_components: int = DEFAULT_N_COMPONENTS,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        self.base_api_url = base_api_url
        self.collection_name = collection_name
        self.cfg_path = cfg_path
        self.db_path = db_path
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.random_seed = random_seed
        self.endpoints = API_ENDPOINTS
        self.db_helper = DuckDBHelper(db_path)
        self.datasets: Optional[List[Dataset]] = None

    def _get_api_url(self, endpoint: str) -> str:
        return f"{self.base_api_url}{self.endpoints[endpoint]}"

    def _construct_load_embeddings_payload(self) -> LoadChromaDataPayload:
        return LoadChromaDataPayload(
            collection_name=self.collection_name,
            cfg_path=self.cfg_path,
            include=["embeddings"]  # Only need embeddings for dimensionality reduction
        )

    def load_embeddings_from_api(self) -> Optional[np.ndarray]:
        """Load embeddings from ChromaDB via API call."""
        payload = self._construct_load_embeddings_payload()
        full_url = self._get_api_url("load_embeddings")
        
        logger.info(f"Loading embeddings with URL: {full_url}")
        logger.info(f"Using collection: {self.collection_name}")
        
        try:
            response = requests.post(full_url, json=payload.model_dump(exclude_none=True))
            if response.status_code == 200:
                result = LoadChromaDataResult.model_validate(response.json())
                if result.success and result.results:
                    logger.info(f"✅ Successfully loaded {len(result.results)} embeddings")
                    
                    # Convert embeddings to numpy arrays if they aren't already
                    dat = result.results
                    dataset_embeddings = np.array(dat["embeddings"])
                    logger.info(f"✅ Embeddings shape: {dataset_embeddings.shape}")
                    
                    return dataset_embeddings
                else:
                    error_msg = result.error or "No embeddings returned"
                    logger.error(f"❌ Failed to load embeddings: {error_msg}")
                    return None
            else:
                logger.error(f"Load embeddings failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Load embeddings request failed: {str(e)}")
            return None

    def load_datasets_from_duckdb(self) -> Optional[List[Dataset]]:
        """Load all Dataset objects from DuckDB and persist for reuse across methods."""
        try:
            logger.info("Loading datasets from DuckDB...")
            datasets = self.db_helper.get_all_datasets()
            logger.info(f"✅ Successfully loaded {len(datasets)} datasets from DuckDB")
            # self.db_helper.close()
            self.datasets = datasets
            return datasets
        except Exception as e:
            logger.error(f"Failed to load datasets from DuckDB: {str(e)}")
            return None

    def load_feature_clusters(self, feature_clusters_path: str = DEFAULT_FEATURE_CLUSTERS_PATH) -> Optional[Dict[str, str]]:
        """Load feature cluster mapping from JSON file."""
        try:
            logger.info(f"Loading feature clusters from {feature_clusters_path}...")
            with open(feature_clusters_path, 'r') as f:
                feature_cluster_map = json.load(f)
            logger.info(f"✅ Successfully loaded {len(feature_cluster_map)} feature cluster mappings")
            return feature_cluster_map
        except Exception as e:
            logger.error(f"Failed to load feature clusters: {str(e)}")
            return None

    def _run_pca_on_cluster(
        self,
        cluster_id: str,
        ds_idx: List[int],
        dataset_embeddings: np.ndarray,
        random_seed: int,
    ) -> np.ndarray:
        """
        Run PCA on a *feature* cluster; return the PC1 projection for every sample.

        Parameters
        ----------
        cluster_id   : cluster label (e.g. "cluster_7")
        feature_idx  : column indices belonging to this cluster
       dataset_embeddings            : full design matrix, shape (n_samples, n_features)
        random_seed  : reproducibility

        Returns
        -------
        np.ndarray   : PC1 scores, shape (n_samples,)
        """
        # Transpose the dataset embeddings to match the expected shape
        try:
            if len(ds_idx) < 2:
                logger.warning(
                    f"Dataset-cluster {cluster_id} has <2 datasets – skipping PCA"
                )
                # fallback → values of the single column
                return dataset_embeddings[ds_idx[0],:].copy()

            # slice once; shape (n_samples, n_cluster_features)
            X_sub = dataset_embeddings[ds_idx, :]
            X_sub = X_sub.T
            logger.info(f"Dataset cluster {cluster_id} shape: {X_sub.shape}")

            pca = PCA(
                n_components=1,
                random_state=random_seed,
                svd_solver="full",
            )
            pc1 = pca.fit_transform(X_sub).ravel()

            var_ratio = float(pca.explained_variance_ratio_[0])
            logger.info(
                f"{cluster_id}: PC1 var ratio = {var_ratio:.4f}"
            )
            return pc1

        except Exception as exc:
            logger.error(f"PCA failed for {cluster_id}: {exc}")
            raise

    @timer_wrap
    def run_per_cluster_pca(
        self,
        dataset_embeddings: np.ndarray,                         # shape (n_samples, n_features)
        dataset_cluster_map: Dict[str, str],   # {feature_name: cluster_label}
        dataset_ids: List[str],                 # ordered like X rows
        random_seed: int = None,
    ) -> bool:
        """
        Group features by cluster, run PCA on each feature cluster:
        - Keep PC1 (highest explained variance) for each cluster
        - Save as LEIDEN_1, LEIDEN_2, etc. features to DuckDB
        - Fail gracefully if no cluster assignments exist
        """
        if not SKLEARN_AVAILABLE:
            logger.error("❌ Scikit-learn is not available. Please install scikit-learn")
            return False
            
        random_seed = random_seed or self.random_seed
        
        try:
            logger.info("🔄 Running feature-cluster PCA...")
            
            # Load datasets if not already loaded
            if self.datasets is None:
                self.datasets = self.load_datasets_from_duckdb()
            
            if self.datasets is None:
                logger.error("Failed to load datasets from DuckDB")
                return False
            
            # Build index lists per cluster
            name_to_idx = {ds_id: i for i, ds_id in enumerate(dataset_ids)}
            cluster2rows: Dict[str, List[int]] = defaultdict(list)
            for ds_id, cid in dataset_cluster_map.items():  # dataset_cluster_map already computed
                cluster2rows[cid].append(name_to_idx[ds_id])

            # - Parallel compute prototypes
            comp_matrix: List[Tuple[str, np.ndarray]] = []
            with ThreadPoolExecutor(max_workers=min(8, len(cluster2rows))) as ex:
                futures = {
                    ex.submit(self._run_pca_on_cluster, cid, rows, dataset_embeddings, random_seed): cid
                    for cid, rows in cluster2rows.items()
                }
                for fut in as_completed(futures):
                    cid = futures[fut]
                    prototype = fut.result()                 # (384,)
                    comp_matrix.append((cid, prototype))
            
            # - Assemble n_clusters × 384
            comp_matrix.sort()
            labels, vectors = zip(*comp_matrix)
            prototypes = np.vstack(vectors)                  # (n_clusters, 384)

            # - Optional: DataFrame with cluster labels as index
            pca_features = pd.DataFrame(prototypes, index=labels)

            # Save PCA features
            if not pca_features.empty:
                success = self._save_prototypes_to_pickle(pca_features)
                if success:
                    logger.info("✅ Feature-cluster PCA completed successfully!")
                    return True
                logger.error("❌ Failed to save PCA features to pickle")
                return False
            logger.warning("No PCA features generated")
            return False
                
        except Exception as e:
            logger.error(f"Feature-cluster PCA failed: {str(e)}")
            return False
    
    # TODO: Add a function to save the PCA features to a pickle file
    def _save_prototypes_to_pickle(self, pca_features: pd.DataFrame) -> bool:
        """
        Save PCA features to a pickle file.
        """
        import pickle
        with open("prototypes.pkl", "wb") as f:
            pickle.dump(pca_features, f)
        return True
    
    def run_pipeline(self) -> dict:
        """
        Complete pipeline: load embeddings/datasets → UMAP → save UMAP features 
        → visualizations → per-cluster PCA → save PCA features
        """
        results = {
            "load_embeddings": None,
            "load_datasets": None,
            "cluster_visualization": None,
            "ground_truth_visualization": None,
            "per_cluster_pca": None,
            "overall_success": False
        }
        
        logger.info("🚀 Starting dimensionality reduction pipeline...")
        
        # Step 1: Load embeddings from ChromaDB
        logger.info("📥 Loading embeddings from ChromaDB...")
        dataset_embeddings = self.load_embeddings_from_api()
        results["load_embeddings"] = {"success": dataset_embeddings is not None}
        
        if dataset_embeddings is None:
            logger.error("❌ Failed to load embeddings, aborting pipeline")
            return results
        
        # Step 2: Load datasets from DuckDB
        logger.info("📥 Loading datasets from DuckDB...")
        datasets = self.load_datasets_from_duckdb()
        results["load_datasets"] = {"success": datasets is not None}
        
        if datasets is None:
            logger.error("❌ Failed to load datasets, aborting pipeline")
            return results
        
        # Step 6: Run feature-cluster PCA
        logger.info("🔄 Running feature-cluster PCA...")
        
        # Load feature clusters
        feature_cluster_map = self.load_feature_clusters()
        if feature_cluster_map is None:
            logger.warning("❌ Failed to load feature clusters, skipping feature-cluster PCA")
            pca_success = False
        else:
            # Convert embeddings dict to matrix format
            dataset_ids = list(dataset_embeddings.keys())
            dataset_embeddings = np.array([dataset_embeddings[ds_id] for ds_id in dataset_ids])
            
            # Run feature-cluster PCA
            pca_success = self.run_per_cluster_pca(dataset_embeddings, feature_cluster_map, dataset_ids)
            
        results["per_cluster_pca"] = {"success": pca_success}
        
        # Overall success
        results["overall_success"] = all([
            results["load_embeddings"]["success"],
            results["load_datasets"]["success"],
            results["per_cluster_pca"]["success"]
        ])
        
        if results["overall_success"]:
            logger.info("✅ Complete dimensionality reduction pipeline finished successfully!")
        else:
            logger.error("❌ Pipeline completed with errors")
        
        return results


@timer_wrap
def main():
    """Main function to run the dimensionality reduction pipeline."""
    
    logger.info("Initializing dimensionality reduction pipeline...")
    
    # Check dependencies
    missing_deps = []
    # if not UMAP_AVAILABLE or umap_lib is None:
    #     missing_deps.append("umap-learn")
    if not SKLEARN_AVAILABLE:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.warning("Some features may not work. Please install missing dependencies.")
    
    # Initialize the pipeline with default parameters
    pipeline = Reducer(
        collection_name=DEFAULT_COLLECTION_NAME,
        cfg_path=DEFAULT_CHROMA_CONFIG,
        db_path=DEFAULT_DUCKDB_PATH,
        n_neighbors=DEFAULT_N_NEIGHBORS,
        min_dist=DEFAULT_MIN_DIST,
        n_components=DEFAULT_N_COMPONENTS,
        random_seed=DEFAULT_RANDOM_SEED
    )
    dataset_embeddings = pipeline.load_embeddings_from_api()
    datasets = pipeline.load_datasets_from_duckdb()
    dataset_ids = [ds.dataset_id for ds in datasets]
    # turn dataset embeddings into a numpy array
    # dataset_embeddings = np.array([dataset_embeddings[ds_id] for ds_id in dataset_ids])
    logger.info(f"Dataset embeddings shape: {dataset_embeddings.shape}")
    
    logger.info("Starting clustering pipeline demonstration")
    logger.info("🔄 Starting clustering pipeline...")
    logger.info(f"Clustering {len(dataset_embeddings)} datasets with parameters:")
    logger.info(f"  - k_neighbors: {DEFAULT_K_NEIGHBORS}")
    logger.info(f"  - similarity_threshold: {DEFAULT_SIMILARITY_THRESHOLD}")
    logger.info(f"  - threshold_method: {DEFAULT_THRESHOLD_METHOD}")
    logger.info(f"  - resolution: {DEFAULT_RESOLUTION}")
    logger.info(f"  - min_cluster_size: {DEFAULT_MIN_CLUSTER_SIZE}")
    logger.info(f"  - random_seed: {DEFAULT_RANDOM_SEED}")
    logger.info(f"  - target_n: {DEFAULT_TARGET_N}")
    logger.info(f"  - tol: {DEFAULT_TOL}")
    
    try:
        dataset_cluster_map = run_clustering_pipeline(
            dataset_embeddings=dataset_embeddings,
            k_neighbors=DEFAULT_K_NEIGHBORS,
            similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
            threshold_method=DEFAULT_THRESHOLD_METHOD,
            # resolution=DEFAULT_RESOLUTION,
            min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
            max_cluster_size=DEFAULT_MAX_CLUSTER_SIZE,
            split_factor=DEFAULT_SPLIT_FACTOR,
            random_seed=DEFAULT_RANDOM_SEED,
            target_n=DEFAULT_TARGET_N,
            tol=DEFAULT_TOL,
            output_dir="reports/clustering"
        )
        
        if dataset_cluster_map:
            logger.info("✅ Clustering pipeline completed successfully!")
        else:
            logger.error("❌ Clustering pipeline failed!")
        
    except Exception as e:
        logger.error(f"Clustering pipeline failed with error: {str(e)}")

    # Run per cluster pca
    results = pipeline.run_per_cluster_pca(dataset_embeddings, dataset_cluster_map, dataset_ids)
    return results
    
    # Prepare output directory and save results
    # output_dir = "reports/dimensionality_reduction"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, "dimensionality_reduction_results.json")
    
    # # Add timestamp to results
    # results["timestamp"] = datetime.now(timezone.utc).isoformat()
    # results["parameters"] = {
    #     "n_neighbors": pipeline.n_neighbors,
    #     "min_dist": pipeline.min_dist,
    #     "n_components": pipeline.n_components,
    #     "random_seed": pipeline.random_seed
    # }
    
    # # Save results
    # logger.info(f"Saving results to: {output_file}")
    # with open(output_file, "w") as f:
    #     json.dump(results, f, indent=2)
    
    # # Summary logging
    # if results["overall_success"]:
    #     logger.info("🎉 Dimensionality reduction pipeline completed successfully!")
    # else:
    #     logger.error("💥 Pipeline failed - check the logs and results file for details")
        
    #     # Log specific failures
    #     for step, result in results.items():
    #         if isinstance(result, dict) and not result.get("success", True):
    #             logger.error(f"Step '{step}' failed")


if __name__ == "__main__":
    main()