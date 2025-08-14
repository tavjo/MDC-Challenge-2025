#!/usr/bin/env python3
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
from src.models import LoadChromaDataPayload, LoadChromaDataResult, Dataset, EngineeredFeatures
from src.helpers import initialize_logging, timer_wrap
from api.utils.duckdb_utils import DuckDBHelper
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

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "load_embeddings": "/load_embeddings"
}

# Default paths and configurations
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DEFAULT_COLLECTION_NAME = "dataset-aggregates-train"
DEFAULT_FEATURE_CLUSTERS_PATH = "reports/clustering/feature_clusters.json"

# UMAP parameters
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_N_COMPONENTS = 2
DEFAULT_RANDOM_SEED = 42


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

    def load_embeddings_from_api(self) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings from ChromaDB via API call using same endpoints as clustering."""
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
                    dataset_embeddings = {id: np.array(embeddings) for id, embeddings in zip(dat["ids"], dat["embeddings"])}
                    
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
        feature_idx: List[int],
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
        try:
            if len(feature_idx) < 2:
                logger.warning(
                    f"Feature-cluster {cluster_id} has <2 features – skipping PCA"
                )
                # fallback → values of the single column
                return dataset_embeddings[:, feature_idx[0]].copy()

            # slice once; shape (n_samples, n_cluster_features)
            X_sub = dataset_embeddings[:, feature_idx]

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
        feature_cluster_map: Dict[str, str],   # {feature_name: cluster_label}
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
            name_to_idx = {fname: i for i, fname in enumerate(feature_cluster_map)}
            cluster2cols: Dict[str, List[int]] = defaultdict(list)
            for fname, cid in feature_cluster_map.items():
                cluster2cols[cid].append(name_to_idx[fname])
            
            logger.info(f"Processing {len(cluster2cols)} feature clusters")
            
            # Parallel PCA
            with ThreadPoolExecutor(max_workers=min(8, len(cluster2cols))) as ex:
                futures = {
                    ex.submit(
                        self._run_pca_on_cluster,
                        cid, cols, dataset_embeddings, random_seed
                    ): cid
                    for cid, cols in cluster2cols.items()
                }

                # collect → stacked array
                comp_matrix = []
                for fut in as_completed(futures):
                    cid = futures[fut]
                    try:
                        pc1_result = fut.result()
                        comp_matrix.append((cid, pc1_result))
                    except Exception as e:
                        logger.error(f"PCA failed for cluster {cid}: {str(e)}")
                        continue
            
            if not comp_matrix:
                logger.warning("No PCA results generated")
                return True
            
            # Assemble reduced dataframe
            comp_matrix.sort()               # keep cluster order stable
            labels, vectors = zip(*comp_matrix)
            X_reduced = np.column_stack(vectors)   # shape (n_samples, n_clusters)
            
            logger.info(f"✅ Completed parallel PCA on {len(labels)} clusters")
            
            # Persist into DuckDB
            pca_features: Dict[str, Dict[str, float]] = {}
            for row_idx, ds_id in enumerate(dataset_ids):
                for col_idx, cid in enumerate(labels):
                    pca_features.setdefault(ds_id, {})[
                        f"LEIDEN_{cid}"
                    ] = float(X_reduced[row_idx, col_idx])

            # Save PCA features to DuckDB
            if pca_features:
                success = self._save_pca_features_to_duckdb(pca_features)
                if success:
                    logger.info("✅ Feature-cluster PCA completed successfully!")
                    return True
                else:
                    logger.error("❌ Failed to save PCA features to DuckDB")
                    return False
            else:
                logger.warning("No PCA features generated")
                return False
                
        except Exception as e:
            logger.error(f"Feature-cluster PCA failed: {str(e)}")
            return False

    def _save_pca_features_to_duckdb(self, pca_features: Dict[str, Dict[str, float]]) -> bool:
        """Save per-cluster PCA features to DuckDB."""
        try:
            logger.info(f"Saving PCA features for {len(pca_features)} datasets to DuckDB...")
            
            # Create mapping from dataset_id to document_id
            dataset_to_doc = {ds.dataset_id: ds.document_id for ds in self.datasets}
            
            # Create EngineeredFeatures objects with PCA features
            features_list = []
            for dataset_id, cluster_features in pca_features.items():
                if dataset_id in dataset_to_doc:
                    # Start with required fields (use 0.0 for UMAP if not available)
                    feature_dict = {
                        "dataset_id": dataset_id,
                        "document_id": dataset_to_doc[dataset_id],
                        "UMAP_1": 0.0,  # Will be updated if UMAP was run
                        "UMAP_2": 0.0,  # Will be updated if UMAP was run
                        "LEIDEN_1": 0.0  # Default, will be overridden
                    }
                    
                    # Add PCA features for each cluster
                    feature_dict.update(cluster_features)
                    
                    features = EngineeredFeatures(**feature_dict)
                    features_list.append(features)
            
            # Batch upsert to DuckDB
            success = self.db_helper.upsert_engineered_features_batch(features_list)
            if success:
                logger.info(f"✅ Successfully saved PCA features for {len(features_list)} datasets")
                return True
            else:
                logger.error("❌ Failed to save PCA features to DuckDB")
                return False
                
        except Exception as e:
            logger.error(f"Error saving PCA features: {str(e)}")
            return False
    
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
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Prepare output directory and save results
    output_dir = "reports/dimensionality_reduction"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dimensionality_reduction_results.json")
    
    # Add timestamp to results
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    results["parameters"] = {
        "n_neighbors": pipeline.n_neighbors,
        "min_dist": pipeline.min_dist,
        "n_components": pipeline.n_components,
        "random_seed": pipeline.random_seed
    }
    
    # Save results
    logger.info(f"Saving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary logging
    if results["overall_success"]:
        logger.info("🎉 Dimensionality reduction pipeline completed successfully!")
    else:
        logger.error("💥 Pipeline failed - check the logs and results file for details")
        
        # Log specific failures
        for step, result in results.items():
            if isinstance(result, dict) and not result.get("success", True):
                logger.error(f"Step '{step}' failed")


if __name__ == "__main__":
    main()