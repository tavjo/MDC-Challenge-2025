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
import pandas as pd

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
import umap as umap_lib
EPS = 1e-8          # threshold for “effectively zero variance”
    
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# try:
#     SKLEARN_AVAILABLE = True
# except ImportError:
#     SKLEARN_AVAILABLE = False

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
            self.db_helper.close()
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

    @timer_wrap  
    def run_umap_reduction(
        self,
        embeddings: Dict[str, np.ndarray],
        n_neighbors: int = None,
        min_dist: float = None,
        n_components: int = None,
        random_seed: int = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Apply UMAP dimensionality reduction with reproducibility:
        - Set random_state for deterministic embedding layout
        - Save UMAP_1, UMAP_2 features to DuckDB via upsert_engineered_features_batch
        """
        # if not UMAP_AVAILABLE or umap_lib is None:
        #     logger.error("❌ UMAP is not available. Please install umap-learn")
        #     return None
            
        # Use instance defaults if not provided
        n_neighbors = n_neighbors or self.n_neighbors
        min_dist = min_dist or self.min_dist
        n_components = n_components or self.n_components
        random_seed = random_seed or self.random_seed
        
        logger.info(f"🔄 Running UMAP reduction on {len(embeddings)} embeddings...")
        logger.info(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, "
                   f"n_components={n_components}, random_state={random_seed}")
        
        try:
            # Prepare embeddings matrix
            dataset_ids = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[dataset_id] for dataset_id in dataset_ids])
            # embedding_matrix = embedding_matrix.T
            
            logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
            
            # Initialize and fit UMAP
            reducer = umap_lib.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_seed,
                verbose=True
            )
            
            # Fit and transform embeddings
            umap_embeddings = reducer.fit_transform(embedding_matrix)
            logger.info(f"UMAP embeddings shape: {umap_embeddings.shape}")
            
            # Create dictionary mapping dataset_id to UMAP coordinates
            umap_results = {}
            for i, dataset_id in enumerate(dataset_ids):
                umap_results[dataset_id] = umap_embeddings[i]
            
            # Save UMAP features to DuckDB
            self._save_umap_features_to_duckdb(umap_results)
            
            logger.info("✅ UMAP reduction completed successfully!")
            return umap_results
            
        except Exception as e:
            logger.error(f"UMAP reduction failed: {str(e)}")
            return None

    def _save_umap_features_to_duckdb(self, umap_results: Dict[str, np.ndarray]) -> bool:
        """Save UMAP features to DuckDB using EngineeredFeatures model."""
        try:
            logger.info(f"Saving UMAP features for {len(umap_results)} datasets to DuckDB...")
            
            # Load datasets if not already loaded
            if self.datasets is None:
                self.datasets = self.load_datasets_from_duckdb()
            
            if self.datasets is None:
                logger.error("Failed to load datasets from DuckDB")
                return False
            
            # Create mapping from dataset_id to document_id
            dataset_to_doc = {ds.dataset_id: ds.document_id for ds in self.datasets}

            # turn into pandas dataframe
            df = pd.DataFrame(umap_results) # rows should be UMAP_1 and UMAP_2 and columns should be dataset_id
            
            # Create EngineeredFeatures objects
            features_list = []
            for dataset_id in df.columns:
                umap_coords = df[[dataset_id]].values
                if len(umap_coords) == 0:
                    logger.error(f"No UMAP coordinates found for dataset {dataset_id}")
                    continue
                if dataset_id in dataset_to_doc:
                    features = EngineeredFeatures(
                        dataset_id=dataset_id,
                        document_id=dataset_to_doc[dataset_id],
                        UMAP_1=float(umap_coords[0]),
                        UMAP_2=float(umap_coords[1]) if len(umap_coords) > 1 else 0.0,
                    )
                    features_list.append(features)
            
            # Batch upsert to DuckDB
            success = self.db_helper.upsert_engineered_features_batch(features_list)
            self.db_helper.close()
            if success:
                logger.info(f"✅ Successfully saved UMAP features for {len(features_list)} datasets")
                return True
            else:
                logger.error("❌ Failed to save UMAP features to DuckDB")
                return False
                
        except Exception as e:
            logger.error(f"Error saving UMAP features: {str(e)}")
            return False

    @timer_wrap
    def create_cluster_visualization(
        self,
        umap_embeddings: Dict[str, np.ndarray],
        output_path: str = "reports/dimensionality_reduction/cluster_visualization.png"
    ) -> bool:
        """Create scatter plot colored by Leiden cluster labels using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("❌ Matplotlib is not available. Please install matplotlib")
            return False
            
        try:
            logger.info("🎨 Creating cluster visualization...")
            
            # Load datasets if not already loaded
            if self.datasets is None:
                self.datasets = self.load_datasets_from_duckdb()
            
            if self.datasets is None:
                logger.error("Failed to load datasets from DuckDB")
                return False
            
            # Create mapping from dataset_id to cluster
            dataset_to_cluster = {ds.dataset_id: ds.cluster for ds in self.datasets}
            
            # Prepare data for plotting
            x_coords = []
            y_coords = []
            clusters = []
            
            for dataset_id, coords in umap_embeddings.items():
                if dataset_id in dataset_to_cluster:
                    x_coords.append(coords[0])
                    y_coords.append(coords[1] if len(coords) > 1 else 0)
                    clusters.append(dataset_to_cluster[dataset_id] or "None")
            
            if not x_coords:
                logger.warning("No datasets with cluster assignments found for visualization")
                return False
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Get unique clusters and assign colors
            unique_clusters = list(set(clusters))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster in enumerate(unique_clusters):
                cluster_mask = [c == cluster for c in clusters]
                cluster_x = [x for x, mask in zip(x_coords, cluster_mask) if mask]
                cluster_y = [y for y, mask in zip(y_coords, cluster_mask) if mask]
                
                plt.scatter(cluster_x, cluster_y, c=[colors[i]], label=f"Cluster {cluster}", alpha=0.7, s=50)
            
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.title("Dataset Embeddings Visualization - Colored by Leiden Clusters")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Cluster visualization saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create cluster visualization: {str(e)}")
            return False

    @timer_wrap  
    def create_ground_truth_visualization(
        self,
        umap_embeddings: Dict[str, np.ndarray], 
        output_path: str = "reports/dimensionality_reduction/ground_truth_visualization.png"
    ) -> bool:
        """Create scatter plot colored by dataset_type (PRIMARY/SECONDARY) labels using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("❌ Matplotlib is not available. Please install matplotlib")
            return False
            
        try:
            logger.info("🎨 Creating ground truth visualization...")
            
            # Load datasets if not already loaded
            if self.datasets is None:
                self.datasets = self.load_datasets_from_duckdb()
            
            if self.datasets is None:
                logger.error("Failed to load datasets from DuckDB")
                return False
            
            # Create mapping from dataset_id to dataset_type
            dataset_to_type = {ds.dataset_id: ds.dataset_type for ds in self.datasets}
            
            # Prepare data for plotting
            x_coords = []
            y_coords = []
            types = []
            
            for dataset_id, coords in umap_embeddings.items():
                if dataset_id in dataset_to_type:
                    x_coords.append(coords[0])
                    y_coords.append(coords[1] if len(coords) > 1 else 0)
                    types.append(dataset_to_type[dataset_id] or "Unknown")
            
            if not x_coords:
                logger.warning("No datasets with type labels found for visualization")
                return False
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            
            # Define colors for each type
            type_colors = {
                "PRIMARY": "#1f77b4",    # Blue
                "SECONDARY": "#ff7f0e",  # Orange
                "Unknown": "#d62728"     # Red
            }
            
            for dataset_type, color in type_colors.items():
                type_mask = [t == dataset_type for t in types]
                if not any(type_mask):
                    continue
                    
                type_x = [x for x, mask in zip(x_coords, type_mask) if mask]
                type_y = [y for y, mask in zip(y_coords, type_mask) if mask]
                
                plt.scatter(type_x, type_y, c=color, label=dataset_type, alpha=0.7, s=50)
            
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.title("Dataset Embeddings Visualization - Colored by Ground Truth Labels")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Ground truth visualization saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create ground truth visualization: {str(e)}")
            return False

    
    def run_pipeline(self) -> dict:
        """
        Complete pipeline: load embeddings/datasets → UMAP → save UMAP features 
        → visualizations → per-cluster PCA → save PCA features
        """
        results = {
            "load_embeddings": None,
            "load_datasets": None,
            "umap_reduction": None,
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
        
        # Step 3: Run UMAP reduction
        logger.info("🔄 Running UMAP dimensionality reduction...")
        umap_embeddings = self.run_umap_reduction(dataset_embeddings)
        results["umap_reduction"] = {"success": umap_embeddings is not None}
        
        if umap_embeddings is None:
            logger.error("❌ UMAP reduction failed, skipping visualizations")
        # else:
        #     # Step 4: Create cluster visualization
        #     logger.info("🎨 Creating cluster visualization...")
        #     cluster_viz_success = self.create_cluster_visualization(umap_embeddings)
        #     results["cluster_visualization"] = {"success": cluster_viz_success}
            
        #     # Step 5: Create ground truth visualization
        #     logger.info("🎨 Creating ground truth visualization...")
        #     gt_viz_success = self.create_ground_truth_visualization(umap_embeddings)
        #     results["ground_truth_visualization"] = {"success": gt_viz_success}
        
        # Overall success
        results["overall_success"] = all([
            results["load_embeddings"]["success"],
            results["load_datasets"]["success"],
            results["umap_reduction"]["success"],
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
    if not MATPLOTLIB_AVAILABLE:
        missing_deps.append("matplotlib")
    # if not SKLEARN_AVAILABLE:
    #     missing_deps.append("scikit-learn")
    
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