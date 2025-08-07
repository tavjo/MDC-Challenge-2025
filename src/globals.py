# src/globals.py

"""
In this script, we will calculate the following global variables:

## save the first 2 to DuckDB as EngineeredFeatures

1) Sample(dataset citation) UMAP embeddings (dim 1 and dim 2). The resulting dataset will have the shape: 487 (n_samples) x 2 (dim 1 and dim 2 if working on the training set. (code already exists in ``src/umap.py``) 

2) Sample(dataset citation) PCA loadings (PC1 and PC2). The resulting dataset will have the shape: 487 (n_samples) x 2 (PC1 and PC2) if working on the training set. 

## Retrieval helpers (for testing set but generated from training set): save as pickle file and load from there. 

3) Feature UMAP embeddings (dim 1 and dim 2). The resulting dataset will have the shape: 384 (n_features) x 2 (dim 1 and dim 2). Can ONLY be run on training set. However, output be used to retrieve relevant context from testing set. Run once on train set and save output to file. 

4) Feature PCA loadings (PC1 and PC2). The resulting dataset will have the shape: 384 (n_features) x 2 (PC1 and PC2). Can ONLY be run on training set. However, output can be used to retrieve relevant context from testing set. Run once on train set and save output to file. 

"""

import os, sys
from typing import Optional, Dict, List
import pandas as pd
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_DUCKDB_PATH = "../artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DEFAULT_COLLECTION_NAME = "dataset-aggregates-train"

# UMAP parameters
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_N_COMPONENTS = 2
DEFAULT_RANDOM_SEED = 42

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "load_embeddings": "/load_embeddings"
}

# from src.umap import Reducer
from src.helpers import timer_wrap, initialize_logging
from api.utils.duckdb_utils import DuckDBHelper
from src.models import LoadChromaDataPayload, LoadChromaDataResult, Dataset, EngineeredFeatures
from typing import Dict, List, Optional
import numpy as np
import requests


class GlobalFeatures:
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
        self.random_seed = random_seed
        self.endpoints = API_ENDPOINTS
        self.db_helper = DuckDBHelper(db_path)
        self.datasets: Optional[List[Dataset]] = None
        self.embeddings: Optional[Dict[str, np.ndarray]] = None

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
    
    #########################################################
    # Sample (dataset citation) UMAP + PCA
    #########################################################

    def run_sample_umap(self,
                        n_neighbors: int = None,
                        min_dist: float = None,
                        n_components: int = None,
                        random_seed: int = None) -> Optional[Dict[str, np.ndarray]]:
        from src.umap import Reducer
        reducer = Reducer(self.db_path)
        return reducer.run_umap_reduction(self.embeddings, n_neighbors, min_dist, n_components, random_seed)
    
    @timer_wrap
    def run_sample_pca(self, n_components: int = None, random_seed: int = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Apply PCA dimensionality reduction with reproducibility:
        - Set random_state for deterministic PCA components
        - Save PCA_1, PCA_2 features to DuckDB via upsert_engineered_features_batch
        """
        from sklearn.decomposition import PCA
        
        # Use instance defaults if not provided
        n_components = n_components or self.n_components
        random_seed = random_seed or self.random_seed
        
        # Load embeddings if not already loaded
        if self.embeddings is None:
            self.embeddings = self.load_embeddings_from_api()
        
        if self.embeddings is None:
            logger.error("Failed to load embeddings for PCA")
            return None
            
        logger.info(f"🔄 Running PCA reduction on {len(self.embeddings)} embeddings...")
        logger.info(f"Parameters: n_components={n_components}, random_state={random_seed}")
        
        try:
            # Prepare embeddings matrix
            dataset_ids = list(self.embeddings.keys())
            embedding_matrix = np.array([self.embeddings[dataset_id] for dataset_id in dataset_ids])
            
            logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
            
            # Initialize and fit PCA
            pca = PCA(
                n_components=n_components,
                random_state=random_seed
            )
            
            # Fit and transform embeddings
            pca_embeddings = pca.fit_transform(embedding_matrix)
            logger.info(f"PCA embeddings shape: {pca_embeddings.shape}")
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            
            # Create dictionary mapping dataset_id to PCA coordinates
            pca_results = {}
            for i, dataset_id in enumerate(dataset_ids):
                pca_results[dataset_id] = pca_embeddings[i]
            
            # Save PCA features to DuckDB
            self._save_pca_features_to_duckdb(pca_results)
            
            logger.info("✅ PCA reduction completed successfully!")
            return pca_results
            
        except Exception as e:
            logger.error(f"PCA reduction failed: {str(e)}")
            return None

    def _save_pca_features_to_duckdb(self, pca_results: Dict[str, np.ndarray]) -> bool:
        """Save PCA features to DuckDB using EngineeredFeatures model."""
        try:
            logger.info(f"Saving PCA features for {len(pca_results)} datasets to DuckDB...")
            
            # Load datasets if not already loaded
            if self.datasets is None:
                self.datasets = self.load_datasets_from_duckdb()
            
            if self.datasets is None:
                logger.error("Failed to load datasets from DuckDB")
                return False
            
            # Create mapping from dataset_id to document_id
            dataset_to_doc = {ds.dataset_id: ds.document_id for ds in self.datasets}

            # turn into pandas dataframe
            df = pd.DataFrame(pca_results) # rows should be PCA_1 and PCA_2 and columns should be dataset_id
            
            # Create EngineeredFeatures objects
            features_list = []
            for dataset_id in df.columns:
                pca_coords = df[[dataset_id]].values
                if len(pca_coords) == 0:
                    logger.error(f"No PCA coordinates found for dataset {dataset_id}")
                    continue
                if dataset_id in dataset_to_doc:
                    features = EngineeredFeatures(
                        dataset_id=dataset_id,
                        document_id=dataset_to_doc[dataset_id],
                        UMAP_1=0.0,  # Placeholder, will be updated by UMAP
                        UMAP_2=0.0,  # Placeholder, will be updated by UMAP
                        LEIDEN_1=0.0,  # Placeholder, will be updated by clustering
                        PCA_1=float(pca_coords[0]),
                        PCA_2=float(pca_coords[1]) if len(pca_coords) > 1 else 0.0
                    )
                    features_list.append(features)
            
            # Batch upsert to DuckDB
            success = self.db_helper.upsert_engineered_features_batch(features_list)
            self.db_helper.close()
            if success:
                logger.info(f"✅ Successfully saved PCA features for {len(features_list)} datasets")
                return True
            else:
                logger.error("❌ Failed to save PCA features to DuckDB")
                return False
                
        except Exception as e:
            logger.error(f"Error saving PCA features: {str(e)}")
            return False
    
    #########################################################
    # Feature UMAP + PCA
    #########################################################

    @timer_wrap  
    def run_feature_umap(
        self,
        embeddings: Dict[str, np.ndarray],
        n_neighbors: int = None,
        min_dist: float = None,
        n_components: int = None,
        random_seed: int = None
    ) -> pd.DataFrame:
        """
        Apply UMAP dimensionality reduction with reproducibility:
        - Set random_state for deterministic embedding layout
        - Save UMAP_1, UMAP_2 features to DuckDB via upsert_engineered_features_batch
        """
        import umap
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
            embedding_matrix = embedding_matrix.T
            
            logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
            
            # Initialize and fit UMAP
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_seed,
                verbose=True
            )
            
            # Fit and transform embeddings
            umap_embeddings = reducer.fit_transform(embedding_matrix)
            logger.info(f"UMAP embeddings shape: {umap_embeddings.shape}")

            # turn into pandas dataframe
            umap_results = pd.DataFrame(umap_embeddings)
            
            logger.info("✅ UMAP reduction completed successfully!")
            return umap_results
            
        except Exception as e:
            logger.error(f"UMAP reduction failed: {str(e)}")
            return pd.DataFrame()

    @timer_wrap  
    def run_feature_pca(
        self,
        embeddings: Dict[str, np.ndarray],
        n_components: int = None,
        random_seed: int = None
    ) -> pd.DataFrame:
        """
        Apply PCA dimensionality reduction with reproducibility:
        - Set random_state for deterministic PCA components
        - Returns DataFrame without saving to DuckDB (for retrieval helpers)
        """
        from sklearn.decomposition import PCA
            
        # Use instance defaults if not provided
        n_components = n_components or self.n_components
        random_seed = random_seed or self.random_seed
        
        logger.info(f"🔄 Running PCA reduction on {len(embeddings)} embeddings...")
        logger.info(f"Parameters: n_components={n_components}, random_state={random_seed}")
        
        try:
            # Prepare embeddings matrix
            dataset_ids = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[dataset_id] for dataset_id in dataset_ids])
            embedding_matrix = embedding_matrix.T
            
            logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
            
            # Initialize and fit PCA
            pca = PCA(
                n_components=n_components,
                random_state=random_seed
            )
            
            # Fit and transform embeddings
            pca_embeddings = pca.fit_transform(embedding_matrix)
            logger.info(f"PCA embeddings shape: {pca_embeddings.shape}")
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")

            # turn into pandas dataframe
            pca_results = pd.DataFrame(pca_embeddings)
            
            logger.info("✅ PCA reduction completed successfully!")
            return pca_results
            
        except Exception as e:
            logger.error(f"PCA reduction failed: {str(e)}")
            return pd.DataFrame()
        
    def feature_decomposition(self, embeddings: Dict[str, np.ndarray], output_path: str = "feature_decomposition.pkl") -> pd.DataFrame:
        """Run feature UMAP + PCA, combine results, and save to pickle."""
        import pickle
        
        umap_df = self.run_feature_umap(embeddings)
        pca_df = self.run_feature_pca(embeddings)
        
        combined_df = pd.concat([umap_df, pca_df], axis=1)
        
        with open(output_path, 'wb') as f:
            pickle.dump(combined_df, f)
        
        logger.info(f"✅ Feature decomposition saved to {output_path}")
        return combined_df
    
    def load_feature_decomposition(self, output_path: str = "feature_decomposition.pkl") -> pd.DataFrame:
        """Load feature decomposition from pickle."""
        import pickle
        
        with open(output_path, 'rb') as f:
            return pickle.load(f)


    
