"""
Kaggle dimensionality reduction utilities mirroring src/dimensionality_reduction.py

- Loads datasets from DuckDB via KaggleDuckDBHelper
- Runs per-cluster PCA on feature clusters (LEIDEN_* features)
- Writes engineered features (EAV) back to DuckDB
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
import sys, os

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None  # type: ignore
try:
    from src.kaggle.models import EngineeredFeatures
    from src.kaggle.duckdb_utils import get_duckdb_helper
    from src.kaggle.helpers import timer_wrap, initialize_logging
except Exception:
    from .models import EngineeredFeatures
    from .duckdb_utils import get_duckdb_helper
    from .helpers import timer_wrap, initialize_logging

logger = initialize_logging()

base_tmp = "/kaggle/temp/"

artifacts = os.path.join(base_tmp, "artifacts")
dataset_embeddings_path = os.path.join(base_tmp, "dataset_embeddings.parquet")
DEFAULT_DUCKDB = os.path.join(artifacts, "mdc_challenge.db")

DEFAULT_EMB_PATH = dataset_embeddings_path

@timer_wrap
def _run_pca_on_cluster(
    cluster_id: str,
    feature_idx: List[int],
    dataset_embeddings: np.ndarray,
    random_seed: int,
) -> Tuple[str, np.ndarray]:
    """Compute 1D PCA component for a given cluster column subset.

    Parameters:
    - cluster_id: Label of the feature cluster.
    - feature_idx: Column indices in `dataset_embeddings` belonging to this cluster.
    - dataset_embeddings: Full embeddings matrix, shape (n_samples, n_features).
    - random_seed: Random seed for deterministic PCA results.

    Returns:
    - Tuple of (cluster_id, pc1_vector) where pc1_vector has shape (n_samples,).
    """
    if PCA is None:
        logger.error("scikit-learn is required for PCA but not available")
        raise RuntimeError("scikit-learn is required for PCA")
    if len(feature_idx) < 2:
        # fallback to the single column values
        logger.debug(f"Cluster {cluster_id}: <2 features, skipping PCA and returning raw column")
        try:
            return cluster_id, dataset_embeddings[:, feature_idx[0]].copy()
        except Exception as e:
            logger.error(f"Failed extracting single column for cluster {cluster_id}", exc_info=e)
            raise

    try:
        X_sub = dataset_embeddings[:, feature_idx]
        logger.debug(f"Cluster {cluster_id}: running PCA on shape {X_sub.shape}")
        pca = PCA(n_components=1, random_state=random_seed, svd_solver="full")
        pc1 = pca.fit_transform(X_sub).ravel()
        return cluster_id, pc1
    except Exception as e:
        logger.error(f"PCA failed for cluster {cluster_id}", exc_info=e)
        raise

@timer_wrap
def run_per_cluster_pca(
    dataset_embeddings: np.ndarray,
    feature_cluster_map: Dict[str, str],  # {feature_name: cluster_label}
    dataset_ids: List[str],
    db_path: str = DEFAULT_DUCKDB,
    random_seed: int = 42,
) -> bool:
    """Run 1D PCA per feature cluster and persist LEIDEN_* features to DuckDB.

    Parameters:
    - dataset_embeddings: Matrix (n_samples, n_features) of dataset embeddings.
    - feature_cluster_map: Mapping from feature name to cluster label.
    - dataset_ids: Dataset identifiers aligned with rows in `dataset_embeddings`.
    - db_path: Path to DuckDB database file.
    - random_seed: Seed for deterministic PCA.

    Returns True on success (even if there are no clusters to process). Raises on critical errors.
    """
    logger.info(
        f"run_per_cluster_pca: n_samples={dataset_embeddings.shape[0] if dataset_embeddings is not None else 'NA'}, "
        f"n_features={dataset_embeddings.shape[1] if dataset_embeddings is not None and dataset_embeddings.ndim==2 else 'NA'}, "
        f"n_feature_names={len(feature_cluster_map)}"
    )
    if dataset_embeddings is None or dataset_embeddings.ndim != 2:
        logger.error("Invalid dataset_embeddings: expected 2D numpy array")
        raise ValueError("dataset_embeddings must be a 2D numpy array")
    if len(dataset_ids) != dataset_embeddings.shape[0]:
        logger.error("Row count of dataset_embeddings does not match length of dataset_ids")
        raise ValueError("dataset_ids length must match number of rows in dataset_embeddings")
    # Group features by cluster
    name_to_idx = {fname: i for i, fname in enumerate(feature_cluster_map)}
    cluster2cols: Dict[str, List[int]] = defaultdict(list)
    for fname, cid in feature_cluster_map.items():
        cluster2cols[cid].append(name_to_idx[fname])

    # Parallel PCA
    comp_matrix: List[Tuple[str, np.ndarray]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(cluster2cols))) as ex:
        futures = {
            ex.submit(_run_pca_on_cluster, cid, cols, dataset_embeddings, random_seed): cid
            for cid, cols in cluster2cols.items()
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                cluster_id, pc1_result = fut.result()
                comp_matrix.append((cluster_id, pc1_result))
            except Exception as e:
                logger.warning(f"Skipping cluster {cid} due to PCA error", exc_info=e)

    if not comp_matrix:
        logger.info("No PCA components produced; nothing to persist.")
        return True

    # Assemble reduced matrix [n_samples, n_clusters]
    comp_matrix.sort()
    labels, vectors = zip(*comp_matrix)
    X_reduced = np.column_stack(vectors)
    logger.info(f"Assembled reduced matrix with shape {X_reduced.shape} across {len(labels)} clusters")

    # Persist into DuckDB as engineered features (LEIDEN_<cluster_label>)
    db = get_duckdb_helper(db_path)
    try:
        # Build dataset_id → document_id mapping
        try:
            datasets = db.get_all_datasets()
            dataset_to_doc = {ds.dataset_id: ds.document_id for ds in datasets}
        except Exception as e:
            logger.error("Failed to load datasets from DuckDB", exc_info=e)
            raise

        features_to_write: List[EngineeredFeatures] = []
        for row_idx, ds_id in enumerate(dataset_ids):
            if ds_id not in dataset_to_doc:
                logger.debug(f"Dataset id {ds_id} not found in DuckDB; skipping feature write")
                continue
            feature_dict = {
                "dataset_id": ds_id,
                "document_id": dataset_to_doc[ds_id],
                "UMAP_1": 0.0,
                "UMAP_2": 0.0,
            }
            for col_idx, cid in enumerate(labels):
                feature_dict[f"LEIDEN_{cid}"] = float(X_reduced[row_idx, col_idx])
            features_to_write.append(EngineeredFeatures(**feature_dict))

        if features_to_write:
            logger.info(f"Writing {len(features_to_write)} EngineeredFeatures rows to DuckDB")
            for f in features_to_write:
                try:
                    db.insert_engineered_features(f)
                except Exception as e:
                    logger.warning(f"Failed to insert EngineeredFeatures for dataset_id={f.dataset_id}", exc_info=e)
        return True
    finally:
        db.close()


__all__ = [
    "run_per_cluster_pca",
]


