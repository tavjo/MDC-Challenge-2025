"""
Kaggle dimensionality reduction utilities mirroring src/dimensionality_reduction.py

- Loads datasets from DuckDB via KaggleDuckDBHelper
- Runs per-cluster PCA on feature clusters (LEIDEN_* features)
- Writes engineered features (EAV) back to DuckDB
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None  # type: ignore

from src.kaggle.models import Dataset, EngineeredFeatures
from src.kaggle.duckdb import get_duckdb_helper


def _run_pca_on_cluster(
    cluster_id: str,
    feature_idx: List[int],
    dataset_embeddings: np.ndarray,
    random_seed: int,
) -> Tuple[str, np.ndarray]:
    if PCA is None:
        raise RuntimeError("scikit-learn is required for PCA")
    if len(feature_idx) < 2:
        # fallback to the single column values
        return cluster_id, dataset_embeddings[:, feature_idx[0]].copy()

    X_sub = dataset_embeddings[:, feature_idx]
    pca = PCA(n_components=1, random_state=random_seed, svd_solver="full")
    pc1 = pca.fit_transform(X_sub).ravel()
    return cluster_id, pc1


def run_per_cluster_pca(
    dataset_embeddings: np.ndarray,
    feature_cluster_map: Dict[str, str],  # {feature_name: cluster_label}
    dataset_ids: List[str],
    db_path: str = "/kaggle/tmp/mdc.duckdb",
    random_seed: int = 42,
) -> bool:
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
            cluster_id, pc1_result = fut.result()
            comp_matrix.append((cluster_id, pc1_result))

    if not comp_matrix:
        return True

    # Assemble reduced matrix [n_samples, n_clusters]
    comp_matrix.sort()
    labels, vectors = zip(*comp_matrix)
    X_reduced = np.column_stack(vectors)

    # Persist into DuckDB as engineered features (LEIDEN_<cluster_label>)
    db = get_duckdb_helper(db_path)
    try:
        # Build dataset_id → document_id mapping
        datasets = db.get_all_datasets()
        dataset_to_doc = {ds.dataset_id: ds.document_id for ds in datasets}

        features_to_write: List[EngineeredFeatures] = []
        for row_idx, ds_id in enumerate(dataset_ids):
            if ds_id not in dataset_to_doc:
                continue
            feature_dict = {
                "dataset_id": ds_id,
                "document_id": dataset_to_doc[ds_id],
                "UMAP_1": 0.0,
                "UMAP_2": 0.0,
                "LEIDEN_1": 0.0,
            }
            for col_idx, cid in enumerate(labels):
                feature_dict[f"LEIDEN_{cid}"] = float(X_reduced[row_idx, col_idx])
            features_to_write.append(EngineeredFeatures(**feature_dict))

        if features_to_write:
            db.insert_engineered_features(features=features_to_write[0]) if len(features_to_write) == 1 else [db.insert_engineered_features(f) for f in features_to_write]
        return True
    finally:
        db.close()


__all__ = [
    "run_per_cluster_pca",
]


