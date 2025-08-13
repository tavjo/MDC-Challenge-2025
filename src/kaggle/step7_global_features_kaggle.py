"""
Global engineered features for Kaggle (Step 8).

What this adds (without redoing per-cluster PCA):
- Global PCA (PCA_1, PCA_2) computed over dataset embeddings matrix
- Optional global UMAP (UMAP_1, UMAP_2)
- Optional neighborhood statistics from local kNN on embeddings

All reports are saved under /kaggle/tmp/reports/global_features/.
Engineered features are stored in DuckDB via KaggleDuckDBHelper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd

from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

try:
    from sklearn.decomposition import PCA  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    PCA = None  # type: ignore
    SKLEARN_AVAILABLE = False

try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except Exception:  # pragma: no cover
    umap = None  # type: ignore
    UMAP_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors  # type: ignore

from src.kaggle.duckdb import get_duckdb_helper
from src.kaggle.models import EngineeredFeatures


REPORT_DIR = Path("/kaggle/tmp/reports/global_features")
DEFAULT_DB_PATH = "/kaggle/tmp/mdc.duckdb"
DEFAULT_EMB_PATH = "/kaggle/tmp/dataset_embeddings.parquet"


def _ensure_report_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR


def load_embeddings_parquet(path: str = DEFAULT_EMB_PATH) -> Tuple[List[str], np.ndarray]:
    df = pd.read_parquet(path)
    # first column should be dataset_id
    if df.columns[0] != "dataset_id":
        raise ValueError("First column of embeddings parquet must be 'dataset_id'")
    dataset_ids = df["dataset_id"].astype(str).tolist()
    X = df.drop(columns=["dataset_id"]).to_numpy(dtype=float, copy=False)
    return dataset_ids, X


def run_global_pca(
    dataset_ids: List[str],
    X: np.ndarray,
    db_path: str = DEFAULT_DB_PATH,
    n_components: int = 2,
    random_seed: int = 42,
) -> Dict[str, List[float]]:
    if not SKLEARN_AVAILABLE:
        return {}
    if X.shape[0] == 0:
        return {}
    pca = PCA(n_components=n_components, random_state=random_seed, svd_solver="full")
    Z = pca.fit_transform(X)  # shape (N, n_components)
    # Persist to DuckDB as PCA_1, PCA_2
    db = get_duckdb_helper(db_path)
    try:
        # Map dataset_id -> document_id
        datasets = db.get_all_datasets()
        ds2doc = {ds.dataset_id: ds.document_id for ds in datasets}
        for idx, ds_id in enumerate(dataset_ids):
            if ds_id not in ds2doc:
                continue
            coords = Z[idx]
            feature = EngineeredFeatures(
                dataset_id=ds_id,
                document_id=ds2doc[ds_id],
                UMAP_1=0.0,
                UMAP_2=0.0,
                LEIDEN_1=0.0,
            )
            # attach PCA_* keys by using model's extra allowance via model_dump/update
            feature_dict = feature.model_dump()
            feature_dict.update({
                "PCA_1": float(coords[0]),
                "PCA_2": float(coords[1]) if n_components > 1 and len(coords) > 1 else 0.0,
            })
            db.insert_engineered_features(EngineeredFeatures(**feature_dict))
    finally:
        db.close()

    # Save PCA report
    report = {
        "n_datasets": int(X.shape[0]),
        "n_dims": int(X.shape[1]),
        "n_components": n_components,
        "explained_variance_ratio": [float(v) for v in getattr(pca, "explained_variance_ratio_", [])],
    }
    _ensure_report_dir()
    with open(REPORT_DIR / "pca_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return {ds_id: [float(v) for v in Z[i][:2]] for i, ds_id in enumerate(dataset_ids)}


def run_global_umap(
    dataset_ids: List[str],
    X: np.ndarray,
    db_path: str = DEFAULT_DB_PATH,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_seed: int = 42,
) -> Dict[str, List[float]]:
    if not UMAP_AVAILABLE:
        return {}
    if X.shape[0] == 0:
        return {}
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_seed,
        verbose=False,
    )
    Z = reducer.fit_transform(X)
    db = get_duckdb_helper(db_path)
    try:
        datasets = db.get_all_datasets()
        ds2doc = {ds.dataset_id: ds.document_id for ds in datasets}
        for idx, ds_id in enumerate(dataset_ids):
            if ds_id not in ds2doc:
                continue
            coords = Z[idx]
            feature = EngineeredFeatures(
                dataset_id=ds_id,
                document_id=ds2doc[ds_id],
                UMAP_1=float(coords[0]),
                UMAP_2=float(coords[1]) if n_components > 1 and len(coords) > 1 else 0.0,
                LEIDEN_1=0.0,
            )
            db.insert_engineered_features(feature)
    finally:
        db.close()

    report = {
        "n_datasets": int(X.shape[0]),
        "params": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": n_components,
            "random_seed": random_seed,
        },
    }
    _ensure_report_dir()
    with open(REPORT_DIR / "umap_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return {ds_id: [float(v) for v in Z[i][:2]] for i, ds_id in enumerate(dataset_ids)}


def run_neighborhood_stats(
    dataset_ids: List[str],
    X: np.ndarray,
    db_path: str = DEFAULT_DB_PATH,
    k: int = 5,
) -> Dict[str, Dict[str, float]]:
    if X.shape[0] == 0:
        return {}
    # kNN with cosine distance
    nbrs = NearestNeighbors(n_neighbors=min(k, X.shape[0]), metric="cosine", algorithm="brute")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    stats_per_ds: Dict[str, Dict[str, float]] = {}
    for i, ds_id in enumerate(dataset_ids):
        d = distances[i]
        idxs = indices[i]
        # exclude self if present at index 0
        mask = idxs != i
        d = d[mask]
        neigh_idxs = idxs[mask]
        if d.size == 0:
            continue
        sims = 1.0 - d
        neighbor_norms = np.linalg.norm(X[neigh_idxs], axis=1)
        stats = {
            "neighbor_similarity_mean": float(np.mean(sims)),
            "neighbor_similarity_max": float(np.max(sims)),
            "neighbor_similarity_var": float(np.var(sims)),
            "neighbor_norm_mean": float(np.mean(neighbor_norms)),
            "neighbor_norm_max": float(np.max(neighbor_norms)),
            "neighbor_norm_var": float(np.var(neighbor_norms)),
        }
        stats_per_ds[ds_id] = stats

    # persist to EAV
    db = get_duckdb_helper(db_path)
    try:
        datasets = db.get_all_datasets()
        ds2doc = {ds.dataset_id: ds.document_id for ds in datasets}
        for ds_id, s in stats_per_ds.items():
            if ds_id not in ds2doc:
                continue
            feature = EngineeredFeatures(dataset_id=ds_id, document_id=ds2doc[ds_id], UMAP_1=0.0, UMAP_2=0.0, LEIDEN_1=0.0)
            feature_dict = feature.model_dump()
            feature_dict.update(s)
            db.insert_engineered_features(EngineeredFeatures(**feature_dict))
    finally:
        db.close()

    # report
    report = {
        "n_datasets": len(stats_per_ds),
        "avg_neighbor_similarity_mean": float(np.mean([v["neighbor_similarity_mean"] for v in stats_per_ds.values()])) if stats_per_ds else 0.0,
        "k": k,
    }
    _ensure_report_dir()
    with open(REPORT_DIR / "neighborhood_stats_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return stats_per_ds


def run_step8_global_features(
    emb_path: str = DEFAULT_EMB_PATH,
    db_path: str = DEFAULT_DB_PATH,
    do_pca: bool = True,
    do_umap: bool = False,
    do_neighbors: bool = True,
) -> Dict[str, bool]:
    dataset_ids, X = load_embeddings_parquet(emb_path)
    results = {"pca": False, "umap": False, "neighbors": False}

    if do_pca and SKLEARN_AVAILABLE:
        run_global_pca(dataset_ids, X, db_path=db_path)
        results["pca"] = True

    if do_umap and UMAP_AVAILABLE:
        run_global_umap(dataset_ids, X, db_path=db_path)
        results["umap"] = True

    if do_neighbors:
        run_neighborhood_stats(dataset_ids, X, db_path=db_path, k=5)
        results["neighbors"] = True

    return results


__all__ = [
    "load_embeddings_parquet",
    "run_global_pca",
    "run_global_umap",
    "run_neighborhood_stats",
    "run_step8_global_features",
]


