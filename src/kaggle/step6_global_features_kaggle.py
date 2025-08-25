"""
Global engineered features for Kaggle (Step 8).

What this adds (without redoing per-cluster PCA):
- Global PCA (PCA_1, PCA_2) computed over dataset embeddings matrix
- Optional global UMAP (UMAP_1, UMAP_2)
- Optional neighborhood statistics from local kNN on embeddings

All reports are saved under /kaggle/temp/reports/global_features/.
Engineered features are stored in DuckDB via KaggleDuckDBHelper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import numpy as np

from pathlib import Path
import sys, os

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
try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:
    NearestNeighbors = None
    NEIGHBORS_AVAILABLE = False

# try:
#     from src.kaggle.duckdb_utils import get_duckdb_helper
#     from src.kaggle.models import EngineeredFeatures
#     from src.kaggle.helpers import timer_wrap, initialize_logging
#     from src.kaggle.step4_construct_datasets_kaggle import load_embeddings_parquet
# except Exception:
from sklearn.neighbors import NearestNeighbors  # type: ignore
from duckdb_utils import get_duckdb_helper
from models import EngineeredFeatures
from helpers import timer_wrap, initialize_logging
from step4_construct_datasets_kaggle import load_embeddings_parquet

logger = initialize_logging()

base_tmp = "/kaggle/temp/"

REPORT_DIR = Path(os.path.join(base_tmp, "reports/global_features"))

artifacts = os.path.join(base_tmp, "artifacts")
dataset_embeddings_path = os.path.join(base_tmp, "dataset_embeddings.parquet")
DEFAULT_DUCKDB = os.path.join(artifacts, "mdc_challenge.db")

DEFAULT_EMB_PATH = dataset_embeddings_path


def _ensure_report_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR


@timer_wrap
def run_global_pca(
    dataset_ids: List[str],
    X: np.ndarray,
    db_path: str = DEFAULT_DUCKDB,
    n_components: int = 2,
    random_seed: int = 42,
) -> Dict[str, List[float]]:
    """Compute global PCA on dataset embeddings and persist PCA_1/PCA_2 to DuckDB.

    Parameters:
    - dataset_ids: List of dataset IDs aligned with rows in X
    - X: Embeddings matrix of shape (n_samples, n_dims)
    - db_path: DuckDB path to persist engineered features
    - n_components: Number of PCA components to compute (stored up to 2)
    - random_seed: Deterministic seed for PCA
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available; skipping global PCA")
        return {}
    if X.shape[0] == 0:
        logger.info("Empty embeddings matrix; skipping global PCA")
        return {}
    try:
        pca = PCA(n_components=n_components, random_state=random_seed, svd_solver="full")
        Z = pca.fit_transform(X)  # shape (N, n_components)
        logger.info(f"Global PCA computed: X={X.shape} -> Z={Z.shape}")
    except Exception as e:
        logger.error("Global PCA failed", exc_info=e)
        return {}
    # Persist to DuckDB as PCA_1, PCA_2
    db = get_duckdb_helper(db_path)
    try:
        # Map dataset_id -> document_id
        try:
            datasets = db.get_all_datasets()
            ds2doc = {ds.dataset_id: ds.document_id for ds in datasets}
        except Exception as e:
            logger.error("Failed to load datasets for PCA persistence", exc_info=e)
            return {}
        for idx, ds_id in enumerate(dataset_ids):
            try:
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
            except Exception as e:
                logger.warning(f"Failed to insert PCA features for dataset_id={ds_id}", exc_info=e)
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


@timer_wrap
def run_global_umap(
    dataset_ids: List[str],
    X: np.ndarray,
    db_path: str = DEFAULT_DUCKDB,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_seed: int = 42,
) -> Dict[str, List[float]]:
    """Compute global UMAP on dataset embeddings and persist UMAP_1/UMAP_2 to DuckDB.

    Parameters mirror run_global_pca; returns dict of dataset_id -> [UMAP_1, UMAP_2].
    """
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available; skipping global UMAP")
        return {}
    if X.shape[0] == 0:
        logger.info("Empty embeddings matrix; skipping global UMAP")
        return {}
    try:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_seed,
            verbose=False,
        )
        Z = reducer.fit_transform(X)
        logger.info(f"Global UMAP computed: X={X.shape} -> Z={Z.shape}")
    except Exception as e:
        logger.error("Global UMAP failed", exc_info=e)
        return {}
    db = get_duckdb_helper(db_path)
    try:
        try:
            datasets = db.get_all_datasets()
            ds2doc = {ds.dataset_id: ds.document_id for ds in datasets}
        except Exception as e:
            logger.error("Failed to load datasets for UMAP persistence", exc_info=e)
            return {}
        for idx, ds_id in enumerate(dataset_ids):
            try:
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
            except Exception as e:
                logger.warning(f"Failed to insert UMAP features for dataset_id={ds_id}", exc_info=e)
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


@timer_wrap
def run_neighborhood_stats(
    dataset_ids: List[str],
    X: np.ndarray,
    db_path: str = DEFAULT_DUCKDB,
    k: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Compute simple neighborhood statistics via cosine kNN and persist to DuckDB.

    Returns dict of dataset_id -> stats summary used for reports.
    """
    if X.shape[0] == 0:
        logger.info("Empty embeddings matrix; skipping neighborhood stats")
        return {}
    # kNN with cosine distance
    try:
        nbrs = NearestNeighbors(n_neighbors=min(k, X.shape[0]), metric="cosine", algorithm="brute")
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
    except Exception as e:
        logger.error("kNN computation failed", exc_info=e)
        return {}

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
        try:
            sims = 1.0 - d
            neighbor_norms = np.linalg.norm(X[neigh_idxs], axis=1)
        except Exception as e:
            logger.warning(f"Failed to compute stats for dataset_id={ds_id}", exc_info=e)
            continue
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
        try:
            datasets = db.get_all_datasets()
            ds2doc = {ds.dataset_id: ds.document_id for ds in datasets}
        except Exception as e:
            logger.error("Failed to load datasets for neighborhood stats persistence", exc_info=e)
            return {}
        for ds_id, s in stats_per_ds.items():
            try:
                if ds_id not in ds2doc:
                    continue
                feature = EngineeredFeatures(dataset_id=ds_id, document_id=ds2doc[ds_id], UMAP_1=0.0, UMAP_2=0.0, LEIDEN_1=0.0)
                feature_dict = feature.model_dump()
                feature_dict.update(s)
                db.insert_engineered_features(EngineeredFeatures(**feature_dict))
            except Exception as e:
                logger.warning(f"Failed to insert neighborhood stats for dataset_id={ds_id}", exc_info=e)
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


@timer_wrap
def run_step8_global_features(
    emb_path: str = DEFAULT_EMB_PATH,
    db_path: str = DEFAULT_DUCKDB,
    do_pca: bool = True,
    do_umap: bool = True,
    do_neighbors: bool = False,
) -> Dict[str, bool]:
    """Run global feature computations (PCA/UMAP/kNN stats) and persist results.

    Returns a dict indicating which computations ran successfully.
    """
    dataset_ids, X = load_embeddings_parquet(emb_path)
    results = {"pca": False, "umap": False, "neighbors": False}

    if do_pca and SKLEARN_AVAILABLE:
        try:
            run_global_pca(dataset_ids, X, db_path=db_path)
            results["pca"] = True
        except Exception as e:
            logger.warning("Global PCA step failed", exc_info=e)

    if do_umap and UMAP_AVAILABLE:
        try:
            run_global_umap(dataset_ids, X, db_path=db_path)
            results["umap"] = True
        except Exception as e:
            logger.warning("Global UMAP step failed", exc_info=e)

    if do_neighbors:
        try:
            run_neighborhood_stats(dataset_ids, X, db_path=db_path, k=5)
            results["neighbors"] = True
        except Exception as e:
            logger.warning("Neighborhood stats step failed", exc_info=e)

    return results


__all__ = [
    "load_embeddings_parquet",
    "run_global_pca",
    "run_global_umap",
    "run_neighborhood_stats",
    "run_step8_global_features",
]


