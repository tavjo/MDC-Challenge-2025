#!/usr/bin/env python3
""" Train Random Forest classifier for MDC Challenge

This script implements the *exact* checklist described in the training guide.

Usage
-----
python scripts/phase_09_train_rf.py \
    --input_csv data/train_data.csv \
    --output_dir artifacts/models

Arguments
---------
--input_csv      Path to CSV containing the feature table **including** the target column.
--duckdb_path    Optional path to a DuckDB database.  If supplied, model metadata will
                 be added to the `models` table.
--target_col     Name of the target column in the CSV (default: `label`).
--group_col      Optional column that identifies groups (e.g., `article_id`) for
                 StratifiedGroupKFold.  If absent, normal StratifiedKFold is used.
--seed           Random seed (default: 42).
--n_iter         Number of parameter samples for RandomizedSearch (default: 100).
--use_balanced_rf  If set, use ``imblearn.ensemble.BalancedRandomForestClassifier``
                 instead of the vanilla RandomForestClassifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Optional Balanced RF
try:
    from imblearn.ensemble import BalancedRandomForestClassifier  # type: ignore
except ImportError:  # pragma: no cover
    BalancedRandomForestClassifier = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Random‑Forest (Phase 09) according to MDC checklist."
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to input CSV containing features + label.")
    parser.add_argument("--duckdb_path", type=str, default=None,
                        help="Optional DuckDB file; if provided, model metadata is stored there.")
    parser.add_argument("--output_dir", type=str, default="artifacts/models",
                        help="Directory to write all artefacts into.")
    parser.add_argument("--target_col", type=str, default="label",
                        help="Name of the target column in CSV (default: label).")
    parser.add_argument("--group_col", type=str, default=None,
                        help="Optional grouping column for StratifiedGroupKFold.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--n_iter", type=int, default=100,
                        help="RandomizedSearch iterations (default: 100).")
    parser.add_argument("--use_balanced_rf", action="store_true",
                        help="If set, use BalancedRandomForestClassifier from imblearn.")
    return parser.parse_args()


def make_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
    )
    logger = logging.getLogger("phase_09_train_rf")
    return logger


def load_data(csv_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Sanity checks
    if X.shape != (487, 55):
        logging.warning(
            "Data shape is %s (expected 487, 55). Continuing anyway.", X.shape
        )
    numeric_cols = X.select_dtypes(include="number").columns
    if len(numeric_cols) != X.shape[1]:
        raise ValueError("Non‑numeric feature columns detected; aborting.")
    if X.isna().any().any():
        logging.info("Missing values detected – applying median imputation.")
    return X, y


def build_cv(y: pd.Series, groups: Optional[pd.Series], seed: int):
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cv


def build_pipeline(seed: int, use_balanced: bool):
    # Imputer
    imputer = ("imputer", SimpleImputer(strategy="median"))
    if use_balanced:
        if BalancedRandomForestClassifier is None:
            raise ImportError(
                "imblearn is not installed. Re‑install with `pip install imbalanced-learn`.")

        clf = BalancedRandomForestClassifier(
            n_jobs=-1, random_state=seed, bootstrap=True
        )
    else:
        clf = RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=seed,
            bootstrap=True,
        )
    pipeline = Pipeline(steps=[imputer, ("rf", clf)])
    return pipeline


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    logger = make_logger(out_dir)

    logger.info("✅ Phase 09 training started")
    logger.info("Input CSV: %s", args.input_csv)

    X, y = load_data(Path(args.input_csv), args.target_col)

    # Groups?
    groups = None
    if args.group_col and args.group_col in X.columns:
        groups = X[args.group_col]
        X = X.drop(columns=[args.group_col])
        logger.info("Using '%s' column for grouping in CV.", args.group_col)
    elif args.group_col:
        logger.warning("Group column '%s' not found – defaulting to StratifiedKFold.", args.group_col)

    # Class imbalance log
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=np.unique(y),
                                         y=y.values)
    logger.info("Class distribution: %s", y.value_counts(normalize=True).to_dict())
    logger.info("Computed balanced class weights: %s", dict(zip(np.unique(y), class_weights)))

    cv = build_cv(y, groups, args.seed)
    pipe = build_pipeline(args.seed, args.use_balanced_rf)

    # Hyper‑parameter space
    param_dist = {
        "rf__n_estimators": ss.randint(200, 1001),
        "rf__max_depth": [None, 10, 20, 30],
        "rf__max_features": ["sqrt", "log2", 0.7],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_samples": [0.8],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=args.seed,
        refit=True,
    )

    search.fit(X, y, groups=groups)
    logger.info("Best parameters: %s", search.best_params_)
    logger.info("Best CV F1: %.4f", search.best_score_)

    # Cross‑validation metrics for baseline reproducibility
    cv_results = cross_validate(
        search.best_estimator_, X, y,
        cv=cv,
        scoring=["f1", "precision", "recall", "accuracy"],
        n_jobs=-1,
        return_estimator=False
    )
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    logger.info("Saved cv_results.csv")

    # Permutation importance on entire data (or could be on hold‑out set)
    logger.info("Computing permutation importances (30 repeats)…")
    perm = permutation_importance(
        search.best_estimator_, X, y,
        n_repeats=30,
        random_state=args.seed,
        n_jobs=-1,
        scoring="f1"
    )
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importances.head(20).to_csv(out_dir / "feature_importance_permutation.csv", index=False)
    logger.info("Saved feature_importance_permutation.csv (top 20 features).")

    # Save model + feature names + params
    model_path = out_dir / "rf_model.pkl"
    joblib.dump(search.best_estimator_, model_path, compress=3)
    json.dump(search.best_params_, open(out_dir / "best_params.json", "w"), indent=2)
    json.dump(list(X.columns), open(out_dir / "feature_names.json", "w"), indent=2)
    logger.info("Model and artefacts persisted to %s", out_dir)

    # Optional: store metadata to DuckDB
    if args.duckdb_path:
        try:
            import duckdb  # Deferred import
            con = duckdb.connect(args.duckdb_path)
            con.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_hash VARCHAR PRIMARY KEY,
                trained_at TIMESTAMP,
                best_f1 DOUBLE,
                path VARCHAR
            );
            """)
            sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
            con.execute(
                "INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?)",
                (sha256, datetime.utcnow(), float(search.best_score_), str(model_path))
            )
            con.close()
            logger.info("Model metadata registered in DuckDB (%s).", args.duckdb_path)
        except Exception as exc:  # pragma: no cover
            logger.exception("DuckDB metadata storage failed: %s", exc)

    logger.info("🏁 Training complete. Artefacts available in %s", out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
