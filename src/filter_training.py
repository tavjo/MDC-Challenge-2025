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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
import skops.io as sio
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, cross_validate,
                                     train_test_split, GroupShuffleSplit)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Optional Balanced RF
try:
    from imblearn.ensemble import BalancedRandomForestClassifier  # type: ignore
except ImportError:  # pragma: no cover
    BalancedRandomForestClassifier = None  # type: ignore

# Optional LightGBM / XGBoost
try:
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Random Forest training script.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - input_csv: Path to input CSV file with features and target
        - duckdb_path: Optional path to DuckDB database for metadata storage
        - output_dir: Directory for saving model artifacts
        - target_col: Name of the target column (default: 'target')
        - group_col: Optional grouping column for stratified group k-fold
        - seed: Random seed for reproducibility (default: 42)
        - n_iter: Number of RandomizedSearch iterations (default: 100)
        - use_balanced_rf: Whether to use BalancedRandomForestClassifier
        
    Raises
    ------
    SystemExit
        If required arguments are missing or invalid
    """
    try:
        parser = argparse.ArgumentParser(
            description="Train Classifier (Phase 09) according to MDC checklist."
        )
        parser.add_argument("--input_csv", type=str, required=True,
                            help="Path to input CSV containing features + label.")
        parser.add_argument("--duckdb_path", type=str, default="artifacts/mdc-challenge.db",
                            help="Optional DuckDB file; if provided, model metadata is stored there.")
        parser.add_argument("--output_dir", type=str, default="artifacts/models",
                            help="Directory to write all artefacts into.")
        parser.add_argument("--target_col", type=str, default="target",
                            help="Name of the target column in CSV (default: target).")
        parser.add_argument("--group_col", default="document_id",
                    help="Grouping column for CV + hold‑out split (default: document_id).")

        parser.add_argument("--val_frac", type=float, default=0.20,
                    help="Fraction of data to reserve as frozen validation set (0 = no hold‑out).")
        parser.add_argument("--no_holdout", action="store_true",
                    help="Disable hold‑out even if --val_frac > 0.")
        parser.add_argument("--seed", type=int, default=42,
                            help="Random seed.")
        parser.add_argument("--n_iter", type=int, default=100,
                            help="RandomizedSearch iterations (default: 100).")
        parser.add_argument("--use_balanced_rf", action="store_true",
                            help="If set, use BalancedRandomForestClassifier from imblearn.")
        parser.add_argument("--model", choices=["lightgbm", "xgboost", "rf"], default="lightgbm",
                            help="Model to train (default: lightgbm; rf is fallback baseline).")
        parser.add_argument("--calibrate", choices=["none", "platt", "isotonic"], default="platt",
                            help="Probability calibration method (default: platt).")
        parser.add_argument("--doc_recall_target", type=float, default=0.95,
                            help="Target document-level recall for doc gate threshold selection (default: 0.95).")
        parser.add_argument("--save_val_probs", action="store_true",
                            help="Save validation probabilities for offline threshold tuning.")
        return parser.parse_args()
    except Exception as e:
        logging.error(f"Failed to parse command-line arguments: {e}")
        sys.exit(1)


def make_logger(out_dir: Path) -> logging.Logger:
    """Create and configure a logger for the training process.
    
    Parameters
    ----------
    out_dir : Path
        Output directory where log file will be created
        
    Returns
    -------
    logging.Logger
        Configured logger instance that writes to both stdout and log file
        
    Raises
    ------
    OSError
        If unable to create output directory or log file
    PermissionError
        If insufficient permissions to write to the specified directory
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = out_dir / "train_log.txt"
        
        # Clear any existing handlers to avoid duplicates
        logging.getLogger().handlers.clear()
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s — %(levelname)s — %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ],
        )
        logger = logging.getLogger("phase_09_train_rf")
        logger.info(f"Logger initialized. Log file: {log_file}")
        return logger
        
    except (OSError, PermissionError) as e:
        print(f"Error setting up logger: {e}", file=sys.stderr)
        # Fall back to console-only logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s — %(levelname)s — %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logger = logging.getLogger("phase_09_train_rf")
        logger.warning(f"Could not create log file, using console logging only: {e}")
        return logger
    except Exception as e:
        print(f"Unexpected error setting up logger: {e}", file=sys.stderr)
        raise

def _split_train_val(
    df: pd.DataFrame,
    target_col: str,
    group_col: Optional[str],
    val_frac: float,
    seed: int,

) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """Return X_train, y_train, X_val, y_val, groups_train, groups_val.

    If *val_frac* == 0 → validation outputs are None and all rows go to train.
    """

    y_full = df[target_col]

    # Decide whether to split
    if val_frac <= 0:
        logging.info("Hold‑out disabled (val_frac <= 0). Using all %d rows for CV.", len(df))
        return (
            df.drop(columns=[target_col] + ([group_col] if group_col else [])),
            y_full,
            None,
            None,
            df[group_col] if group_col else None,
            None,
        )

    if group_col and group_col in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, y_full, groups=df[group_col]))
        groups_train = df[group_col].iloc[train_idx]
        groups_val = df[group_col].iloc[val_idx]
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(df)),
            test_size=val_frac,
            stratify=y_full,
            random_state=seed,
        )
        groups_train = None
        groups_val = None

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    X_train = train_df.drop(columns=[target_col] + ([group_col] if group_col else []))
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col] + ([group_col] if group_col else []))
    y_val = val_df[target_col]

    logging.info(
        "Hold‑out split: %d train / %d validation rows (%.1f %%).",
        len(train_df), len(val_df), val_frac * 100,
    )
    return X_train, y_train, X_val, y_val, groups_train, groups_val


def load_data(
        csv_path: Path, 
        target_col: str,
        group_col: Optional[str],
        val_frac: float,
        disable_holdout: bool,
        seed: int
        ) -> tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """Load and validate training data from CSV file.
    
    Parameters
    ----------
    csv_path : Path
        Path to the CSV file containing features and target column
    target_col : str
        Name of the target column in the CSV file
        
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Features (X) and target (y) data
        
    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist
    ValueError
        If target column is missing or non-numeric features are detected
    pd.errors.EmptyDataError
        If the CSV file is empty
    pd.errors.ParserError
        If the CSV file is malformed
    """
    try:
        # Check if file exists
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Load data with error handling
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {csv_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV file {csv_path}: {e}")
            
        # Validate data is not empty
        if df.empty:
            raise ValueError(f"CSV file contains no data: {csv_path}")
            
        # Check target column exists
        if target_col not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(
                f"Target column '{target_col}' not found in CSV. "
                f"Available columns: {available_cols}"
            )
        
        X, y, X_val, y_val, groups_train, groups_val = _split_train_val(
        df, target_col, group_col, 0 if disable_holdout else val_frac, seed
    )
        
            
        # Split features and target
        # X = df.drop(columns=[target_col])
        # y = df[target_col]
        
        # Validate we have features
        if X.empty or X.shape[1] == 0:
            raise ValueError("No feature columns found after removing target column")
            
        # Validate target column has valid values
        if y.isna().all():
            raise ValueError(f"Target column '{target_col}' contains only missing values")
            
        # Sanity checks (skip brittle fixed-shape expectations)
            
        # Check for numeric columns
        numeric_cols = X.select_dtypes(include="number").columns
        non_numeric_cols = X.select_dtypes(exclude="number").columns
        
        if len(non_numeric_cols) > 0:
            logging.warning(f"Non-numeric feature columns detected: {list(non_numeric_cols)}. Removing non-numeric columns.")
            # remove non-numeric columns
            X = X[numeric_cols]
            
        # Log missing values info
        if X.isna().any().any():
            missing_counts = X.isna().sum()
            missing_features = missing_counts[missing_counts > 0]
            logging.info(
                "Missing values detected in %d features – applying median imputation. "
                "Features with missing values: %s",
                len(missing_features), dict(missing_features)
            )
            
        logging.info(
            "Data loaded successfully: %d samples, %d features, target distribution: %s",
            X.shape[0], X.shape[1], y.value_counts().to_dict()
        )
        # Final sanity check
        if not disable_holdout:
            #keep only numeric columns in X_val
            X_val = X_val[numeric_cols]
            #check that X_val has the same number of columns as X
            if X_val.shape[1] != X.shape[1]:
                raise ValueError(f"X_val has {X_val.shape[1]} columns, but X has {X.shape[1]} columns")
        return X, y, X_val, y_val, groups_train, groups_val
        
    except Exception as e:
        logging.error(f"Failed to load data from {csv_path}: {e}")
        raise


def build_cv(y: pd.Series, groups: Optional[pd.Series], seed: int):
    """Build cross-validation strategy based on available grouping information.
    
    Parameters
    ----------
    y : pd.Series
        Target variable for stratification
    groups : Optional[pd.Series]
        Optional grouping variable for StratifiedGroupKFold
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    sklearn cross-validation object
        Either StratifiedGroupKFold or StratifiedKFold depending on groups
        
    Raises
    ------
    ValueError
        If insufficient samples for cross-validation or invalid parameters
    """
    try:
        # Validate inputs
        if len(y) < 10:  # Minimum reasonable sample size for 5-fold CV
            raise ValueError(f"Insufficient samples for cross-validation: {len(y)} (minimum: 10)")
            
        # Check class distribution for stratification
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if min_class_count < 5:  # Need at least 5 samples per class for 5-fold CV
            logging.warning(
                "Small class detected (count=%d). Cross-validation may be unstable. "
                "Class distribution: %s", min_class_count, class_counts.to_dict()
            )
            
        if groups is not None:
            # Validate groups
            if len(groups) != len(y):
                raise ValueError(f"Groups length ({len(groups)}) doesn't match target length ({len(y)})")
                
            unique_groups = groups.nunique()
            if unique_groups < 5:
                logging.warning(
                    "Few unique groups (%d) for GroupKFold. Consider using regular StratifiedKFold.",
                    unique_groups
                )
                
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            logging.info("Using StratifiedGroupKFold with %d unique groups", unique_groups)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            logging.info("Using StratifiedKFold")
            
        return cv
        
    except Exception as e:
        logging.error(f"Failed to build cross-validation strategy: {e}")
        raise


def build_pipeline(seed: int, use_balanced: bool, model: str):
    """Build pipeline with imputation and selected classifier (LightGBM/XGBoost/RF).
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    use_balanced : bool
        Whether to use BalancedRandomForestClassifier from imblearn
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with median imputer and Random Forest classifier
        
    Raises
    ------
    ImportError
        If imblearn is not installed when use_balanced=True
    ValueError
        If invalid parameters are provided
    """
    try:
        # Validate seed
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Seed must be a non-negative integer, got: {seed}")
            
        # Build imputer
        imputer = ("imputer", SimpleImputer(strategy="median"))
        
        # Build classifier based on requested model
        if model == "lightgbm":
            if lgb is None:
                raise ImportError("lightgbm is not installed. pip install lightgbm")
            # Class-imbalance knobs: is_unbalance True lets LGBM auto-weight classes
            clf = lgb.LGBMClassifier(
                n_estimators=600,
                learning_rate=0.05,
                num_leaves=64,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary",
                is_unbalance=True,
                n_jobs=-1,
                random_state=seed,
            )
            logging.info("Using LightGBM classifier (is_unbalance=True)")
        elif model == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed. pip install xgboost")
            # Use scale_pos_weight as imbalance knob; set later after seeing y
            clf = XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                n_jobs=-1,
                tree_method="hist",
                random_state=seed,
            )
            logging.info("Using XGBoost classifier (hist)")
        else:
            # RandomForest fallback
            if use_balanced and BalancedRandomForestClassifier is not None:
                clf = BalancedRandomForestClassifier(
                    n_jobs=-1,
                    random_state=seed,
                    bootstrap=True,
                )
                logging.info("Using BalancedRandomForestClassifier from imblearn")
            else:
                clf = RandomForestClassifier(
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    bootstrap=True,
                )
                logging.info("Using standard RandomForestClassifier with balanced class weights")
            
        # Build pipeline
        pipeline = Pipeline(steps=[imputer, ("clf", clf)])
        logging.info("Pipeline created successfully with steps: %s", [step[0] for step in pipeline.steps])
        
        return pipeline
        
    except Exception as e:
        logging.error(f"Failed to build pipeline: {e}")
        raise

def evaluate_holdout(best_estimator, X_val, y_val, out_dir: Path):
    preds = best_estimator.predict(X_val)
    metrics = {
        "f1": float(f1_score(y_val, preds)),
        "precision": float(precision_score(y_val, preds)),
        "recall": float(recall_score(y_val, preds)),
        "accuracy": float(accuracy_score(y_val, preds)),
    }
    json.dump(metrics, open(out_dir / "holdout_metrics.json", "w"), indent=2)
    logging.info("Hold‑out metrics → %s", metrics)
    return metrics

def main():
    """Main function to execute the Random Forest training pipeline.
    
    This function orchestrates the complete training workflow including:
    - Data loading and validation
    - Cross-validation setup
    - Hyperparameter optimization
    - Model evaluation and persistence
    - Optional metadata storage in DuckDB
    
    The function handles all major error scenarios and provides comprehensive
    logging throughout the process.
    
    Raises
    ------
    SystemExit
        If critical errors occur that prevent training completion
    """
    logger = None
    try:
        # Parse arguments and setup logging
        args = parse_args()
        out_dir = Path(args.output_dir)
        logger = make_logger(out_dir)

        logger.info("✅ Phase 09 training started")
        logger.info("Input CSV: %s", args.input_csv)
        logger.info("Output directory: %s", out_dir)
        logger.info("Random seed: %d", args.seed)

        # Load and validate data
        try:
            # X, y = load_data(Path(args.input_csv), args.target_col)
            X, y, X_val, y_val, groups_train, groups_val = load_data(
                Path(args.input_csv),
                args.target_col,
                args.group_col,
                args.val_frac,
                args.no_holdout,
                args.seed,
            )
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as e:
            logger.error("Data loading failed: %s", e)
            sys.exit(1)

        # Handle grouping column using groups from the split (no leakage)
        groups = groups_train if args.group_col else None
        if args.group_col and groups is not None:
            logger.info("Using '%s' for grouping in CV.", args.group_col)
        elif args.group_col:
            logger.warning("Group column requested but no groups available – defaulting to StratifiedKFold.")

        # Analyze class distribution and compute weights
        try:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y),
                y=y.values
            )
            logger.info("Class distribution: %s", y.value_counts(normalize=True).to_dict())
            logger.info("Computed balanced class weights: %s", dict(zip(np.unique(y), class_weights)))
        except Exception as e:
            logger.error("Failed to compute class weights: %s", e)
            sys.exit(1)

        # Build cross-validation and pipeline
        try:
            cv = build_cv(y, groups, args.seed)
            pipe = build_pipeline(args.seed, args.use_balanced_rf, args.model)
        except (ValueError, ImportError) as e:
            logger.error("Failed to build CV or pipeline: %s", e)
            sys.exit(1)

        # Define hyperparameter search space
        # Hyperparameter space by model
        if args.model == "lightgbm":
            param_dist = {
                "clf__n_estimators": ss.randint(300, 901),
                "clf__num_leaves": ss.randint(31, 129),
                "clf__learning_rate": ss.uniform(0.02, 0.08),
                "clf__subsample": ss.uniform(0.7, 0.3),
                "clf__colsample_bytree": ss.uniform(0.6, 0.4),
                "clf__reg_lambda": ss.uniform(0.0, 2.0),
            }
        elif args.model == "xgboost":
            param_dist = {
                "clf__n_estimators": ss.randint(300, 901),
                "clf__max_depth": ss.randint(4, 9),
                "clf__learning_rate": ss.uniform(0.02, 0.08),
                "clf__subsample": ss.uniform(0.7, 0.3),
                "clf__colsample_bytree": ss.uniform(0.6, 0.4),
                "clf__reg_lambda": ss.uniform(0.0, 2.0),
            }
        else:
            param_dist = {
                "clf__n_estimators": ss.randint(200, 1001),
                "clf__max_depth": [None, 10, 20, 30],
                "clf__max_features": ["sqrt", "log2", 0.7],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_samples": [0.8],
            }

        # Perform hyperparameter optimization
        try:
            logger.info("Starting hyperparameter optimization with %d iterations", args.n_iter)
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

            # If using XGBoost, set scale_pos_weight from class imbalance
            if args.model == "xgboost" and XGBClassifier is not None:
                pos = (y == 1).sum()
                neg = (y == 0).sum()
                spw = max(1.0, neg / max(1, pos))
                if hasattr(pipe.named_steps["clf"], "set_params"):
                    pipe.named_steps["clf"].set_params(scale_pos_weight=spw)
                logger.info("Set XGBoost scale_pos_weight=%.3f (neg/pos)", spw)

            search.fit(X, y, groups=groups)
            logger.info("Best parameters: %s", search.best_params_)
            logger.info("Best CV F1: %.4f", search.best_score_)
            best_est = search.best_estimator_

            # Probability calibration (optional)
            calibrated_est = best_est
            if args.calibrate != "none":
                from sklearn.calibration import CalibratedClassifierCV
                method = "isotonic" if args.calibrate == "isotonic" else "sigmoid"
                if X_val is not None and y_val is not None:
                    calibrator = CalibratedClassifierCV(best_est, method=method, cv="prefit")
                    calibrator.fit(X_val, y_val)
                    calibrated_est = calibrator
                    logger.info("Applied %s calibration on holdout.", method)
                else:
                    calibrator = CalibratedClassifierCV(best_est, method=method, cv=5)
                    calibrated_est = calibrator.fit(X, y)
                    logger.info("Applied %s calibration via CV=5.", method)

        except Exception as e:
            logger.error("Hyperparameter optimization failed: %s", e)
            sys.exit(1)

        # Perform cross-validation for reproducibility metrics
        try:
            logger.info("Computing cross-validation metrics...")
            cv_results = cross_validate(
                calibrated_est if 'calibrated_est' in locals() else search.best_estimator_, X, y,
                cv=cv,
                scoring=["f1", "precision", "recall", "accuracy"],
                n_jobs=-1,
                return_estimator=False
            )
            cv_df = pd.DataFrame(cv_results)
            cv_df.to_csv(out_dir / "cv_results.csv", index=False)
            logger.info("Saved cv_results.csv")
            
            # Log summary statistics
            for metric in ["test_f1", "test_precision", "test_recall", "test_accuracy"]:
                if metric in cv_df.columns:
                    mean_val = cv_df[metric].mean()
                    std_val = cv_df[metric].std()
                    logger.info("%s: %.4f ± %.4f", metric, mean_val, std_val)
                    
        except Exception as e:
            logger.error("Cross-validation evaluation failed: %s", e)
            # Continue execution as this is not critical
        
        # Hold‑out evaluation (if present)
        if X_val is not None:
            try:
                evaluate_holdout(calibrated_est if 'calibrated_est' in locals() else search.best_estimator_, X_val, y_val, out_dir)
                # Save validation probabilities for threshold tuning
                if args.save_val_probs and 'calibrated_est' in locals() and hasattr(calibrated_est, "predict_proba"):
                    val_proba = calibrated_est.predict_proba(X_val)[:, 1]
                    pd.DataFrame({"y_val": y_val.values, "proba": val_proba}).to_csv(
                        out_dir / "val_probs.csv", index=False
                    )
            except Exception as e:
                logger.error("Hold‑out evaluation failed: %s", e)
                # Continue execution as this is not critical

        # Compute permutation importance
        try:
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
            
            importances.to_csv(out_dir / "feature_importance_permutation.csv", index=False)
            logger.info("Saved feature_importance_permutation.csv")
            
            # Log top features
            top_features = importances.head(5)
            for _, row in top_features.iterrows():
                logger.info("Top feature: %s (importance: %.4f ± %.4f)", 
                           row['feature'], row['importance_mean'], row['importance_std'])
                           
        except Exception as e:
            logger.error("Permutation importance computation failed: %s", e)
            # Continue execution as this is not critical

        # Save model artifacts
        try:
            # Threshold selection (chunk/doc gates) on holdout if available
            try:
                if X_val is not None and 'calibrated_est' in locals() and hasattr(calibrated_est, "predict_proba"):
                    from sklearn.metrics import precision_recall_curve
                    proba = calibrated_est.predict_proba(X_val)[:, 1]
                    df_eval = pd.DataFrame({"y": y_val.values, "proba": proba})
                    prec, rec, thr = precision_recall_curve(df_eval["y"], df_eval["proba"])
                    f05 = (1+0.5**2) * (prec * rec) / (0.5**2 * prec + rec + 1e-9)
                    t_idx = np.nanargmax(f05[:-1])
                    gate = {"chunk_threshold": float(thr[t_idx])}
                    with open(out_dir / "gating_thresholds.json", "w") as f:
                        json.dump(gate, f, indent=2)
                    logger.info("Saved gating thresholds: %s", gate)
            except Exception as e:
                logger.warning("Threshold selection failed: %s", e)

            model_path = out_dir / "rf_model.pkl"
            joblib.dump(calibrated_est if 'calibrated_est' in locals() else search.best_estimator_, model_path, compress=3)

            # Also export a safe, portable copy using skops
            skops_path = out_dir / "rf_model.skops"
            try:
                sio.dump(calibrated_est if 'calibrated_est' in locals() else search.best_estimator_, skops_path)
                # Persist list of types required to load safely (useful for Kaggle `trusted=`)
                try:
                    unknown_types = sio.get_untrusted_types(file=str(skops_path))
                    with open(out_dir / "skops_untrusted_types.json", "w") as f:
                        json.dump(sorted(unknown_types), f, indent=2)
                except Exception as e:
                    logger.warning("Could not compute skops untrusted types: %s", e)
            except Exception as e:
                logger.warning("Failed to save skops model: %s", e)
            
            with open(out_dir / "best_params.json", "w") as f:
                json.dump(search.best_params_, f, indent=2)
                
            with open(out_dir / "feature_names.json", "w") as f:
                json.dump(list(X.columns), f, indent=2)
                
            logger.info("Model and artefacts persisted to %s", out_dir)
            logger.info("Model file size: %.2f MB", model_path.stat().st_size / (1024 * 1024))
            
        except Exception as e:
            logger.error("Failed to save model artifacts: %s", e)
            sys.exit(1)

        # Optional: store metadata to DuckDB
        if args.duckdb_path:
            try:
                import duckdb  # Deferred import
                
                # Create parent directory if needed
                duckdb_path = Path(args.duckdb_path)
                duckdb_path.parent.mkdir(parents=True, exist_ok=True)
                
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
                    (sha256, datetime.now(timezone.utc), float(search.best_score_), str(model_path))
                )
                con.close()
                logger.info("Model metadata registered in DuckDB (%s).", args.duckdb_path)
                
            except ImportError:
                logger.warning("DuckDB not available - skipping metadata storage")
            except Exception as exc:
                logger.exception("DuckDB metadata storage failed: %s", exc)

        logger.info("🏁 Training complete. Artefacts available in %s", out_dir)
        
    except KeyboardInterrupt:
        if logger:
            logger.info("Training interrupted by user")
        else:
            print("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        if logger:
            logger.exception("Unexpected error during training: %s", e)
        else:
            print(f"Unexpected error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
