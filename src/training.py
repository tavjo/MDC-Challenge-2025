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
# from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            #  recall_score)
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
            description="Train Random‑Forest (Phase 09) according to MDC checklist."
        )
        parser.add_argument("--input_csv", type=str, required=True,
                            help="Path to input CSV containing features + label.")
        parser.add_argument("--duckdb_path", type=str, default="artifacts/mdc-challenge.db",
                            help="Optional DuckDB file; if provided, model metadata is stored there.")
        parser.add_argument("--output_dir", type=str, default="artifacts/models",
                            help="Directory to write all artefacts into.")
        parser.add_argument("--target_col", type=str, default="target",
                            help="Name of the target column in CSV (default: target).")
        parser.add_argument("--group_col", type=str, default=None,
                            help="Optional grouping column for StratifiedGroupKFold.")
        parser.add_argument("--seed", type=int, default=42,
                            help="Random seed.")
        parser.add_argument("--n_iter", type=int, default=100,
                            help="RandomizedSearch iterations (default: 100).")
        parser.add_argument("--use_balanced_rf", action="store_true",
                            help="If set, use BalancedRandomForestClassifier from imblearn.")
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


def load_data(csv_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
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
            
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Validate we have features
        if X.empty or X.shape[1] == 0:
            raise ValueError("No feature columns found after removing target column")
            
        # Validate target column has valid values
        if y.isna().all():
            raise ValueError(f"Target column '{target_col}' contains only missing values")
            
        # Sanity checks
        if X.shape != (487, 55):
            logging.warning(
                "Data shape is %s (expected 487, 55). Continuing anyway.", X.shape
            )
            
        # Check for numeric columns
        numeric_cols = X.select_dtypes(include="number").columns
        non_numeric_cols = X.select_dtypes(exclude="number").columns
        
        if len(non_numeric_cols) > 0:
            logging.warning(f"Non-numeric feature columns detected: {list(non_numeric_cols)}. Removing non-numeric columns.")
            # remove non-numeric columns
            X = X[numeric_cols]
            # raise ValueError(
            #     f"Non-numeric feature columns detected: {list(non_numeric_cols)}. "
            #     "All features must be numeric."
            # )
            
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
        
        return X, y
        
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


def build_pipeline(seed: int, use_balanced: bool):
    """Build scikit-learn pipeline with imputation and Random Forest classifier.
    
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
        
        # Build classifier based on balance preference
        if use_balanced:
            if BalancedRandomForestClassifier is None:
                raise ImportError(
                    "imblearn is not installed. Install with: pip install imbalanced-learn"
                )
                
            clf = BalancedRandomForestClassifier(
                n_jobs=-1, 
                random_state=seed, 
                bootstrap=True
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
        pipeline = Pipeline(steps=[imputer, ("rf", clf)])
        logging.info("Pipeline created successfully with steps: %s", [step[0] for step in pipeline.steps])
        
        return pipeline
        
    except Exception as e:
        logging.error(f"Failed to build pipeline: {e}")
        raise


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
            X, y = load_data(Path(args.input_csv), args.target_col)
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as e:
            logger.error("Data loading failed: %s", e)
            sys.exit(1)

        # Handle grouping column
        groups = None
        if args.group_col:
            if args.group_col in X.columns:
                groups = X[args.group_col]
                X = X.drop(columns=[args.group_col])
                logger.info("Using '%s' column for grouping in CV.", args.group_col)
            else:
                logger.warning("Group column '%s' not found – defaulting to StratifiedKFold.", args.group_col)

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
            pipe = build_pipeline(args.seed, args.use_balanced_rf)
        except (ValueError, ImportError) as e:
            logger.error("Failed to build CV or pipeline: %s", e)
            sys.exit(1)

        # Define hyperparameter search space
        param_dist = {
            "rf__n_estimators": ss.randint(200, 1001),
            "rf__max_depth": [None, 10, 20, 30],
            "rf__max_features": ["sqrt", "log2", 0.7],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__max_samples": [0.8],
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

            search.fit(X, y, groups=groups)
            logger.info("Best parameters: %s", search.best_params_)
            logger.info("Best CV F1: %.4f", search.best_score_)
            
        except Exception as e:
            logger.error("Hyperparameter optimization failed: %s", e)
            sys.exit(1)

        # Perform cross-validation for reproducibility metrics
        try:
            logger.info("Computing cross-validation metrics...")
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
            
            # Log summary statistics
            for metric in ["test_f1", "test_precision", "test_recall", "test_accuracy"]:
                if metric in cv_df.columns:
                    mean_val = cv_df[metric].mean()
                    std_val = cv_df[metric].std()
                    logger.info("%s: %.4f ± %.4f", metric, mean_val, std_val)
                    
        except Exception as e:
            logger.error("Cross-validation evaluation failed: %s", e)
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
            model_path = out_dir / "rf_model.pkl"
            joblib.dump(search.best_estimator_, model_path, compress=3)
            
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
                    (sha256, datetime.utcnow(), float(search.best_score_), str(model_path))
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
