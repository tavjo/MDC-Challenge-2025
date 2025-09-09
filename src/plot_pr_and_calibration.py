#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    PrecisionRecallDisplay,
    f1_score, precision_score, recall_score,
)
from sklearn.calibration import CalibrationDisplay

# import load_data from your training script (no retrain happens)
from filter_training import load_data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--target_col", default="target")
    p.add_argument("--group_col", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.20)
    p.add_argument("--out_dir", default="artifacts/plots")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Recreate the same split as training (no fitting here)
    X, y, X_val, y_val, _ = load_data(
        Path(args.input_csv),
        args.target_col,
        args.group_col,
        args.val_frac,
        disable_holdout=False,
        seed=args.seed,
    )

    model = joblib.load(args.model_path)
    # Ensure feature alignment with training
    model_dir = Path(args.model_path).parent
    feat_path = model_dir / "feature_names.json"
    if feat_path.exists():
        with open(feat_path, "r") as f:
            train_features = json.load(f)
        # add any missing columns as NaN (imputer in the saved Pipeline will handle them)
        for col in train_features:
            if col not in X_val.columns:
                X_val[col] = np.nan
        # drop any unexpected extra columns and order exactly as during training
        X_val = X_val[train_features]

    val_proba = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, val_proba)

    # --- Precision–Recall curve & table ---
    disp = PrecisionRecallDisplay.from_predictions(
        y_val, val_proba, name=f"Hold-out (AP={ap:.3f})"
    )
    disp.ax_.set_title("Precision–Recall (hold-out)")
    pr_png = out_dir / "pr_curve.png"
    plt.savefig(pr_png, dpi=150, bbox_inches="tight")
    plt.close()

    precision, recall, thr = precision_recall_curve(y_val, val_proba)
    # thresholds align with precision[1:], recall[1:]
    rows = []
    for t, p_, r_ in zip(thr, precision[1:], recall[1:]):
        preds = (val_proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(p_),
            "recall": float(r_),
            "f1": float(f1_score(y_val, preds)),
            "selected": int(preds.sum()),
        })
    pr_table = pd.DataFrame(rows).sort_values("threshold", ascending=False)
    pr_csv = out_dir / "threshold_table.csv"
    pr_table.to_csv(pr_csv, index=False)

    # quick suggestions for common precision floors
    summary = {}
    for floor in (0.30, 0.50, 0.70):
        cand = pr_table[pr_table["precision"] >= floor]
        if len(cand):
            best = cand.tail(1)  # smallest threshold meeting the floor
            summary[f"precision>={floor:.2f}"] = best.iloc[0].to_dict()
        else:
            summary[f"precision>={floor:.2f}"] = None

    with open(out_dir / "pr_summary.json", "w") as f:
        json.dump({"average_precision": float(ap), **summary}, f, indent=2)

    # --- Calibration / reliability diagram ---
    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(
        y_true=y_val,
        y_prob=val_proba,
        n_bins=10,
        strategy="uniform",
        name="Hold-out"
    ).plot(ax=ax)
    ax.set_title("Calibration curve (hold-out)")
    calib_png = out_dir / "calibration_curve.png"
    plt.savefig(calib_png, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Score histogram (helps pick cut-offs) ---
    plt.hist(val_proba, bins=30)
    plt.title("Predicted probability histogram (hold-out)")
    plt.xlabel("p(y=1)"); plt.ylabel("count")
    hist_png = out_dir / "proba_hist.png"
    plt.savefig(hist_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved:\n- {pr_png}\n- {pr_csv}\n- {out_dir/'pr_summary.json'}\n- {calib_png}\n- {hist_png}")

if __name__ == "__main__":
    main()
