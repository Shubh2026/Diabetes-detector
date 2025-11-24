# src/predict.py
"""
Load a saved model artifact and run prediction on a single input row (8 numeric values).

Usage examples:
# 1) Provide 8 values directly:
python src\predict.py --vals 0.1 -0.2 1.0 0.5 0.0 -0.3 0.2 0.8

# 2) Use a CSV row (default row 0):
python src\predict.py --csv data\sample_diabetes.csv --row 0

# 3) Specify a model:
python src\predict.py --model models/logreg.joblib --csv data\sample_diabetes.csv --row 2
"""

import sys
import os

# Add project root (parent of 'src') so relative imports work when running "python src\predict.py"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd

from src.utils import load_artifact
from src.config import FEATURES, DEFAULT_MODEL_NAME, MODEL_DIR

def parse_vals_list(vals_list):
    vals = [float(v) for v in vals_list]
    if len(vals) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} numeric values (order: {FEATURES}), got {len(vals)}")
    return np.array(vals).reshape(1, -1)

def extract_from_csv(csv_path, row_idx, artifact_feature_order):
    """
    Read csv_path and extract feature values for row_idx.
    Prefer artifact_feature_order if all feature names present in CSV.
    Otherwise select first len(FEATURES) numeric columns (excluding any target column).
    Returns: numpy array shape (1, len(FEATURES)), list of used column names
    """
    df = pd.read_csv(csv_path)
    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"Row index {row_idx} out of bounds (csv has {len(df)} rows).")

    # If artifact stored a feature order and all those columns exist in CSV, use them
    if artifact_feature_order is not None:
        missing = [c for c in artifact_feature_order if c not in df.columns]
        if not missing and len(artifact_feature_order) == len(FEATURES):
            cols = artifact_feature_order
            vals = df.loc[row_idx, cols].astype(float).tolist()
            return np.array(vals).reshape(1, -1), cols

    # Otherwise, try to pick numeric columns (exclude non-numeric, keep order)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # If numeric columns include the target column name 'target', remove it from selection
    if "target" in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != "target"]

    if len(numeric_cols) < len(FEATURES):
        raise ValueError(f"CSV does not contain at least {len(FEATURES)} numeric columns. Found: {numeric_cols}")

    cols = numeric_cols[: len(FEATURES)]
    vals = df.loc[row_idx, cols].astype(float).tolist()
    return np.array(vals).reshape(1, -1), cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME),
                        help="Path to model .joblib")
    parser.add_argument("--vals", nargs="+", help="8 numeric feature values (space separated)")
    parser.add_argument("--csv", type=str, help="Path to CSV file to read a row from")
    parser.add_argument("--row", type=int, default=0, help="Row index (0-based) when using --csv")
    args = parser.parse_args()

    # Load artifact
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model artifact not found: {args.model}. Train first or pass correct path with --model")

    artifact = load_artifact(args.model)
    model = artifact["model"]
    scaler = artifact.get("scaler", None)
    artifact_feature_order = artifact.get("feature_order", None)

    # Build input array
    if args.csv:
        arr, used_cols = extract_from_csv(args.csv, args.row, artifact_feature_order)
        source_info = f"CSV: {args.csv}, row: {args.row}"
    elif args.vals:
        arr = parse_vals_list(args.vals)
        used_cols = artifact_feature_order if artifact_feature_order is not None else FEATURES
        source_info = "Command-line values (--vals)"
    else:
        raise ValueError("Specify either --vals (8 numbers) or --csv path (and optional --row).")

    # Scale if scaler exists
    if scaler is not None:
        arr_s = scaler.transform(arr)
    else:
        arr_s = arr

    # Predict
    pred = model.predict(arr_s)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(arr_s)[0, 1])

    # Output
    print("=== Prediction Result ===")
    print(source_info)
    print("Feature columns used (order):", used_cols)
    print("Input values:", arr.flatten().tolist())
    print("Predicted class (0 = no, 1 = yes):", int(pred))
    if prob is not None:
        print(f"Predicted probability for class=1: {prob:.4f}")

if __name__ == "__main__":
    main()
