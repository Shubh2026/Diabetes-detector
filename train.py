# src/train.py
"""
Train script for the diabetes predictor.
Creates two models (RandomForest and LogisticRegression) and saves them to models/.

Usage:
    python src/train.py                           # uses synthetic data
    python src/train.py --csv data/diabetes.csv   # uses CSV (default target from config)
    python src/train.py --csv data/diabetes.csv --target Outcome
    python src/train.py --csv data/diabetes.csv --save_dir models_custom --target Outcome
"""

import sys
import os

# Add project root (parent of 'src' folder) to sys.path so `from src...` works
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from src.config import FEATURES, TARGET as CONFIG_TARGET, MODEL_DIR, RANDOM_SEED
from src.utils import generate_synthetic, load_csv, prepare_train_test, save_artifact

def train_and_save(df, save_dir=MODEL_DIR, target_col=CONFIG_TARGET):
    # prepare_train_test now returns scaler and also feature_columns used
    X_train_s, X_test_s, y_train, y_test, scaler, feature_columns = prepare_train_test(df, target_col=target_col)

    models = {
        "logreg": LogisticRegression(max_iter=300, random_state=RANDOM_SEED),
        "rf": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    }

    results = {}
    os.makedirs(save_dir, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, probs) if probs is not None else None

        print(f"Model: {name}")
        print(f"  Accuracy: {acc:.4f}")
        if roc is not None:
            print(f"  ROC AUC: {roc:.4f}")
        print("  Classification report:")
        print(classification_report(y_test, preds, digits=4))

        save_path = os.path.join(save_dir, f"{name}.joblib")
        # save feature_columns (the actual column names used) so predict.py can use same order
        save_artifact(model, scaler, feature_columns, save_path)

        results[name] = {"accuracy": acc, "roc_auc": roc, "path": save_path}

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file with data")
    parser.add_argument("--target", type=str, default=None, help="Name of target column in CSV (overrides config TARGET)")
    parser.add_argument("--save_dir", type=str, default=MODEL_DIR, help="Directory to save trained models")
    args = parser.parse_args()

    # Decide which target column to use: CLI override > config default
    target_col = args.target if args.target is not None else CONFIG_TARGET

    if args.csv:
        print(f"Loading CSV from: {args.csv} (target column: {target_col})")
        df = load_csv(args.csv, target_col=target_col)
    else:
        print("No CSV provided — generating synthetic demo dataset.")
        df = generate_synthetic()

    print("Data snapshot:")
    print(df.head())

    results = train_and_save(df, save_dir=args.save_dir, target_col=target_col)
    print("\nTraining complete. Saved models:")
    for k, v in results.items():
        print(f" - {k}: {v['path']}")

if __name__ == "__main__":
    main()
