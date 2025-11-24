# src/evaluate.py
"""
Evaluate a saved model on a CSV dataset and save:
- classification_report.txt
- predictions.csv (with pred & prob)
- confusion_matrix.png (requires matplotlib)
- roc_curve.png (requires matplotlib)
Usage:
    python src/evaluate.py --model models/rf.joblib --csv data/diabetes_clean.csv --target Outcome --out reports
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from src.utils import load_artifact

def evaluate(model_path, csv_path, target_col, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    artifact = load_artifact(model_path)
    model = artifact["model"]
    scaler = artifact.get("scaler", None)
    feature_order = artifact.get("feature_order", None)

    df = pd.read_csv(csv_path)

    # Determine feature columns to use
    if feature_order and all(c in df.columns for c in feature_order):
        used_cols = feature_order
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric:
            numeric = [c for c in numeric if c != target_col]
        used_cols = numeric[:len(feature_order) if feature_order else 8]

    X = df[used_cols].astype(float).values
    y = df[target_col].values

    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X

    preds = model.predict(Xs)
    probs = model.predict_proba(Xs)[:,1] if hasattr(model, "predict_proba") else None

    # Save classification report
    crep = classification_report(y, preds, digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(crep)

    # Save predictions csv
    out_df = df.copy()
    out_df["pred"] = preds
    if probs is not None:
        out_df["prob"] = probs
    out_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # ROC curve (if probabilities available)
    if probs is not None:
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_curve.png"))
        plt.close()

    print("Evaluation completed. Files saved to:", out_dir)
    print(crep)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", dest="out_dir", default="reports")
    args = p.parse_args()
    evaluate(args.model, args.csv, args.target, out_dir=args.out_dir)
