# src/utils.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, TARGET, RANDOM_SEED, MODEL_DIR


def generate_synthetic(n_samples=1200, random_state=RANDOM_SEED):
    """Generate synthetic dataset with 8 numeric features + target."""
    rng = np.random.RandomState(random_state)
    X = rng.normal(size=(n_samples, len(FEATURES)))

    # Synthetic nonlinear pattern
    logits = (
        (X[:, 0] * 0.8)
        - (X[:, 1] * 0.5)
        + (X[:, 2] ** 2 * 0.3)
        + rng.normal(scale=0.5, size=n_samples)
    )

    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET] = y
    return df


def load_csv(path, target_col=TARGET):
    """Load CSV and automatically select first 8 numeric columns if extra columns are present.
       Returns a DataFrame whose last column is the target_col.
    """
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. Columns: {list(df.columns)}"
        )

    # candidate feature columns (exclude target_col)
    feature_cols = [c for c in df.columns if c != target_col]

    # if not exactly the expected number, pick numeric columns (excluding target) and take first len(FEATURES)
    if len(feature_cols) != len(FEATURES):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]

        if len(numeric_cols) < len(FEATURES):
            raise ValueError(
                f"CSV must have at least {len(FEATURES)} numeric features. Found: {numeric_cols}"
            )

        feature_cols = numeric_cols[: len(FEATURES)]

    # reorder dataframe to have features then target_col (feature order used for saving)
    selected = feature_cols + [target_col]
    return df[selected].copy()


def prepare_train_test(df, target_col=TARGET, test_size=0.2, random_state=RANDOM_SEED, stratify=True):
    """Split data & apply StandardScaler.

    df: DataFrame where feature columns are all columns except the target_col.
    target_col: name of the target column to use in df.
    """
    # Ensure target_col exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame. Columns: {list(df.columns)}")

    feature_columns = [c for c in df.columns if c != target_col]
    X = df[feature_columns].values
    y = df[target_col].values

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler, feature_columns


def save_artifact(model_obj, scaler, feature_order, path):
    """Save model + scaler + feature order as a .joblib artifact."""
    os.makedirs(os.path.dirname(path) or MODEL_DIR, exist_ok=True)
    joblib.dump(
        {"model": model_obj, "scaler": scaler, "feature_order": feature_order}, path
    )


def load_artifact(path):
    """Load model + scaler artifact."""
    return joblib.load(path)
