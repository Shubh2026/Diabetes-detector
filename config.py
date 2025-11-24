# src/config.py

# The 8 numeric features required for the diabetes model
FEATURES = [f"f{i}" for i in range(1, 9)]

# Target column name (0 or 1)
TARGET = "target"

# Random seed for reproducibility
RANDOM_SEED = 42

# Default folder for saving models
MODEL_DIR = "models"

# Default model filename (Random Forest)
DEFAULT_MODEL_NAME = "rf.joblib"

# List of allowed model types (for future extension)
ALLOWED_MODELS = ["rf", "logreg"]
