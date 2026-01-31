import os

# =========================
# Project paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_PATH = os.path.join(
    BASE_DIR, "data", "raw", "creditcard.csv"
)

DATA_PROCESSED_PATH = os.path.join(
    BASE_DIR, "data", "processed", "cleaned_data_creditcard.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# =========================
# Training setup
# =========================
TARGET_COLUMN = "Class"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==================================================
# Logistic Regression (baseline model)
# ==================================================
LOGISTIC_CONFIG = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": RANDOM_STATE
}


# ==================================================
# XGBoost (main model)
# ==================================================
XGB_CONFIG = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_STATE
}
