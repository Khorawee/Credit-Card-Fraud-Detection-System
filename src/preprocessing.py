import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.config import SCALER_PATH


def clean_data(df):
    """
    Perform data cleaning.
    - Remove duplicate rows
    - Remove missing values (NaN)
    - Remove infinite values

    Args:
        df: input DataFrame

    Returns:
        DataFrame after cleaning
    """
    print("\n=== Data Cleaning ===")
    original_size = len(df)

    # Remove duplicate rows
    df = df.drop_duplicates()
    duplicates_removed = original_size - len(df)
    if duplicates_removed > 0:
        print(f"Removed duplicate rows: {duplicates_removed}")

    # Remove missing values
    df = df.dropna()
    na_removed = original_size - duplicates_removed - len(df)
    if na_removed > 0:
        print(f"Removed NaN rows: {na_removed}")

    # Remove infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols]).any(axis=1)

    if inf_mask.sum() > 0:
        print(f"Removed infinite values: {inf_mask.sum()} rows")
        df = df[~inf_mask]

    print(f"Final dataset size: {len(df):,} rows")

    return df


def validate_data(df, expected_target="Class"):
    """
    Validate dataset structure and target distribution.

    Args:
        df: DataFrame to validate
        expected_target: target column name
    """
    print("\n=== Data Validation ===")

    # Check empty dataset
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check target column existence
    if expected_target not in df.columns:
        raise ValueError(f"Target column '{expected_target}' not found.")

    # Check class distribution
    class_counts = df[expected_target].value_counts()
    print("Class distribution:")

    for cls, count in class_counts.items():
        print(f"Class {cls}: {count:,} samples ({count / len(df) * 100:.2f}%)")

    # Warn about class imbalance
    if len(class_counts) == 2:
        minority_ratio = class_counts.min() / class_counts.sum()
        if minority_ratio < 0.1:
            print("Warning: Dataset is highly imbalanced.")
            print("Class weighting or resampling is recommended.")

    print("Data validation completed successfully.")


def split_xy(df, target_column):
    """
    Split dataset into features (X) and target (y).

    Args:
        df: DataFrame
        target_column: name of target column

    Returns:
        X: features
        y: target values
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print(f"\nData split completed: X={X.shape}, y={y.shape}")

    return X, y


def scale_features(X, is_training=True, scaler=None):
    """
    Scale features using StandardScaler.

    Important:
    - The scaler must be fitted ONLY on training data.
    - The same scaler must be reused for testing and prediction.

    Args:
        X: feature matrix
        is_training: True for training mode, False for inference mode
        scaler: optional scaler object

    Returns:
        X_scaled: scaled features
        scaler: fitted scaler (training mode only)
    """
    if is_training:
        print("\n=== Feature Scaling (Training Mode) ===")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)

        print(f"Scaler saved to: {SCALER_PATH}")

        return X_scaled, scaler

    else:
        print("\n=== Feature Scaling (Inference Mode) ===")

        if scaler is None:
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError("Saved scaler not found.")
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded from: {SCALER_PATH}")

        X_scaled = scaler.transform(X)

        return X_scaled


def get_feature_statistics(X_train, X_test):
    """
    Display feature statistics for training and testing sets.
    Used to verify that no data leakage has occurred.

    Args:
        X_train: training features
        X_test: testing features
    """
    print("\n=== Feature Statistics ===")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape:  {X_test.shape}")

    train_mean = np.mean(X_train, axis=0).mean()
    test_mean = np.mean(X_test, axis=0).mean()

    print(f"Training mean: {train_mean:.4f}")
    print(f"Testing mean:  {test_mean:.4f}")

def save_processed_data(df):
    """
    Save cleaned & processed dataset to data/processed directory
    """

    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")

    os.makedirs(processed_dir, exist_ok=True)

    file_path = os.path.join(processed_dir, "clean_data.csv")
    df.to_csv(file_path, index=False)

    print(f"\nProcessed clean data saved to:")
    print(f"→ {file_path}")