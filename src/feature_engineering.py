import numpy as np
import pandas as pd


def feature_engineering(df):
    """
    Create additional features from the original dataset.

    Args:
        df: input DataFrame

    Returns:
        DataFrame with engineered features
    """
    print("\n=== Feature Engineering ===")

    df = df.copy()

    if "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))

        print("Created feature: Amount_log")
        print(
            f"Amount range: "
            f"min={df['Amount'].min():.2f}, "
            f"max={df['Amount'].max():.2f}"
        )

    if "Time" in df.columns:
        df["Time_hour"] = (df["Time"].fillna(0) // 3600).astype(int) % 24
        print("Created feature: Time_hour")

    print(f"Feature engineering completed: {len(df.columns)} features")

    return df


def show_feature_importance(feature_names, importance_scores, top_n=10):
    """
    Display the most important features after model training.

    Args:
        feature_names: list of feature names
        importance_scores: feature importance values
        top_n: number of top features to display
    """
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(importance_scores)
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print(f"\nTop {top_n} Important Features")

    for _, row in feature_importance.head(top_n).iterrows():
        print(f"{row['feature']:20s}: {row['importance']:.4f}")
