from xgboost import XGBClassifier
import joblib
import os
import numpy as np
from src.config import MODEL_PATH, XGB_CONFIG


def train_model(X_train, y_train):

    print("\n" + "=" * 50)
    print("Model Training (XGBoost)")
    print("=" * 50)

    try:
        # ===============================
        # Training info
        # ===============================
        print(f"Training samples: {X_train.shape[0]:,}")
        print(f"Number of features: {X_train.shape[1]}")

        # ===============================
        # Class distribution
        # ===============================
        unique, counts = np.unique(y_train, return_counts=True)

        neg, pos = counts
        scale_pos_weight = neg / pos

        print("\nClass distribution:")
        print(f"Class 0: {neg:,}")
        print(f"Class 1: {pos:,}")
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")

        # ===============================
        # Initialize model
        # ===============================
        print("\nInitializing XGBoost model:")
        for k, v in XGB_CONFIG.items():
            print(f"  {k}: {v}")

        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            **XGB_CONFIG
        )

        # ===============================
        # Train
        # ===============================
        print("\nTraining model...")
        model.fit(X_train, y_train)

        print("Training completed successfully ✅")

        # ===============================
        # Save model
        # ===============================
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Model saved to: {MODEL_PATH}")

        return model

    except Exception as e:
        print(f"Error occurred during model training: {e}")
        raise
