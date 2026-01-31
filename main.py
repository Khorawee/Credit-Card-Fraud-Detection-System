# main.py

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay

from src.data_loader import load_raw_data
from src.preprocessing import (
    clean_data,
    validate_data,
    split_xy,
    scale_features,
    get_feature_statistics
)
from src.feature_engineering import feature_engineering
from src.train import train_model
from src.evaluate import (
    evaluate_model,
    print_prediction_examples,
    plot_class_distribution,
    plot_amount_distribution,
    plot_precision_recall_curve,
    plot_feature_importance
)
from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    METRICS_DIR
)


def main():
    """
    Credit Card Fraud Detection - Training Pipeline
    """

    print("\n" + "=" * 70)
    print("Credit Card Fraud Detection - Training Pipeline")
    print("=" * 70)

    try:
        # =====================================================
        # Step 1: Create directories
        # =====================================================
        print("\n[Step 1] Creating directories")

        os.makedirs("models", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("output/metrics", exist_ok=True)
        os.makedirs("output/figures", exist_ok=True)
        os.makedirs("output/predictions", exist_ok=True)

        # =====================================================
        # Step 2: Load dataset
        # =====================================================
        print("\n[Step 2] Loading dataset")
        df = load_raw_data()

        # =====================================================
        # Step 3: Data preprocessing
        # =====================================================
        print("\n[Step 3] Data preprocessing")

        df = clean_data(df)
        validate_data(df, TARGET_COLUMN)
        df = feature_engineering(df)

        processed_path = "data/processed/clean_data.csv"
        df.to_csv(processed_path, index=False)
        print(f"Processed data saved to: {processed_path}")

        # ===============================
        # Visualization — Data Analysis
        # ===============================
        print("\n[Visualization] Data distribution")

        plot_class_distribution(df[TARGET_COLUMN])
        plot_amount_distribution(df, TARGET_COLUMN)

        # =====================================================
        # Step 4: Split X and y
        # =====================================================
        print("\n[Step 4] Splitting features and target")

        X, y = split_xy(df, TARGET_COLUMN)

        # =====================================================
        # Step 5: Train-test split
        # =====================================================
        print("\n[Step 5] Train-test split")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        print(f"Training samples: {len(X_train):,}")
        print(f"Testing samples:  {len(X_test):,}")

        # =====================================================
        # Step 6: Feature scaling
        # =====================================================
        print("\n[Step 6] Feature scaling")

        X_train_scaled, scaler = scale_features(
            X_train,
            is_training=True
        )

        X_test_scaled = scale_features(
            X_test,
            is_training=False,
            scaler=scaler
        )

        get_feature_statistics(X_train_scaled, X_test_scaled)

        # =====================================================
        # Step 7: Train model
        # =====================================================
        print("\n[Step 7] Model training")

        model = train_model(X_train_scaled, y_train)

        # =====================================================
        # Step 8: Model evaluation
        # =====================================================
        print("\n[Step 8] Model evaluation")

        results = evaluate_model(model, X_test_scaled, y_test)

        print_prediction_examples(
            model,
            X_test_scaled,
            y_test,
            n_examples=10
        )

        # ===============================
        # Visualization — Model Analysis
        # ===============================
        print("\n[Visualization] Model analysis")

        plot_precision_recall_curve(
            model,
            X_test_scaled,
            y_test
        )

        plot_feature_importance(
            model,
            X.columns
        )

        # =====================================================
        # Save predictions
        # =====================================================
        print("\nSaving predictions")

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        prediction_df = X_test.copy()
        prediction_df["True_Label"] = y_test.values
        prediction_df["Prediction"] = y_pred
        prediction_df["Fraud_Probability"] = y_proba

        prediction_path = "output/predictions/predictions.csv"
        prediction_df.to_csv(prediction_path, index=False)

        print(f"Predictions saved to: {prediction_path}")

        # =====================================================
        # Step 8.5: Confusion Matrix & ROC
        # =====================================================
        print("\n[Step 8.5] Saving evaluation figures")

        cm = results["confusion_matrix"]

        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()

        cm_path = "output/figures/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved: {cm_path}")

        RocCurveDisplay.from_predictions(
            y_test,
            model.predict_proba(X_test_scaled)[:, 1]
        )

        roc_path = "output/figures/roc_curve.png"
        plt.savefig(roc_path)
        plt.close()
        print(f"Saved: {roc_path}")

        # =====================================================
        # Step 9: Save evaluation results
        # =====================================================
        print("\n[Step 9] Saving evaluation results")

        result_path = os.path.join(METRICS_DIR, "result.txt")

        with open(result_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("Credit Card Fraud Detection - Evaluation Results\n")
            f.write("=" * 70 + "\n\n")

            f.write("CLASSIFICATION REPORT\n")
            f.write(results["classification_report"])

            f.write("\n\nKEY METRICS\n")
            f.write(f"Accuracy:   {results['accuracy']:.4f}\n")
            f.write(f"Precision:  {results['precision']:.4f}\n")
            f.write(f"Recall:     {results['recall']:.4f}\n")
            f.write(f"F1-Score:   {results['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:    {results['roc_auc']:.4f}\n")

        print(f"Results saved to: {result_path}")

        print("\n" + "=" * 70)
        print("Training completed successfully ✅")
        print("=" * 70)

        return model, results

    except Exception as e:
        print(f"\nPipeline failed ❌ : {e}")
        raise


if __name__ == "__main__":
    model, results = main()
