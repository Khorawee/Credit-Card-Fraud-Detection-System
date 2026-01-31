from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score



def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model using multiple performance metrics.

    This evaluation is designed specifically for imbalanced problems
    such as credit card fraud detection.

    Key focus:
    - Recall: ability to catch fraudulent transactions
    - Precision: reliability of fraud predictions
    - ROC-AUC: overall class separation quality

    Args:
        model: trained machine learning model
        X_test: test feature set
        y_test: true labels

    Returns:
        dict: evaluation results
    """

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    try:
        # Predict class labels
        y_pred = model.predict(X_test)

        # Predict fraud probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print(f"Prediction completed for {len(y_pred):,} samples")

        # -----------------------------
        # Classification Report
        # -----------------------------
        report = classification_report(
            y_test,
            y_pred,
            digits=4
        )

        print("\n--- Classification Report ---")
        print(report)

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        cm = confusion_matrix(y_test, y_pred)

        print("\n--- Confusion Matrix ---")
        print(cm)

        tn, fp, fn, tp = cm.ravel()

        # -----------------------------
        # Evaluation Metrics
        # -----------------------------
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        # -----------------------------
        # Display key metrics
        # -----------------------------
        print("\n--- Key Metrics ---")
        print(f"Accuracy:              {accuracy:.4f}")
        print(f"ROC-AUC:               {roc_auc:.4f}")
        print(f"Precision (Fraud):     {precision:.4f}")
        print(f"Recall (Fraud):        {recall:.4f}")
        print(f"F1-score:              {f1:.4f}")
        print(f"False Positive Rate:   {false_positive_rate:.4f}")

        # -----------------------------
        # Store results
        # -----------------------------
        results = {
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positive_rate": false_positive_rate,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        }

        return results

    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


def print_prediction_examples(model, X_test, y_test, n_examples=5):
    """
    Display sample predictions for manual inspection.

    This helps verify whether the model behavior makes sense.

    Args:
        model: trained model
        X_test: test features
        y_test: true labels
        n_examples: number of samples to display
    """

    print("\n" + "=" * 60)
    print(f"PREDICTION EXAMPLES (First {n_examples} samples)")
    print("=" * 60)

    y_pred = model.predict(X_test[:n_examples])
    y_pred_proba = model.predict_proba(X_test[:n_examples])[:, 1]

    for i in range(n_examples):
        true_label = "Fraud" if y_test.iloc[i] == 1 else "Normal"
        pred_label = "Fraud" if y_pred[i] == 1 else "Normal"
        probability = y_pred_proba[i]

        status = "CORRECT" if y_test.iloc[i] == y_pred[i] else "WRONG"

        print(
            f"[{status}] Sample {i + 1} | "
            f"True: {true_label} | "
            f"Predicted: {pred_label} | "
            f"Fraud Probability: {probability:.4f}"
        )

def plot_class_distribution(y):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Count")

    path = "output/figures/class_distribution.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")

def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plot Precision-Recall Curve for imbalanced classification.
    """

    y_scores = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()

    path = "output/figures/precision_recall_curve.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot top feature importances.
    """

    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature importance")
        return

    importance = model.feature_importances_

    fi = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=fi,
        x="Importance",
        y="Feature"
    )

    plt.title("Top Feature Importances")

    path = "output/figures/feature_importance.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")

def plot_amount_distribution(df, target_column):
    """
    Plot transaction amount distribution for fraud vs normal.
    """

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[df[target_column] == 0]["Amount"],
        bins=50,
        stat="density",
        label="Normal",
        alpha=0.6
    )

    sns.histplot(
        df[df[target_column] == 1]["Amount"],
        bins=50,
        stat="density",
        label="Fraud",
        alpha=0.6
    )

    plt.legend()
    plt.xlabel("Transaction Amount")
    plt.title("Transaction Amount Distribution")

    path = "output/figures/amount_distribution.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")
