import os
import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, SCALER_PATH, DATA_RAW_PATH
from src.preprocessing import clean_data
from src.feature_engineering import feature_engineering


def predict(input_data, return_proba=False):
    """
    Predict whether transactions are fraudulent.

    Args:
        input_data: pandas DataFrame or numpy array
        return_proba: return prediction probabilities if True

    Returns:
        predictions or (predictions, probabilities)
    """
    try:
        print("\n=== Prediction ===")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        if isinstance(input_data, pd.DataFrame):
            print(f"Input shape: {input_data.shape}")

            input_data = clean_data(input_data)
            input_data = feature_engineering(input_data)

            if "Class" in input_data.columns:
                input_data = input_data.drop("Class", axis=1)

        input_scaled = scaler.transform(input_data)

        predictions = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)

        normal_count = np.sum(predictions == 0)
        fraud_count = np.sum(predictions == 1)

        print(f"Normal: {normal_count}")
        print(f"Fraud: {fraud_count}")

        if return_proba:
            return predictions, probabilities

        return predictions

    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Model or scaler not found. Please train the model first."
        ) from e


def predict_single_transaction(transaction_data):
    """
    Predict a single transaction.

    Args:
        transaction_data: dict or DataFrame

    Returns:
        dict with prediction results
    """
    if isinstance(transaction_data, dict):
        df = pd.DataFrame([transaction_data])
    else:
        df = transaction_data

    predictions, probabilities = predict(df, return_proba=True)

    result = {
        "prediction": "Fraud" if predictions[0] == 1 else "Normal",
        "is_fraud": bool(predictions[0]),
        "fraud_probability": float(probabilities[0][1]),
        "normal_probability": float(probabilities[0][0]),
    }

    return result


def batch_predict(csv_path, output_path=None):
    """
    Predict transactions from a CSV file.

    Args:
        csv_path: input CSV file
        output_path: optional output CSV path

    Returns:
        DataFrame with predictions
    """
    df = pd.read_csv(csv_path)
    original_df = df.copy()

    predictions, probabilities = predict(df, return_proba=True)

    original_df["Prediction"] = predictions
    original_df["Fraud_Probability"] = probabilities[:, 1]
    original_df["Normal_Probability"] = probabilities[:, 0]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        original_df.to_csv(output_path, index=False)

    return original_df


if __name__ == "__main__":

    if not os.path.exists(DATA_RAW_PATH):
        raise FileNotFoundError(
            "creditcard.csv not found in data/raw/"
        )

    df = pd.read_csv(DATA_RAW_PATH)

    sample_df = df.head(1000)

    predictions, probabilities = predict(
        sample_df,
        return_proba=True
    )

    result_df = sample_df.copy()
    result_df["Prediction"] = predictions
    result_df["Fraud_Probability"] = probabilities[:, 1]

    os.makedirs("output", exist_ok=True)
    result_df.to_csv(
        "output/predictions.csv",
        index=False
    )

    print("Prediction completed.")
