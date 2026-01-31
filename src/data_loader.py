import os
import pandas as pd
from src.config import DATA_RAW_PATH


def load_raw_data():
    """
    Load raw dataset from CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    try:
        if not os.path.exists(DATA_RAW_PATH):
            raise FileNotFoundError(f"File not found: {DATA_RAW_PATH}")

        print(f"Loading data from {DATA_RAW_PATH}...")
        df = pd.read_csv(DATA_RAW_PATH)

        if df.empty:
            raise ValueError("Dataset is empty.")

        print(f"Data loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
        print(f"Columns preview: {list(df.columns[:5])}...")

        return df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure creditcard.csv is located in data/raw/")
        raise

    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {DATA_RAW_PATH}")
        raise

    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        raise


def save_processed_data(df, filepath):
    """
    Save processed dataset to CSV file.

    Args:
        df (pandas.DataFrame): Processed dataset.
        filepath (str): Output file path.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")

    except Exception as e:
        print(f"Failed to save processed data: {e}")
        raise
