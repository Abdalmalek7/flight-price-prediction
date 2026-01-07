# src/train.py
"""
TRAINING PIPELINE

Responsibilities:
- Load PREPROCESSED data from data/processed/
- Build the model (from model.py)
- Train the model
- Evaluate it
- Save trained model to models/success_model.pkl

Run after data_pipeline:
    python -m src.train
"""

import os
import joblib
import pandas as pd
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import build_model
from sklearn.metrics import accuracy_score, classification_report

# Path where the final trained model will be saved
MODEL_PATH = os.path.join("models", "success_model.pkl")


def load_processed_data():
    """
    Load preprocessed train/validation data from the CSV files
    generated in the data pipeline.
    """
    processed_dir = os.path.join("data", "processed")

    # Load feature matrices for train and validation sets
    X = pd.read_csv(os.path.join(processed_dir, "X.csv"))
    y = pd.read_csv(os.path.join(processed_dir, "y.csv"))
    # Return all datasets for training pipeline
    return X, y


def train():
    print("=== Training pipeline started ===")

    # ------------------------------------------------------------
    # 1) Load preprocessed data created by the data pipeline
    # ------------------------------------------------------------
    print("Loading preprocessed data...")
    X, y = load_processed_data()
    y_log = np.log1p(y)
    # ------------------------------------------------------------
    # 2) Build the ML model (architecture defined in model.py)
    # ------------------------------------------------------------
    print("Building model...")
    model = build_model()

    # ------------------------------------------------------------
    # 3) Train model on training data
    # ------------------------------------------------------------
    print("Training model...")
    model.fit(X, y_log)
    # ------------------------------------------------------------
    # 5) Save the trained model to the models/ folder
    # ------------------------------------------------------------
    print(f"Saving model to {MODEL_PATH} ...")
    joblib.dump(model, MODEL_PATH)

    print("=== Training pipeline finished successfully ===")


if __name__ == "__main__":
    # Allows running with:
    # python -m src.train
    train()
