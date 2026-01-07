# src/model.py
"""
Model definition module.
Contains ONLY the model construction logic.
"""

from xgboost import XGBRegressor   # Importing XGBoost classifier for building the model

def build_model():
    """
    Build and return an XGBoost classification model.
    This function centralizes model configuration so it can be reused consistently.
    """

    # Create the XGBoost classifier with tuned hyperparameters
    model = XGBRegressor(
        n_estimators=300,        # Number of trees (boosting rounds)
        learning_rate=0.05,       # Step size shrinkage
        max_depth=6,             # Maximum depth of each tree
        subsample=1.0,            # Fraction of rows used per tree (helps avoid overfitting)
        colsample_bytree=0.8,     # Fraction of columns used per tree
        reg_alpha=0.1,              # L1 regularization term
        reg_lambda=1.5,             # L2 regularization term
        random_state=42,          # Ensures reproducibility
        eval_metric='logloss'     # Loss function used during training
    )

    return model   # Return the configured XGBoost model
