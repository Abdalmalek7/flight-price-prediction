# src/data_pipeline.py
"""
DATA PIPELINE SCRIPT

This script is responsible for:
- Loading raw dataset
- Cleaning and preprocessing the data
- Feature engineering
- Train/validation split
- Saving processed data and pipeline transformers for later model training and inference
"""

import os
import numpy as np
import pandas as pd
import joblib

# Pipeline and preprocessing utilities
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder, RobustScaler

# Dataset split
from sklearn.model_selection import train_test_split

# For custom transformers
from sklearn.base import BaseEstimator, TransformerMixin

# Path where the preprocessing pipeline will be saved
PIPELINE_PATH = os.path.join("models", "pipeline.pkl")


# ===========================
# CUSTOM Cyclical TRANSFORMER
# ===========================
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X).astype(float).reshape(-1, 1)
        sin = np.sin(2 * np.pi * X / self.period)
        cos = np.cos(2 * np.pi * X / self.period)
        return np.concatenate([sin, cos], axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return ["sin", "cos"]
        return [f"{input_features[0]}_sin", f"{input_features[0]}_cos"]


# Columns to keep after preprocessing
FEATURE_COLUMNS =['Airline', 'Source', 'Destination', 'Duration', 'Total_Stops',
       'status', 'Many_Stops', 'Arrival_hour', 'Dep_period', 'Journey_day',
       'Journey_month', 'Journey_weekday', 'Is_weekend', 'Is_long_flight']


# ===========================
# LOAD RAW DATA
# ===========================
def load_raw_data() -> pd.DataFrame:
    """Loads Zomato dataset from CSV file."""
    df = pd.read_csv('Data_Train.csv')
    return df


# ===========================
# MAIN PREPROCESSING FUNCTION
# ===========================
def preprocess_dataframe_part1(mdf: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning and feature engineering:
    - Handle missing values
    - Normalize text fields
    - Fix address info
    - Build 'combined' feature (cuisines + rest_type)
    - Convert target 'rate' to binary classification
    """

    df= mdf.copy()
    
    df.drop_duplicates(inplace=True,ignore_index=True)
    df=df.dropna()
    
    df['Source']=df['Source'].replace('New Delhi','Delhi')
    df['Destination']=df['Destination'].replace('New Delhi','Delhi')
    df['Total_Stops']=df['Total_Stops'].replace({'non-stop':0, '2 stops':2 , '1 stop':1 ,  '3 stops':3, '4 stops':4})
    df['Duration']
    def duration(x):
        item=str(x).split(' ')
        d=[]
        for i in item:
            if "h" in i:
                d.append(int(i.replace('h',''))*60)
            elif "m" in i:
                d.append(int(i.replace('m','')))
        return np.round(sum(d)/60,2)
    df['Duration']=df['Duration'].apply(duration)

    df['Total_Stops'] = df['Total_Stops'].clip(upper=3)

    def status(x):
        if x in ['Vistara Premium economy' ,'Multiple carriers Premium economy' ]:
            return 'Premium'
        elif x in ['Jet Airways Business']:
            return 'Business'
        return 'economy'
    df['status']=df['Airline'].apply(status)

    df['Airline']=df['Airline'].replace({'Vistara Premium economy':'Vistara' ,'Multiple carriers Premium economy':'Multiple carriers','Jet Airways Business':'Jet Airways'})

    df.drop(index=df[df['Airline']=='Trujet'].index ,inplace=True)

    df.reset_index(inplace=True , drop=True)
    df['Many_Stops'] = (df['Total_Stops'] == 3).astype(int)

    df['Additional_Info'].replace('No info',np.nan).isna().sum()

    df.drop(columns=['Additional_Info'],inplace=True)     

    df['Dep_hour']=pd.to_datetime(df['Dep_Time']).dt.hour
    df['Arrival_hour']=pd.to_datetime(df['Arrival_Time']).dt.hour
    
    def part_of_day(hour):
        if hour < 6:
            return 'Early_Morning'
        elif hour < 12:
            return 'Morning'
        elif hour < 18:
            return 'Afternoon'
        else:
            return 'Night'

    df['Dep_period'] = df['Dep_hour'].apply(part_of_day)

    df['Journey_day'] = pd.to_datetime(df['Date_of_Journey']).dt.day
    df['Journey_month'] = pd.to_datetime(df['Date_of_Journey']).dt.month
    df['Journey_weekday'] = pd.to_datetime(df['Date_of_Journey']).dt.dayofweek
    df['Is_weekend'] = df['Journey_weekday'].apply(lambda x: 1 if x >= 5 else 0)    

    df['Is_long_flight'] = (df['Duration'] > 24).astype(int)

    df.drop(columns=['Date_of_Journey', 'Dep_Time','Arrival_Time','Route'],inplace=True)

    df.drop(columns=['Dep_hour'],inplace=True)


    return df


# ===========================
# CREATE PREPROCESSING PIPELINE
# ===========================
def pipeline_preprocesing():
    """Builds column transformer pipeline for categorical, numeric, boolean, and multi-label features."""
    hour_pipe = Pipeline([
        ('cyc', CyclicalEncoder(period=24))
    ])

    month_pipe = Pipeline([
        ('cyc', CyclicalEncoder(period=12))
    ])

    weekday_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipe = Pipeline([
        ('scaler', RobustScaler())
    ])

    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    bin_pipe = Pipeline([
        ('passthrough', 'passthrough')
    ])

    # Column groups
    # Categorical (Nominal)
    categorical_cols = [
        'Airline',
        'Source',
        'Destination',
        'Dep_period',
        'status'
    ]

    # Binary (0 / 1)
    binary_cols = [
        'Many_Stops',
        'Is_weekend',
        'Is_long_flight'
    ]

    # Numeric (Ordinal / Continuous)
    numeric_cols = [
        'Duration',
        'Total_Stops',
        'Journey_day'
    ]

    # Cyclical
    hour_cols = ['Arrival_hour']
    month_cols = ['Journey_month']
    weekday_cols = ['Journey_weekday']

    # Combine into one column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('hour', hour_pipe, hour_cols),
            ('month', month_pipe, month_cols),
            ('weekday', weekday_pipe, weekday_cols),
            ('num', num_pipe, numeric_cols),
            ('bin', 'passthrough', binary_cols),
            ('cat', cat_pipe, categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor


# ===========================
# RUN FULL DATA PIPELINE
# ===========================
def run_data_pipeline(test_size: float = 0.2, random_state: int = 42):
    """Runs the complete pipeline: load → preprocess → split → save."""

    print("=== Data pipeline started ===")

    # Load dataset
    print("Loading raw data...")
    raw_df = load_raw_data()

    # Save raw copy for debugging
    raw_path = os.path.join("data", "raw", "load_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path}")

    # Preprocess data
    print("Preprocessing data...")
    processed_df = preprocess_dataframe_part1(raw_df)

    # Train-validation split
    feature_cols = FEATURE_COLUMNS
    target_col = "Price"

    X = processed_df[feature_cols]
    y = processed_df[target_col]
    # Build pipeline
    preproc = pipeline_preprocesing()
    # Transform train/val
    X = preproc.fit_transform(X)
    # Save preprocessing pipeline
    joblib.dump(preproc, PIPELINE_PATH)

    # Convert back to DataFrame with readable column names
    feature_names = preproc.get_feature_names_out()
    X= pd.DataFrame(X, columns=feature_names)

    # Save processed datasets
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    X_path = os.path.join(processed_dir, "X.csv")
    y_path = os.path.join(processed_dir, "y.csv")

    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)

    print("Processed data saved successfully.")

    print("=== Data pipeline finished successfully ===")


# Run pipeline when script is executed directly
if __name__ == "__main__":
    run_data_pipeline()
