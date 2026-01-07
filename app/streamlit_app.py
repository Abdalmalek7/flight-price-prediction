# app/streamlit_app.py
"""
STREAMLIT APP

Run from project root:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
from PIL import Image
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import load_trained_model, load_pipeline, predict_from_raw
from src.data_pipeline import *

st.title("Zomato Restaurant Success Predictor")

st.write(
    """
This demo predicts whether a restaurant is likely to **succeed or not** based on its features.

You can enter restaurant details below and see the predicted success score.
"""
)



@st.cache_resource
def get_model():
    return load_trained_model()

def get_pipe():
    return load_pipeline()
model = get_model()
pipe=get_pipe()
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


# ---------------------------------
# Load Model
# ---------------------------------
# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(
    page_title="Flight Fare Prediction System",
    layout="wide"
)

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Feature Explanation", "Price Prediction"]
)

# ---------------------------------
# PAGE 1: Project Overview
# ---------------------------------
if page == "Project Overview":

    st.title("✈️ Flight Fare Prediction System")

    image = Image.open("app/India_Air.jpg")
    st.image(image, use_container_width=True)

    st.markdown("## 📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Rows", "10,680")
    with col2:
        st.metric("Number of Features", "11")

    st.markdown("""
### 🔍 About the Dataset
This dataset contains **historical flight booking data for domestic Indian airlines**.
It focuses on flights operating between major Indian cities such as
Delhi, Mumbai, Bangalore, Chennai, and Kolkata.

The data captures key factors that influence **flight ticket prices in India**, including:
- Airline operator  
- Source and destination cities  
- Journey date  
- Number of stops  
- Flight duration  
- Arrival and departure time characteristics  

### 🎯 Project Objective
To build a **machine learning regression system** that accurately predicts
domestic flight prices in India, helping users:
- Estimate ticket prices before booking  
- Compare different flight options  
- Understand the impact of travel timing and route choices  
- Make more informed travel decisions  

### 🛠 Workflow
- Data Cleaning & Feature Engineering  
- Handling Categorical & Numerical Features  
- Pipeline + ColumnTransformer  
- Model Training & Cross-Validation  
- Performance Evaluation using **RMSE**  
- Deployment using **Streamlit**  

### 💡 Why this project matters?
This project simulates a **real-world airline pricing system** similar to those
used by Indian travel platforms. It demonstrates strong skills in:
- Feature engineering from raw booking data  
- Building end-to-end ML pipelines  
- Preventing data leakage  
- Deploying production-ready ML applications  
""")


# ---------------------------------
# PAGE 2: Feature Explanation
# ---------------------------------
elif page == "Feature Explanation":

    st.title("📘 Feature Description")

    feature_info = {
    "Airline": 
    "The Indian airline operating the flight (e.g., IndiGo, Air India, Jet Airways).",

    "Source City": 
    "The Indian city from which the flight departs.",

    "Destination City": 
    "The Indian city where the flight arrives. Must be different from the source city.",

    "Journey Date": 
    "The date of travel. Used to automatically extract day, month, weekday, and weekend information.",

    "Flight Duration (hours)": 
    "Total duration of the flight in hours, including layovers if any.",

    "Total Stops": 
    "Number of stops during the journey (non-stop, one-stop, multi-stop).",

    "Departure Period": 
    "Time-of-day category of departure (Morning, Afternoon, Evening, or Night).",

    "Arrival Hour": 
    "The hour at which the flight arrives (0–23)."
}


    df_features = pd.DataFrame(
        feature_info.items(),
        columns=["Feature", "Description"]
    )

    st.dataframe(df_features, use_container_width=True)
# ---------------------------------
# PAGE 3: Prediction
# ---------------------------------
elif page == "Price Prediction":

    st.title("💰 Flight Price Prediction")
    st.markdown("### Enter Flight Details")

    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            "Airline",
            ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara', 'Other']
        )

        cities = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bangalore']

        source = st.selectbox(
            "Source City",
            cities
        )

        destination = st.selectbox(
            "Destination City",
            [city for city in cities if city != source]
        )

        total_stops = st.selectbox(
            "Total Stops",
            [0, 1, 2, 3, 4]
        )
        
        dep_period = st.selectbox(
            "Departure Period",
            ["Morning", "Afternoon", "Evening", "Night"]
        )
        

        

    with col2:
        
        # 📅 Journey Date 
        journey_date = st.date_input(
            "Journey Date",
            value=pd.to_datetime("2024-06-15")
        )

        arrival_hour = st.slider(
            "Arrival Hour",
            min_value=0,
            max_value=23,
            value=10
        )

        # ⏱️ Duration
        duration = st.slider(
            "Flight Duration (hours)",
            min_value=0.5,
            max_value=50.0,
            value=2.5,
            step=0.1
        )
        

        

    # ---------------------------------
    # Feature Engineering from Date
    # ---------------------------------
    Journey_day = journey_date.day
    Journey_month = journey_date.month
    Journey_weekday = journey_date.weekday()   # Monday = 0
    Is_weekend = 1 if Journey_weekday >= 5 else 0

    # ⏱️ Long flight flag
    Is_long_flight = 1 if duration > 24 else 0

    if st.button("🎯 Predict Price"):

        input_data = pd.DataFrame([{
            'Airline': airline,
            'Source': source,
            'Destination': destination,
            'Duration': duration,
            'Total_Stops': total_stops,
            'status': 'Available',
            'Many_Stops': 1 if total_stops >= 2 else 0,
            'Arrival_hour': arrival_hour,
            'Dep_period': dep_period,
            'Journey_day': Journey_day,
            'Journey_month': Journey_month,
            'Journey_weekday': Journey_weekday,
            'Is_weekend': Is_weekend,
            'Is_long_flight': Is_long_flight
        }])

        # 🔹 Prediction (log scale)
        log_price = predict_from_raw(model, input_data)

        # 🔹 Inverse log transform
        actual_price = np.expm1(log_price[0])

        st.success(
            f"💰 Estimated Flight Price: **₹ {actual_price:,.0f}**"
        )
