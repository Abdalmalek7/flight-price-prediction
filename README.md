# Dummy ML Project – Teaching Structure
# 🍽️ Zomato Restaurant Success Predictor  
Predict Restaurant Success Using Machine Learning

---

## 📖 Overview
This project uses real Zomato restaurant data to build a **machine learning model** that predicts whether a restaurant is likely to **succeed** based on features like:
- Location  
- Rating  
- Price range  
- Online delivery availability  
- Cuisines  
- Votes  
- And more...

The project demonstrates:
- A clean and scalable **project structure**
- A reusable **data preprocessing pipeline**
- Training and saving a machine learning model
- Loading the same pipeline in a **Streamlit web app** to predict success

---

## 🧠 Machine Learning Workflow
1. **Data Cleaning**
   - Handling missing values  
   - Fixing inconsistent text  
   - Encoding categorical features  
   - Normalizing numerical features  

2. **Feature Engineering**
   - Extracting cuisine counts  
   - Binary flags for delivery/table booking  
   - Location grouping  

3. **Model Training**
   - XGBoost / RandomForest / Logistic Regression  
   - Hyperparameter tuning  
   - Saving the trained model + preprocessing pipeline  

4. **Prediction App (Streamlit)**
   - User inputs restaurant features  
   - Pipeline transforms input  
   - Model predicts success score  

---

## 📂 Project Structure


## Steps to run
1. python -m src.data_pipeline
2. python -m src.train
3. streamlit run app/streamlit_app.py
