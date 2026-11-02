import streamlit as st
import pandas as pd
import joblib   
import numpy as np

model = joblib.load("best_insurance_model.pkl")
scaler = joblib.load("scaler.pkl")  

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Health Insurance Cost Predictor")


age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])
submit_button = st.button("Predict Insurance Cost") 
if submit_button:
   
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    
    input_data_encoded = pd.get_dummies(
        input_data,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int,
    )

   
    expected_columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 
                        'region_northwest', 'region_southeast', 'region_southwest']
    
    for col in expected_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

   
    input_data_encoded = input_data_encoded[expected_columns]

    
    numeric_columns = ["age", "bmi", "children"]
    input_data_encoded[numeric_columns] = scaler.transform(
        input_data_encoded[numeric_columns]
    )

    
    prediction = model.predict(input_data_encoded)

    st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")  
