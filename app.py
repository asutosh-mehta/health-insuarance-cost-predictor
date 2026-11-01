import streamlit as st
import pandas as pd
import joblib   
import numpy as np
# Load the trained model and scaler
model = joblib.load("best_insurance_model.pkl")
scaler = joblib.load("scaler.pkl")  


# Injecting custom CSS to change background to a linear gradient
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


age = st.number_input("Age", min_value=0, max_value=100, value=30)
sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker (0: No, 1: Yes)", [0, 1])
region = st.selectbox("Region (0: Northwest, 1: Northeast, 2: Southwest, 3: Southeast)", [0, 1, 2, 3])  

if st.button("Predict"):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    st.success(f"Predicted Insurance Cost: ${prediction[0]:.2f}")
