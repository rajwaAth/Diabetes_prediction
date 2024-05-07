import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.header('Diabetes Prediction')

diabetes_data = pd.read_csv('diabetes_df.csv')

gender = st.selectbox('Select Gender', diabetes_data['gender'].unique())
age = st.slider('Age',1,100)
hypertension = st.selectbox('Select Hypertension', ['Yes', 'No'])
heart_disease = st.selectbox('Select Heart Disease', ['Yes', 'No'])
smoking_history = st.selectbox('Select Smoking History', diabetes_data['smoking_history'].unique())
bmi = st.slider('BMI', 1,100)
hb = st.slider('HbA1c Level', 1,9)
blood_glucose_level = st.slider('Blood Glucose Level', 80,300)


if st.button('Predict'):
    input_data = pd.DataFrame(
        [[gender, age, hypertension, heart_disease, smoking_history, bmi, hb, blood_glucose_level]],
        columns=['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
    
    input_data['gender'].replace(['Male', 'Female'], [1, 0], inplace=True)
    input_data['hypertension'].replace(['Yes', 'No'], [1, 0], inplace=True)
    input_data['heart_disease'].replace(['Yes', 'No'], [1, 0], inplace=True)
    input_data['smoking_history'].replace(['never', 'No Info', 'current', 'former', 'ever', 'not current'], [4,0,1,3,2,5], inplace=True)

    diabetes_prediction = model.predict(input_data)

    if diabetes_prediction[0] == 1:
        st.markdown("That person is predicted to have diabetes.")
    else:
        st.markdown("That person is predicted not to have diabetes.")