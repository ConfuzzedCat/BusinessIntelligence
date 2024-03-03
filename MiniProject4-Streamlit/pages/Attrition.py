import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle

st.set_page_config(page_title="Operations with attrition model", page_icon="📊")

st.title("Operations with attrition model")
st.sidebar.header("attrition model", divider='rainbow')
st.write(
            """This is to predict the attrition of employees in a company."""
)

dictInput = {
    'Age':{int, "30"},                                      # widget: number input
    'DistanceFromHome': {float, "8.0"},                     # widget: number input
    'Education': {int, "1-5"},                              # widget: slider
    'EnvironmentSatisfaction': {int, "1-5"},                # widget: slider
    'JobInvolvement': {int, "1-5"},                         # widget: slider
    'JobSatisfaction': {int, "1-5"},                        # widget: slider
    'MonthlyIncome': {int, "10000"},                        # widget: number input
    'WorkLifeBalance': {int, "1-5"},                        # widget: slider
    'YearsAtCompany': {int, "4"},                           # widget: number input
    'BusinessTravel_Non-Travel': {bool, "False"},           # widget: Selectbox *1
    'BusinessTravel_Travel_Rarely': {bool, "False"},        # widget: Selectbox *1
    'BusinessTravel_Travel_Frequently': {bool, "True"},     # widget: Selectbox *1
    'Department_Human Resources': {bool, "True"},           # widget: Selectbox *2
    'Department_Research & Development': {bool, "False"},   # widget: Selectbox *2
    'Department_Sales': {bool, "False"},                    # widget: Selectbox *2
    'OverTime': {bool, "True"}                              # widget: Toggle
    }

model_file = './model/attrition_model.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Create inputs
Age_input = st.number_input('Age', min_value=18, max_value=80, value=30, step=1)
DistanceFromHome_input = st.number_input('Distance from home', min_value=1, max_value=100, value=8, step=1)
Education_input = st.slider('Education', min_value=1, max_value=5, value=1, step=1)
EnvironmentSatisfaction_input = st.slider('Environment Satisfaction', min_value=1, max_value=5, value=1, step=1)
JobInvolvement_input = st.slider('Job Involvement', min_value=1, max_value=5, value=1, step=1)
JobSatisfaction_input = st.slider('Job Satisfaction', min_value=1, max_value=5, value=1, step=1)
WorkLifeBalance_input = st.slider('Work Life Balance', min_value=1, max_value=5, value=1, step=1)
MonthlyIncome_input = st.number_input('Monthly Income', min_value=1000, max_value=50000, value=10000, step=100)
YearsAtCompany_input = st.number_input('Years at Company', min_value=1, max_value=40, value=4, step=1)
BusinessTravel_input = st.selectbox('Business Travel', ['Non-Travel', 'Travel Rarely', 'Travel Frequently'])
Department_input = st.selectbox('Department', ['Human Resources', 'Research & Development', 'Sales'])
OverTime_input = st.checkbox('OverTime')

# Create dataframe for model prediction
bt_non_travel = 0
bt_travel_rarely = 0
bt_travel_frequently = 0
match BusinessTravel_input:
    case 'Non-Travel':
        bt_non_travel = 1
    case 'Travel Rarely':
        bt_travel_rarely = 1
    case 'Travel Frequently':
        bt_travel_frequently = 1

department_hr = 0
department_rd = 0
department_sales = 0
match Department_input:
    case 'Human Resources':
        department_hr = 1
    case 'Research & Development':
        department_rd = 1
    case 'Sales':
        department_sales = 1

input_df = pd.DataFrame({
                            'Age': [Age_input],
                            'DistanceFromHome': [DistanceFromHome_input],
                            'Education': [Education_input],
                            'EnvironmentSatisfaction': [EnvironmentSatisfaction_input],
                            'JobInvolvement': [JobInvolvement_input],
                            'JobSatisfaction': [JobSatisfaction_input],
                            'MonthlyIncome': [MonthlyIncome_input],
                            'WorkLifeBalance': [WorkLifeBalance_input],
                            'YearsAtCompany': [YearsAtCompany_input],
                            'BusinessTravel_Non-Travel': [bt_non_travel],
                            'BusinessTravel_Travel_Frequently': [bt_travel_frequently],
                            'BusinessTravel_Travel_Rarely': [bt_travel_rarely],
                            'Department_Human Resources': [department_hr],
                            'Department_Research & Development': [department_rd],
                            'Department_Sales': [department_sales],
                            'OverTime_Yes': [OverTime_input]})

# Predict
if st.button('Predict'):
    prediction = model.predict(input_df)
    y_n = 'Yes' if prediction[0] == 1 else 'No'
    st.subheader(f'Will they leave this job? {y_n}')