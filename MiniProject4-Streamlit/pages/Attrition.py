import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

#@st.cache_data
def readData(tab):
    dictInput = {
        'Age':{int, "30"},
        'DistanceFromHome': {float, "8.0"},
        'Education': {int, "1-5"},
        'EnvironmentSatisfaction': {int, "1-5"},
        'JobInvolvement': {int, "1-5"},
        'JobSatisfaction': {int, "1-5"},
        'MonthlyIncome': {int, "10000"},
        'WorkLifeBalance': {int, "1-5"},
        'YearsAtCompany': {int, "4"},
        'BusinessTravel_Non-Travel': {bool, "False"},
        'BusinessTravel_Travel_Frequently': {bool, "True"},
        'BusinessTravel_Travel_Rarely': {bool, "False"},
        'Department_Human Resources': {bool, "True"},
        'Department_Research & Development': {bool, "False"},
        'Department_Sales': {bool, "False"},
        'OverTime': {bool, "True"}
        }
    
    
    