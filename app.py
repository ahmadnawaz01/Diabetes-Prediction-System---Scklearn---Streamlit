import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Diabetes Predictor Engine",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_prediction_engine():
    with open('svm_model.pkl', 'rb') as file:
        loaded_function = pickle.load(file)
    return loaded_function

@st.cache_resource
def load_standard_scaler():
    with open('scaler.pkl', 'rb') as file:
        loaded_st = pickle.load(file)
    return loaded_st


model = load_prediction_engine()
scale = load_standard_scaler()

st.title("🩺 Diabetes Predictive System")
st.caption("Driven by an optimized Support Vector Machine (SVM) classification model.")
st.write("Provide the clinical diagnostic metrics below to evaluate patient health risks in real-time.")
st.write("##")

col1, col2 = st.columns(2)

with col1:
    p = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    i = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=79)
    a = st.number_input("Age (Years)", min_value=1, max_value=120, value=33)

with col2:
    g = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200, value=120)
    b = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=32.0, format="%.1f")
    d = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=3.000, value=0.471, format="%.3f")

st.write("##")

if st.button("Run Diagnostic Test", use_container_width=True):

    data = (p, g, i, b, d,a)
    np_data = np.array(data).reshape(1, -1)
    np_data = scale.transform(np_data)
    print(f"Input Data: {data}")
    with st.spinner("Analyzing diagnostic markers..."):
        res = model.predict(np_data)
    st.write("---")

    if res[0]==0:
        st.success(f"### ✅ Patient Within Healthy Range")
        st.write("The analytical engine indicates the patient parameters fall within standard health limits.")
    else:
        st.error(f"### ⚠️ High Risk Detected")
        st.write(f"Further clinical evaluation and comprehensive medical testing are advised.")