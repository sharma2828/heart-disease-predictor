import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ==============================
# PAGE CONFIG
# ==============================
st.sidebar.title("About")
st.sidebar.info("This app predicts heart disease risk using ML.")
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ==============================
# CUSTOM CSS (MODERN UI)
# ==============================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.markdown('<div class="title">❤️ Heart Disease Prediction App</div>', unsafe_allow_html=True)

st.write("Enter patient details below:")

# ==============================
# INPUT CARD
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [
        "typical angina", "atypical angina", "non-anginal", "asymptomatic"
    ])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [True, False])

with col2:
    restecg = st.selectbox("Rest ECG", [
        "normal", "st-t abnormality", "lv hypertrophy"
    ])
    thalch = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [True, False])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("Major Vessels", 0, 4, 0)
    thal = st.selectbox("Thal", [
        "normal", "fixed defect", "reversable defect"
    ])

dataset = st.selectbox("Dataset Source", [
    "Cleveland", "Hungary", "Switzerland", "VA Long Beach"
])

st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# ENCODING
# ==============================
sex = 1 if sex == "Male" else 0

cp_map = {"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3}
cp = cp_map[cp]

fbs = 1 if fbs else 0

restecg_map = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
restecg = restecg_map[restecg]

exang = 1 if exang else 0

slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
slope = slope_map[slope]

thal_map = {"normal": 0, "fixed defect": 1, "reversable defect": 2}
thal = thal_map[thal]

dataset_Hungary = 1 if dataset == "Hungary" else 0
dataset_Switzerland = 1 if dataset == "Switzerland" else 0
dataset_VA = 1 if dataset == "VA Long Beach" else 0

# ==============================
# INPUT ARRAY
# ==============================
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalch, exang, oldpeak, slope, ca, thal,
                        dataset_Hungary, dataset_Switzerland, dataset_VA]])

input_data = scaler.transform(input_data)

# ==============================
# PREDICTION BUTTON
# ==============================
if st.button("🔍 Predict Risk"):

    prob = model.predict_proba(input_data)[0][1]

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Result:")

    if prob > 0.5:
        st.error(f"⚠️ High Risk\n\nRisk Score: {prob*100:.1f}%")
    else:
        st.success(f"✅ Low Risk\n\nRisk Score: {prob*100:.1f}%")

    st.progress(int(prob * 100))

    st.markdown('</div>', unsafe_allow_html=True)

st.caption("⚠️ This is a prediction model, not medical advice.")