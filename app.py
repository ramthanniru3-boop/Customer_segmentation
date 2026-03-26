import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# ===============================
# PATH FIX (IMPORTANT FOR STREAMLIT CLOUD)
# ===============================
BASE_DIR = os.path.dirname(__file__)

def load_file(filename):
    return os.path.join(BASE_DIR, filename)

# ===============================
# LOAD FILES
# ===============================
kmeans = pickle.load(open(load_file("kmeans.pkl"), "rb"))
scaler = pickle.load(open(load_file("scaler.pkl"), "rb"))
cluster_meaning = pickle.load(open(load_file("cluster_meaning.pkl"), "rb"))
recommendations = pickle.load(open(load_file("recommendations.pkl"), "rb"))

df = pd.read_csv(load_file("clustered_data.csv"))

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

# ===============================
# CUSTOM UI STYLE
# ===============================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .block {
        background-color: #1e2a38;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================
st.title("🚀 AI Customer Intelligence Platform")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("📋 Customer Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 25)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 50)
score = st.sidebar.slider("Spending Score", 1, 100, 50)

# Encode gender
gender_val = 1 if gender == "Male" else 0

# ===============================
# PREDICT BUTTON
# ===============================
if st.sidebar.button("Predict"):

    # Prepare input
    input_data = np.array([[gender_val, age, income, score]])
    scaled = scaler.transform(input_data)

    cluster = kmeans.predict(scaled)[0]

    # ===============================
    # PERSONA + VALUE
    # ===============================
    persona = cluster_meaning.get(cluster, "Unknown Segment")
    rec = recommendations.get(cluster, "No recommendation available")

    value = "High Value ⭐" if score > 60 else "Medium Value" if score > 30 else "Low Value"

    # ===============================
    # OUTPUT UI
    # ===============================
    st.markdown("### 🧠 Customer Persona")
    st.markdown(f"<div class='block'>{persona}</div>", unsafe_allow_html=True)

    st.markdown("### 💰 Customer Value")
    st.markdown(f"<div class='block'>{value}</div>", unsafe_allow_html=True)

    st.markdown("### 🎯 Recommendation")
    st.markdown(f"<div class='block'>{rec}</div>", unsafe_allow_html=True)

# ===============================
# VISUALS
# ===============================
st.subheader("📊 Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.image(load_file("elbow.png"), caption="Elbow Method")

with col2:
    st.image(load_file("pca_plot.png"), caption="PCA Visualization")

# ===============================
# DATA PREVIEW
# ===============================
st.subheader("📄 Data Preview")
st.dataframe(df.head())