import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

st.title("🚀 AI Customer Intelligence Platform")

# ================================
import os

@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)

    kmeans = pickle.load(open(os.path.join(base_path, "kmeans.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(base_path, "scaler.pkl"), "rb"))
    cluster_meaning = pickle.load(open(os.path.join(base_path, "cluster_meaning.pkl"), "rb"))

    return kmeans, scaler, cluster_meaning
# ================================
# SIDEBAR INPUT
# ================================
st.sidebar.header("🎯 Customer Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 50)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Encode gender
gender_val = 1 if gender == "Male" else 0

# ================================
# PREDICTION
# ================================
input_data = np.array([[gender_val, age, income, score]])
input_scaled = scaler.transform(input_data)

cluster = kmeans.predict(input_scaled)[0]
cluster_name = cluster_meaning.get(cluster, "Unknown")

# ================================
# OUTPUT
# ================================
st.subheader("🎯 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("Cluster", cluster)

with col2:
    st.metric("Segment", cluster_name)

# ================================
# VISUALS
# ================================
st.subheader("📊 Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.image("elbow.png", caption="Elbow Method")

with col2:
    st.image("pca_plot.png", caption="PCA Visualization")

# ================================
# DATA PREVIEW
# ================================
st.subheader("📂 Dataset Preview")

try:
    df = pd.read_csv("Mall_Customers.csv")
    st.dataframe(df.head())
except:
    st.warning("Dataset not found")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("💡 Built with Streamlit | ML Model: KMeans + PCA")