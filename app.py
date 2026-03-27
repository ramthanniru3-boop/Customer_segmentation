import streamlit as st
import pickle
import numpy as np

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="Customer Intelligence", layout="wide")

st.title("🚀 AI Customer Intelligence Platform")

# =====================
# LOAD MODELS (SAFE)
# =====================
@st.cache_resource
def load_models():
    try:
        kmeans = pickle.load(open("kmeans.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        cluster_meaning = pickle.load(open("cluster_meaning.pkl", "rb"))
        return kmeans, scaler, cluster_meaning
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

kmeans, scaler, cluster_meaning = load_models()

# STOP if models not loaded
if kmeans is None:
    st.stop()

# =====================
# USER INPUT
# =====================
st.sidebar.header("🎯 Customer Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 50)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Convert gender to numeric
gender_val = 1 if gender == "Male" else 0

# =====================
# PREDICTION
# =====================
input_data = np.array([[gender_val, age, income, score]])

try:
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.subheader("🎯 Prediction Result")
    st.success(f"Customer belongs to Cluster {cluster}")

    # Show meaning if exists
    if cluster_meaning:
        st.info(f"💡 Segment: {cluster_meaning.get(cluster, 'Unknown')}")

except Exception as e:
    st.error(f"Prediction error: {e}")