import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

st.title("🚀 AI Customer Intelligence Platform")

# =====================
# PERSONAS + RECOMMENDATIONS
# =====================
cluster_meaning = {
    0: "💰 High Income - Low Spending (Careful Customers)",
    1: "🎯 Average Income - Average Spending",
    2: "🔥 High Income - High Spending (Premium Customers)",
    3: "🛍 Low Income - High Spending (Impulsive Buyers)",
    4: "📉 Low Income - Low Spending (Budget Customers)"
}

recommendations = {
    0: "Offer discounts & loyalty programs",
    1: "Provide combo offers & engagement campaigns",
    2: "Target with premium/luxury products",
    3: "Promote impulse deals & flash sales",
    4: "Focus on budget-friendly products"
}

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_models():
    try:
        kmeans = pickle.load(open("kmeans.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return kmeans, scaler
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

kmeans, scaler = load_models()

if kmeans is None:
    st.stop()

# =====================
# LOAD DATA
# =====================
try:
    df = pd.read_csv("Mall_Customers.csv")
except:
    df = None

# =====================
# TABS
# =====================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 EDA", "📉 Elbow", "🌳 Dendrogram", "🎯 Prediction"]
)

# =====================
# 📊 EDA
# =====================
with tab1:
    st.subheader("Dataset Overview")

    if df is not None:
        st.write(df.head())
        st.write(df.describe())
    else:
        st.warning("Dataset not found")

# =====================
# 📉 ELBOW
# =====================
with tab2:
    st.subheader("Elbow Method")

    try:
        img = plt.imread("elbow.png")
        st.image(img)
    except:
        st.warning("elbow.png not found")

# =====================
# 🌳 DENDROGRAM
# =====================
with tab3:
    st.subheader("Dendrogram")

    try:
        img = plt.imread("dendrogram.png")
        st.image(img)
    except:
        st.warning("dendrogram.png not found")

# =====================
# 🎯 PREDICTION
# =====================
with tab4:
    st.subheader("Customer Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 30)
    income = st.slider("Annual Income (k$)", 10, 150, 50)
    score = st.slider("Spending Score", 1, 100, 50)

    gender_val = 1 if gender == "Male" else 0

    input_data = np.array([[gender_val, age, income, score]])

    try:
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]

        st.success(f"🎯 Cluster: {cluster}")

        st.info(f"👤 Persona: {cluster_meaning.get(cluster)}")

        st.warning(f"💡 Recommendation: {recommendations.get(cluster)}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# =====================
# 📍 PCA VISUALIZATION
# =====================
st.subheader("📍 PCA Visualization")

try:
    img = plt.imread("pca_plot.png")
    st.image(img)
except:
    st.warning("pca_plot.png not found")