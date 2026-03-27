import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

# =========================
# LOAD MODELS (ONLY FOR VISUALS)
# =========================
@st.cache_resource
def load_models():
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return kmeans, scaler

kmeans, scaler = load_models()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Mall_Customers.csv")

if df["Gender"].dtype == object:
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# =========================
# TITLE
# =========================
st.title("🚀 AI Customer Intelligence Platform")

# =========================
# 🔥 LOGIC-BASED PERSONA (FINAL)
# =========================
def get_persona(income, spending):
    if income >= 70 and spending >= 70:
        return "💰 Premium Customer"
    elif income >= 70 and spending < 40:
        return "🎯 Target Customer"
    elif income < 40 and spending >= 70:
        return "💸 Impulsive Buyer"
    elif income < 40 and spending < 40:
        return "📉 Low Value Customer"
    else:
        return "⚖️ Average Customer"

# =========================
# TABS
# =========================
tabs = st.tabs(["Prediction", "PCA", "Dendrogram", "Elbow", "Insights", "EDA"])

# =========================
# 1️⃣ PREDICTION (LOGIC BASED)
# =========================
with tabs[0]:
    st.header("🎯 Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 70, 25)

    with col2:
        income = st.slider("Annual Income (k$)", 10, 150, 50)
        spending = st.slider("Spending Score", 1, 100, 50)

    if st.button("Predict Customer Type"):
        persona = get_persona(income, spending)

        st.success(f"🧠 Persona: {persona}")

        st.subheader("📊 Why this result?")
        st.write(f"Income Level: {income}")
        st.write(f"Spending Behavior: {spending}")

        st.subheader("📌 Input Summary")
        st.write(f"Gender: {gender}")
        st.write(f"Age: {age}")
        st.write(f"Income: {income}")
        st.write(f"Spending: {spending}")

# =========================
# 2️⃣ PCA (KMEANS USED HERE ONLY)
# =========================
with tabs[1]:
    st.header("📊 PCA Visualization")

    features = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaled = scaler.transform(features)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)

    clusters = kmeans.predict(scaled)

    fig, ax = plt.subplots()
    ax.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
    ax.set_title("Customer Segments (Visualization Only)")
    st.pyplot(fig)

# =========================
# 3️⃣ DENDROGRAM
# =========================
with tabs[2]:
    st.header("🌳 Dendrogram")

    sample_size = st.slider("Sample Size", 20, 100, 50)
    cutoff = st.slider("Cutoff", 1, 20, 10)

    features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    sample = features.sample(sample_size, random_state=42)

    linked = linkage(sample, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, ax=ax)
    ax.axhline(y=cutoff, color='r', linestyle='--')

    st.pyplot(fig)

# =========================
# 4️⃣ ELBOW
# =========================
with tabs[3]:
    st.header("📉 Elbow Method")

    features = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaled = scaler.transform(features)

    inertia = []
    K = range(1, 10)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled)
        inertia.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, inertia, marker='o')
    ax.set_title("Elbow Curve")
    st.pyplot(fig)

# =========================
# 5️⃣ INSIGHTS (NO CLUSTER DEPENDENCE)
# =========================
with tabs[4]:
    st.header("📊 Business Insights")

    st.write("Customer segmentation based on income & spending:")

    st.markdown("""
    - 💰 Premium → High income & high spending  
    - 🎯 Target → High income & low spending  
    - 💸 Impulsive → Low income & high spending  
    - 📉 Low Value → Low income & low spending  
    - ⚖️ Average → Mid range  
    """)

# =========================
# 6️⃣ EDA
# =========================
with tabs[5]:
    st.header("📈 Dataset Overview")

    st.dataframe(df.head())
    st.dataframe(df.describe())

    fig, ax = plt.subplots()
    df["Age"].hist(ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)