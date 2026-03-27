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
# LOAD MODELS
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

# =========================
# TITLE
# =========================
st.title("🚀 Customer Intelligence Platform")

# =========================
# AUTO PERSONA FUNCTION
# =========================
def generate_persona(center):
    income = center[2]
    spending = center[3]

    if income > 70 and spending > 70:
        return "💰 Premium Customer"
    elif income > 70 and spending < 40:
        return "🎯 Target Customer"
    elif income < 40 and spending > 70:
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
# 1️⃣ PREDICTION
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

    gender_val = 1 if gender == "Male" else 0
    input_data = np.array([[gender_val, age, income, spending]])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Cluster"):
        cluster = kmeans.predict(input_scaled)[0]
        center = kmeans.cluster_centers_[cluster]

        persona = generate_persona(center)

        st.success(f"Cluster: {cluster}")
        st.metric("🧠 Persona", persona)

        # Explainability
        st.subheader("📊 Why this cluster?")
        st.write(f"Avg Income: {center[2]:.2f}")
        st.write(f"Avg Spending: {center[3]:.2f}")

# =========================
# 2️⃣ PCA
# =========================
with tabs[1]:
    st.header("📊 PCA Visualization")

    features = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaled = scaler.transform(features)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)

    clusters = kmeans.predict(scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
    ax.set_title("Customer Segments (PCA)")
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
    st.pyplot(fig)

# =========================
# 5️⃣ INSIGHTS (NEW 🔥)
# =========================
with tabs[4]:
    st.header("📊 Cluster Insights")

    centers = kmeans.cluster_centers_

    insights = []
    for i, center in enumerate(centers):
        persona = generate_persona(center)
        insights.append({
            "Cluster": i,
            "Avg Age": round(center[1], 1),
            "Avg Income": round(center[2], 1),
            "Avg Spending": round(center[3], 1),
            "Persona": persona
        })

    insights_df = pd.DataFrame(insights)

    st.dataframe(insights_df)

    # Download option
    csv = insights_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Report", csv, "cluster_insights.csv")

# =========================
# 6️⃣ EDA
# =========================
with tabs[5]:
    st.header("📈 Dataset Overview")

    st.dataframe(df.head())
    st.dataframe(df.describe())

    fig, ax = plt.subplots()
    df["Age"].hist(ax=ax)
    st.pyplot(fig)