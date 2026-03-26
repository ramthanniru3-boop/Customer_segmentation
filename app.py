import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODELS
# =========================
kmeans = pickle.load(open("kmeans.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
cluster_meaning = pickle.load(open("cluster_meaning.pkl", "rb"))
recommendations = pickle.load(open("recommendations.pkl", "rb"))

df = pd.read_csv("clustered_data.csv")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

# =========================
# CUSTOM CSS 🎨
# =========================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.title {
    font-size:40px;
    font-weight:700;
    color:#4CAF50;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.card-title {
    font-size:20px;
    font-weight:600;
    color:#03DAC6;
}
.card-content {
    font-size:18px;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="title">🚀 AI Customer Segmentation & Recommendation System</div>', unsafe_allow_html=True)
st.write("")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("🧾 Customer Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 25)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 50)
spending = st.sidebar.slider("Spending Score", 1, 100, 50)

predict = st.sidebar.button("🚀 Predict")

# Encode gender
gender_val = 1 if gender == "Male" else 0

# =========================
# MAIN OUTPUT
# =========================
if predict:

    input_df = pd.DataFrame({
        "Gender": [gender_val],
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending]
    })

    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]

    segment = cluster_meaning[cluster]
    recommendation = recommendations[cluster]

    # Persona
    age_label = "Young" if age < 30 else "Middle Age" if age < 50 else "Senior"
    income_label = "Low Income" if income < 40 else "Medium Income" if income < 70 else "High Income"
    spending_label = "Low Spending" if spending < 40 else "Medium Spending" if spending < 70 else "High Spending"

    persona = f"{age_label} {gender}, {income_label}, {spending_label}"

    # =========================
    # CARDS UI
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">👤 Persona</div>
            <div class="card-content">{persona}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📊 Segment</div>
            <div class="card-content">{segment}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">💡 Recommendation</div>
            <div class="card-content">{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # VISUALS
    # =========================
    st.subheader("📈 Model Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.image("elbow.png", caption="Elbow Method")

    with col2:
        st.image("pca.png", caption="PCA Visualization")

    # =========================
    # DATA PREVIEW
    # =========================
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())