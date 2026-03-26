import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Rename
df.rename(columns={"Genre": "Gender"}, inplace=True)

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Select 5 Features
features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔥 ELBOW METHOD
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.savefig("elbow.png")

# Model
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 🔥 PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Save PCA plot
plt.figure()
plt.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])
plt.title("PCA Visualization")
plt.savefig("pca.png")

# 🔥 Label Logic (5 Features Based)
cluster_meaning = {}

for i in range(5):
    cluster = df[df["Cluster"] == i]

    age = cluster["Age"].mean()
    income = cluster["Annual Income (k$)"].mean()
    spending = cluster["Spending Score (1-100)"].mean()

    age_label = "Young" if age < 30 else "Middle Age" if age < 50 else "Senior"
    income_label = "Low Income" if income < 40 else "Medium Income" if income < 70 else "High Income"
    spending_label = "Low Spending" if spending < 40 else "Medium Spending" if spending < 70 else "High Spending"

    label = f"{age_label}, {income_label}, {spending_label}"

    cluster_meaning[i] = label

# 🔥 Recommendations
recommendations = {
    0: "Offer discounts & coupons",
    1: "Promote premium products",
    2: "Target with loyalty programs",
    3: "Upsell high-value products",
    4: "Retarget with ads"
}

# Save all
pickle.dump(kmeans, open("kmeans.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(cluster_meaning, open("cluster_meaning.pkl", "wb"))
pickle.dump(recommendations, open("recommendations.pkl", "wb"))

df.to_csv("clustered_data.csv", index=False)

print("✅ Training Complete")