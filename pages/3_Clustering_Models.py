import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Clustering", layout="wide")
st.title("🤖 Clustering - Voice Data")

# Load dataset
df = pd.read_csv("vocal_gender_features_new.csv")
X = df.drop('label', axis=1)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar selection
method = st.sidebar.selectbox("Select Clustering Method", ["KMeans", "DBSCAN"])

if method == "KMeans":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
elif method == "DBSCAN":
    eps = st.sidebar.slider("Epsilon", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)

# Fit model
clusters = model.fit_predict(X_scaled)
df['Cluster'] = clusters

# PCA for 2D visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters

# Scatter plot
st.subheader("📉 PCA Visualization of Clusters")
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set1', data=df_pca, ax=ax)
ax.set_title(f"{method} Clustering Result (2D PCA)", fontsize=14)
st.pyplot(fig)
