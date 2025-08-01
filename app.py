import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Human Voice Clustering",
    layout="wide",
)

# Sidebar navigation info
st.sidebar.success("Select a page above.")

# App Title
st.title("🔊 Human Voice Clustering & Gender Classification")
st.markdown("---")

# Centered image layout using columns
try:
    image = Image.open("machine_learning_classification.jpg")
    resized_image = image.resize((2000, 1500))  # Set your preferred size

    col1, col2, col3 = st.columns([1, 2, 1])  # Center image
    with col2:
        st.image(resized_image, caption="Machine Learning Classification")
except FileNotFoundError:
    st.warning("📷 Image 'machine_learning_classification.jpg' not found in the project directory.")

# Project Introduction Text
st.markdown("""
Welcome to the **Human Voice Analysis Platform**!
This app helps you explore and model human voice data for **clustering** and **gender classification** using machine learning.

---

### 🚀 What You’ll Find Inside:
- 📊 **EDA** — Visualize distributions, correlations
- 🤖 **Clustering** — KMeans & DBSCAN groups
- 🧠 **Classification** — Predict gender using ML models
- 📁 **Prediction Interface** — Upload CSV and test new voice features
- 👤 **About** — Developer info and contact

Use the sidebar to navigate through the sections. Let's get started! 🎧
""")
