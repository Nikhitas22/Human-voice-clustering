import streamlit as st

st.set_page_config(layout="wide")

st.title("📌 Project Introduction")
st.markdown("---")

st.markdown("""
## 🎙️ Human Voice Clustering and Classification

This project explores the use of machine learning techniques to analyze **human vocal features** for two primary tasks:

1. **Clustering** similar voice patterns (unsupervised learning)
2. **Classifying gender** based on audio features (supervised learning)

---

### 🔧 Technologies Used:
- **Python**, **Pandas**, **Scikit-learn**
- **Streamlit** for web interface
- **Matplotlib** & **Seaborn** for visualization
- **PIL** for image handling

---

### 🎯 Objectives:
- Perform **EDA** to understand voice features
- Apply **KMeans** and **DBSCAN** for clustering
- Train models like **Random Forest**, **SVM** for classification
- Enable **user-uploaded prediction interface**
""")
