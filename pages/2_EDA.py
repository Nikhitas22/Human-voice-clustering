# pages/2_📊_EDA.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Exploratory Data Analysis")

# Load data
df = pd.read_csv("vocal_gender_features_new.csv")

# Gender distribution
st.subheader("🎯 Gender Distribution")
fig1, ax1 = plt.subplots(figsize=(4, 3))
sns.countplot(x='label', data=df, ax=ax1)
ax1.set_title("Gender Count")
st.pyplot(fig1)

# Correlation heatmap
st.subheader("🔗 Feature Correlation")
fig2, ax2 = plt.subplots(figsize=(5, 4))
sns.heatmap(df.corr(), cmap='coolwarm', ax=ax2)
ax2.set_title("Correlation Heatmap")
st.pyplot(fig2)
