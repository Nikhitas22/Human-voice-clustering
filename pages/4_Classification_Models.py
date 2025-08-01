import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Classification", layout="wide")
st.title("🧠 Classification - Gender Prediction")

# Load dataset
try:
    df = pd.read_csv("vocal_gender_features_new.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found. Make sure 'vocal_gender_features_new.csv' is in the root folder.")
    st.stop()

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sidebar: Classifier selection
classifier = st.sidebar.selectbox("Select Classifier", ["Random Forest", "SVM", "Neural Network"])

# Classifier setup
if classifier == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

elif classifier == "SVM":
    C_val = st.sidebar.slider("SVM Regularization (C)", 0.01, 10.0, 1.0)
    model = SVC(C=C_val, kernel="linear", probability=True, random_state=42)

elif classifier == "Neural Network":
    max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 1000)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=max_iter, early_stopping=True, random_state=42)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
st.metric(label="📈 Accuracy", value=f"{accuracy:.2f}")

# Neural Network: show convergence info
if classifier == "Neural Network":
    st.info(f"🔁 Neural network converged in **{model.n_iter_} iterations**")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.subheader("🔍 Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_title(f"{classifier} - Confusion Matrix")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# Classification report
st.subheader(f"📊 {classifier} - Classification Report")
st.text(classification_report(y_test, y_pred))
