# voice_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score

# Load dataset
df = pd.read_csv("vocal_gender_features_new.csv")
df.dropna(inplace=True)

X = df.drop("label", axis=1)
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
joblib.dump(kmeans, "kmeans_model.pkl")
print("KMeans Silhouette Score:", silhouette_score(X_scaled, kmeans.labels_))

# DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan.fit(X_scaled)
joblib.dump(dbscan, "dbscan_model.pkl")
print("DBSCAN Clusters Found:", np.unique(dbscan.labels_))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "rf_model.pkl")
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# SVM Model
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, "svm_model.pkl")
y_pred_svm = svm.predict(X_test)
print("\nSVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
