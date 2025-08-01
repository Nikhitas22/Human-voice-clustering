import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Paths to save model
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "voice_gender_classifier.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")
features_path = os.path.join(model_dir, "expected_features.pkl")

# Load dataset
df = pd.read_csv("vocal_gender_features_new.csv")

# Split features and target
X = df.drop("label", axis=1)
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (just to follow ML practice — we use full train set here)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model (Neural Network)
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=42)
model.fit(X_train, y_train)

# Save model, scaler, and feature names
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(list(X.columns), features_path)

print("✅ Model and scaler saved to 'model/' folder.")
