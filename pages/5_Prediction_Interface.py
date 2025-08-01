import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prediction Interface", layout="wide")
st.title("🎯 Prediction Interface")

# Paths
model_path = "model/voice_gender_classifier.pkl"
scaler_path = "model/scaler.pkl"
features_path = "model/expected_features.pkl"

# Check files
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.warning("⚠️ Model or scaler not found. Please train the model first using model_training.py.")
    st.stop()

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_features = joblib.load(features_path) if os.path.exists(features_path) else None

# File upload
uploaded_file = st.file_uploader("📂 Upload a CSV file with feature data (no label column)", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Drop label column if present
        if "label" in input_df.columns:
            st.warning("⚠️ 'label' column detected and removed.")
            input_df = input_df.drop("label", axis=1)

        # Column validation
        if expected_features:
            missing = [col for col in expected_features if col not in input_df.columns]
            extra = [col for col in input_df.columns if col not in expected_features]

            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.stop()
            if extra:
                st.warning(f"⚠️ Extra columns ignored: {extra}")
                input_df = input_df[expected_features]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        predictions = model.predict(input_scaled)

        # Decode numeric predictions to gender
        label_map = {0: "male", 1: "female"}
        input_df["Predicted_Label"] = [label_map.get(p, "Unknown") for p in predictions]

        st.success("✅ Prediction complete!")

        # Show toggle
        show_all = st.checkbox("Show all predictions", value=False)
        if show_all:
            st.dataframe(input_df, use_container_width=True)
        else:
            st.dataframe(input_df.head(10), use_container_width=True)
            st.caption(f"Showing top 10 of {len(input_df)} predictions")

        # Prediction distribution
        st.subheader("🔢 Prediction Distribution")
        st.bar_chart(input_df["Predicted_Label"].value_counts())

        # Download
        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions as CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
