# 🎤 Human Voice Classification & Clustering

This project classifies and clusters human voice recordings based on extracted audio features like pitch, spectral centroid, MFCCs, and more.

## 💻 Features

- Gender prediction using Random Forest and SVM
- KMeans clustering for voice grouping
- Streamlit interface for uploading and predicting on new data

## 📂 Files

- `voice_model.py`: Preprocessing, training, and saving models
- `app.py`: Streamlit frontend for user predictions
- `*.pkl`: Saved ML models and scaler
- `sample_input.csv`: Try this file for prediction in the app

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the models:

```bash
python voice_model.py
```

3. Launch Streamlit app:

```bash
streamlit run app.py
```

Upload a `.csv` file containing feature values (one or more rows) to get predictions and clustering labels.

## 🧪 Input Format

Must include all features used during training (e.g., MFCCs, spectral features, pitch).

## 📊 Models Used

- Random Forest
- Support Vector Machine (SVM)
- KMeans Clustering
