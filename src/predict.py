# ==========================================
# Prediction Module
# ==========================================

import numpy as np
import joblib
from src.preprocess import preprocess_image
from src.feature_extraction import extract_glcm_features

# Load model and scaler
model = joblib.load("model/svm_model.pkl")
scaler = joblib.load("model/scaler.pkl")

class_names = {
    0: "Benign",
    1: "Malignant",
    2: "Normal"
}

def predict_image(image_path):
    """
    Predict class of given image
    """

    gray = preprocess_image(image_path)
    features = extract_glcm_features(gray)

    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    confidence = round(np.max(probabilities) * 100, 2)

    return class_names[prediction], confidence
