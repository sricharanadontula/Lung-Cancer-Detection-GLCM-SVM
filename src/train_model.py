# ==========================================
# Train 3-Class SVM Model (Benign, Malignant, Normal)
# Optimized Stable Version
# ==========================================

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from src.preprocess import preprocess_image
from src.feature_extraction import extract_glcm_features

print("Training script started...")

# =========================
# FIXED PATH HANDLING
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test")

label_map = {
    "benign": 0,
    "malignant": 1,
    "normal": 2
}

class_names = ["Benign", "Malignant", "Normal"]

# =========================
# Load Dataset Function
# =========================

def load_dataset(path):
    features = []
    labels = []

    for category in os.listdir(path):
        category_path = os.path.join(path, category)

        if category not in label_map:
            continue

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)

            try:
                gray = preprocess_image(file_path)
                feature = extract_glcm_features(gray)

                features.append(feature)
                labels.append(label_map[category])
            except:
                continue

    return np.array(features), np.array(labels)


# =========================
# Load Train & Test Data
# =========================

print("Loading training data...")
X_train, y_train = load_dataset(TRAIN_PATH)

print("Loading testing data...")
X_test, y_test = load_dataset(TEST_PATH)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Train Stable Linear SVM
# =========================

model = SVC(
    kernel='linear',
    C=1,
    probability=True,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# =========================
# Evaluate Model
# =========================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "model", "confusion_matrix.png"))
plt.show()

# =========================
# Save Model & Scaler
# =========================

MODEL_PATH = os.path.join(BASE_DIR, "model")

joblib.dump(model, os.path.join(MODEL_PATH, "svm_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

with open(os.path.join(MODEL_PATH, "accuracy.txt"), "w") as f:
    f.write(str(round(accuracy * 100, 2)))

print("Model Saved Successfully ✅")
