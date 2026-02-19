# ==========================================
# Flask Web App
# ==========================================

from flask import Flask, render_template, request
import os
from src.predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load accuracy
with open("model/accuracy.txt", "r") as f:
    accuracy = f.read()

@app.route('/')
def home():
    return render_template("index.html", accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result, confidence = predict_image(filepath)

    if result == "Normal":
        color = "green"
    elif result == "Benign":
        color = "orange"
    else:
        color = "red"

    return render_template("result.html",
                           result=result,
                           confidence=confidence,
                           image=filepath,
                           color=color)

if __name__ == '__main__':
    app.run(debug=True)

