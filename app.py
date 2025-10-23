import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request

# Load model & label encoder
model = joblib.load('disease_classifier.pkl')
le = joblib.load('label_encoder.pkl')

# Constants
UPLOAD_FOLDER = 'static/uploads'
IMG_SIZE = 100

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess uploaded image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flatten = img.flatten().reshape(1, -1)

    # Predict
    probs = model.predict_proba(img_flatten)[0]
    prediction = le.inverse_transform([np.argmax(probs)])[0]

    # Prepare data for table
    results = zip(le.classes_, np.round(probs * 100, 2))

    return render_template('index.html', prediction=prediction, filepath=filepath, results=results)

if __name__ == '__main__':
    app.run(debug=True)

