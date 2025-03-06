from flask import Flask, request, jsonify, render_template, redirect
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load the model safely
model_path = 'trained_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError("trained_model.h5 not found! Make sure it's uploaded to the server.")

model2 = load_model(model_path)

# Class labels
CLASS_LABELS = ['Basalt', 'Conglomerate', 'Dolostone', 'Gabbro', 'Gneiss', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Rhyolite', 'Sandstone', 'Shale', 'Slate']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))  
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        try:
            processed_image = preprocess_image(image_path)
            predictions = model2.predict(processed_image)
            predicted_class = CLASS_LABELS[np.argmax(predictions[0])]

            return jsonify({
                "predicted_class": predicted_class,
                "uploaded_image_path": f"/{UPLOAD_FOLDER}/{file.filename}"
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
