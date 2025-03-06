from flask import Flask, request, render_template, redirect, jsonify, url_for
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from urllib.parse import quote as url_quote
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import tempfile
import PIL

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

model2 = load_model('trained_model.h5')  # Ensure your .h5 file path is correct

# Class labels corresponding to your model's predictions
CLASS_LABELS = ['Basalt', 'Conglomerate', 'Dolostone', 'Gabbro', 'Gneiss', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Rhyolite', 'Sandstone', 'Shale', 'Slate']

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not open or find the image.")
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = 256
    new_height = int(new_width / aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))
    return image

def detect_rock_boundary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_image = image.copy()
    cv2.drawContours(boundary_image, contours, -1, (0, 255, 0), 2)
    return boundary_image, mask

def find_optimal_clusters(data, max_clusters=15):
    distortions = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    diff = np.diff(distortions)
    second_diff = np.diff(diff)
    optimal_clusters = np.argmin(second_diff) + 2
    return optimal_clusters

def analyze_shades(image, mask, num_shades=None):
    rock_area = cv2.bitwise_and(image, image, mask=mask)
    pixel_values = rock_area[mask == 255].reshape(-1, 3)
    pixel_values = np.float32(pixel_values)

    if num_shades is None:
        num_shades = find_optimal_clusters(pixel_values)

    kmeans = KMeans(n_clusters=num_shades, random_state=42)
    kmeans.fit(pixel_values)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = len(labels)
    percentages = {f'Shade {i+1}': (count / total_pixels) * 100 for i, count in zip(unique, counts)}

    return centers, percentages, labels, mask

def plot_shades(image, centers, percentages, labels, mask):
    segmented_image = np.zeros_like(image)
    segmented_image[mask == 255] = centers[labels]
    segmented_image = segmented_image.astype(np.uint8)

    segmented_image_path = os.path.join(STATIC_FOLDER, 'segmented_image.png')
    boundary_image_path = os.path.join(STATIC_FOLDER, 'boundary_image.png')
    bar_chart_path = os.path.join(STATIC_FOLDER, 'bar_chart.png')
    pie_chart_path = os.path.join(STATIC_FOLDER, 'pie_chart.png')

    # Save segmented shades image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Segmented Shades within Rock Boundary")
    plt.savefig(segmented_image_path)
    plt.close()

    # Save rock boundary image
    boundary_image = image.copy()
    boundary_image[mask == 0] = (0, 0, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(boundary_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Rock Boundary with Shades")
    plt.savefig(boundary_image_path)
    plt.close()

    # Save bar chart
    plt.figure(figsize=(6, 6))
    plt.bar(range(len(percentages)), percentages.values(), color=[centers[i] / 255 for i in range(len(percentages))])
    plt.xticks(range(len(percentages)), list(percentages.keys()), rotation=45)
    plt.title("Percentage of Each Shade")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()

    # Save pie chart
    plt.figure(figsize=(6, 6))
    colors = [centers[i] / 255 for i in range(len(percentages))]
    plt.pie(percentages.values(), labels=percentages.keys(), colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title("Shade Percentage Distribution")
    plt.savefig(pie_chart_path)
    plt.close()

    return os.path.basename(segmented_image_path), os.path.basename(boundary_image_path), \
           os.path.basename(bar_chart_path), os.path.basename(pie_chart_path)

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    print(file)
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        try:
            image = load_and_preprocess_image(image_path)
            boundary_image, rock_mask = detect_rock_boundary(image)
            centers, percentages, labels, mask = analyze_shades(image, rock_mask)
            segmented_image_path, boundary_image_path, bar_chart_path, pie_chart_path = plot_shades(
                image, centers, percentages, labels, mask)
            
            processed_image = preprocess_image(image_path)
            predictions = model2.predict(processed_image)
            print(predictions)
           
            predicted_class = CLASS_LABELS[np.argmax(predictions[0])]
           
            # Return paths to images and data for frontend display
            return jsonify({
                "predicted_class": predicted_class,
                "centers": centers.tolist(),  # Convert NumPy arrays to lists for JSON compatibility
                "percentages": percentages,
                "segmented_image": f"static/images/{segmented_image_path}",
                "boundary_image": f"static/images/{boundary_image_path}",
                "bar_chart": f"static/images/{bar_chart_path}",
                "pie_chart": f"static/images/{pie_chart_path}"
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
def preprocess_image(image_path):
    """
    Preprocess the image to match the input format required by the model.
    Modify as needed based on your model input (e.g., size, normalization).
    """
    img = load_img(image_path, target_size=(64, 64))  # Adjust target size if needed
    img_array = img_to_array(img)   # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if __name__ == '__main__':
    app.run(debug=True)