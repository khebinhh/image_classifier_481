from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ExifTags
import os
import json
import base64
from io import BytesIO

app = Flask(__name__, static_folder='static')

# Load model and mappings
model = tf.keras.models.load_model('saved_model/dog_breed_classifier.keras')
with open('breed_mapping.json', 'r') as f:
    breed_mapping = json.load(f)
index_to_breed = {v: k for k, v in breed_mapping.items()}

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'Orientation':
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Error adjusting orientation: {e}")
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_files = request.files.getlist("files")
    results = []
    
    for file in uploaded_files:
        try:
            image = Image.open(file.stream)
            # Correct orientation
            image = correct_image_orientation(image)
            preprocessed = preprocess_image(image)
            predictions = model.predict(preprocessed)[0]
            
            # Get top 5 predictions
            top_indices = predictions.argsort()[-5:][::-1]
            top_breeds = [
                {"breed": index_to_breed[idx], "probability": float(predictions[idx])}
                for idx in top_indices
            ]

            # Convert image to base64 string
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            results.append({
                'filename': file.filename,
                'top_breeds': top_breeds,
                'image': img_str
            })
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run('0.0.0.0', port=3000)
