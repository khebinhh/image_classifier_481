from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import sys

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load the image passed as a command-line argument
img_path = sys.argv[1]

# Load the image with the target size of 224x224 (expected input size for VGG16)
img = image.load_img(img_path, target_size=(224, 224))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand dimensions to match the batch size (1, 224, 224, 3)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image for the VGG16 model
img_array = preprocess_input(img_array)

# Perform prediction
predictions = model.predict(img_array)

# Decode the predictions
decoded_predictions = decode_predictions(predictions, top=5)[0]

# Print the top 5 predictions with their confidence scores
print("Top 5 Predicted Dog Breeds:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score * 100:.2f}%)")
