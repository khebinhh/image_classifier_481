import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess the image to resize and normalize.
    """
    image = image.resize(target_size)  # Resize to specified target size
    image = np.array(image) / 255.0   # Normalize to [0, 1] range
    return image

def load_breed_mapping(filepath):
    """
    Load the breed-to-index mapping from a JSON file.
    """
    with open(filepath, 'r') as f:
        breed_to_index = json.load(f)
    index_to_breed = {v: k for k, v in breed_to_index.items()}
    return index_to_breed

def predict_top_breeds(model, image_path, index_to_breed, top_k=5):
    """
    Predict the top k breeds of a dog given an image.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        preprocessed_image = preprocess_image(image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = model.predict(preprocessed_image).flatten()
        top_indices = np.argsort(predictions)[-top_k:][::-1]  # Get top k indices
        top_breeds = [index_to_breed[idx] for idx in top_indices]
        top_confidences = [predictions[idx] for idx in top_indices]

        return top_breeds, top_confidences
    except Exception as e:
        print(f"Error predicting breed for {image_path}: {e}")
        return [], []

def display_predictions(image_paths, predictions, confidences, images_per_page=16):
    """
    Display the test images along with their top predicted breeds and confidences, paginated.
    """
    total_images = len(image_paths)
    pages = (total_images // images_per_page) + (1 if total_images % images_per_page else 0)

    for page_num in range(pages):
        start_idx = page_num * images_per_page
        end_idx = min((page_num + 1) * images_per_page, total_images)

        plt.figure(figsize=(12, 12))
        for i, (image_path, top_breeds, top_confidences) in enumerate(zip(image_paths[start_idx:end_idx], predictions[start_idx:end_idx], confidences[start_idx:end_idx])):
            try:
                # Load and display the image
                image = Image.open(image_path)
                plt.subplot(4, 4, i + 1)
                plt.imshow(image)
                plt.axis('off')
                title = "\n".join([f"{breed}: {conf:.2f}" for breed, conf in zip(top_breeds, top_confidences)])
                plt.title(title, fontsize=8)
            except Exception as e:
                print(f"Error displaying image {image_path}: {e}")
        plt.tight_layout()
        plt.show()

def main():
    # Specify paths
    model_path = "saved_model/dog_breed_classifier.keras"
    breed_mapping_path = "breed_mapping.json"
    test_images_dir = "test_images"  # Directory containing test images

    # Load the trained model and breed mapping
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Loading breed-to-index mapping...")
    index_to_breed = load_breed_mapping(breed_mapping_path)

    # Gather test images
    image_paths = [os.path.join(test_images_dir, fname) for fname in os.listdir(test_images_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Validate test images directory
    if not image_paths:
        print(f"No valid images found in the directory: {test_images_dir}")
        return

    # Process test images
    predictions = []
    confidences = []
    for image_path in image_paths:
        top_breeds, top_confidences = predict_top_breeds(model, image_path, index_to_breed)
        predictions.append(top_breeds)
        confidences.append(top_confidences)

    # Display results
    display_predictions(image_paths, predictions, confidences)

    # User input for a custom image
    custom_image_path = input("Enter the path to your image file (or press Enter to skip): ").strip()

    if custom_image_path:
        if os.path.isfile(custom_image_path):
            top_breeds, top_confidences = predict_top_breeds(model, custom_image_path, index_to_breed)
            if top_breeds:
                print("Top 5 Predicted Breeds:")
                for breed, conf in zip(top_breeds, top_confidences):
                    print(f"{breed}: {conf:.2f}")

                # Display the custom image with predictions
                image = Image.open(custom_image_path)
                plt.imshow(image)
                plt.axis('off')
                title = "\n".join([f"{breed}: {conf:.2f}" for breed, conf in zip(top_breeds, top_confidences)])
                plt.title(title, fontsize=10)
                plt.show()
        else:
            print(f"File not found: {custom_image_path}")

if __name__ == "__main__":
    main()
