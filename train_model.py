import tensorflow as tf
import numpy as np
from PIL import Image
import os
import xml.etree.ElementTree as ET
from parse_annotations import load_images_and_annotations, get_image_and_annotation_paths  # Assuming your parsing function is defined here

def preprocess_image(image):
    """
    Preprocess the image to resize and normalize.
    """
    image = image.resize((128, 128))  # Resize to 128x128 pixels
    image = np.array(image) / 255.0   # Normalize to [0, 1] range
    return image

def parse_annotation(annotation_file):
    try:
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        
        filename_element = root.find('filename')
        filename = filename_element.text if filename_element is not None else "Unknown"
        print(f"Filename: {filename}")
        
        size = root.find('size')
        width = int(size.find('width').text) if size is not None and size.find('width') is not None else 0
        height = int(size.find('height').text) if size is not None and size.find('height') is not None else 0
        depth = int(size.find('depth').text) if size is not None and size.find('depth') is not None else 3
        
        objects = []
        for obj in root.findall('object'):
            breed_element = obj.find('name')
            breed = breed_element.text if breed_element is not None else "Unknown"
            print(f"Breed: {breed}")
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) if bbox is not None and bbox.find('xmin') is not None else 0
            ymin = int(bbox.find('ymin').text) if bbox is not None and bbox.find('ymin') is not None else 0
            xmax = int(bbox.find('xmax').text) if bbox is not None and bbox.find('xmax') is not None else 0
            ymax = int(bbox.find('ymax').text) if bbox is not None and bbox.find('ymax') is not None else 0
            objects.append({'breed': breed, 'bbox': (xmin, ymin, xmax, ymax)})
        
        # Debugging prints
        print(f"Parsed annotation for file: {filename}")
        print(f"Image size: {width}x{height}x{depth}")
        print(f"Objects: {objects}")
        
        return {'filename': filename, 'size': (width, height, depth), 'objects': objects}
    except Exception as e:
        print(f"Error parsing annotation: {e}")
        return None


def prepare_data(images_dir, annotations_dir):
    """
    Prepare the dataset by loading images and annotations, and converting them to TensorFlow datasets.
    """
    image_paths, annotation_paths = load_images_and_annotations(images_dir, annotations_dir)
    
    images = []
    labels = []
    
    for img_path, ann_path in zip(image_paths, annotation_paths):
        try:
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue  # Skip if the image file doesn't exist
            if not os.path.exists(ann_path):
                print(f"Annotation not found: {ann_path}")
                continue  # Skip if the annotation file doesn't exist

            # Parse the annotations and get the breed label
            breed = parse_annotation(ann_path)  # Parse the breed from annotation
            if breed is None:
                print(f"Invalid annotation: {ann_path}")
                continue  # Skip if there's an error in annotation parsing
            
            labels.append(breed)
            img = Image.open(img_path)
            img_data = preprocess_image(img)  # Preprocess the image
            images.append(img_data)
        except Exception as e:
            print(f"Error processing image {img_path} and annotation {ann_path}: {e}")
            continue
    
    # Convert labels to categorical values
    unique_breeds = list(set(labels))
    breed_to_index = {breed: index for index, breed in enumerate(unique_breeds)}
    labels = [breed_to_index[breed] for breed in labels]
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, unique_breeds


def build_model(num_classes):
    """
    Build a simple CNN model using TensorFlow/Keras.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),  # Explicitly define the input layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    images_dir = 'D:/DogBreeds/Images'
    annotations_dir = 'D:/DogBreeds/Annotation'
    
    # Get all image and annotation file paths
    image_paths, annotation_paths = get_image_and_annotation_paths(images_dir, annotations_dir)
    
    # Load and process images and annotations in batches
    images, annotations = load_images_and_annotations(image_paths, annotation_paths, batch_size=5)
    
    # Check the results
    print(f"Loaded {len(images)} images and {len(annotations)} annotations.")
    
    # Ensure the images are reshaped correctly
    if len(images) == 0 or len(annotations) == 0:
        print("No data loaded. Please check the dataset and paths.")
        exit()
    
    images = np.reshape(images, (-1, 128, 128, 3))  # Reshape if needed
    
    # Convert annotations to labels
    labels = [ann['objects'][0]['breed'] for ann in annotations]
    unique_breeds = list(set(labels))
    breed_to_index = {breed: index for index, breed in enumerate(unique_breeds)}
    labels = [breed_to_index[breed] for breed in labels]
    labels = np.array(labels)
    
    # Split the dataset into training and validation sets
    split_index = int(0.8 * len(images))
    train_images, val_images = images[:split_index], images[split_index:]
    train_labels, val_labels = labels[:split_index], labels[split_index:]
    
    # Check if training data is available
    if len(train_images) == 0 or len(train_labels) == 0:
        print("No training data available. Please check the dataset.")
        exit()
    
    # Build the model
    model = build_model(num_classes=len(unique_breeds))
    
    # Calculate steps per epoch and validation steps
    batch_size = 32  # Adjust the batch size as per your configuration
    steps_per_epoch = max(1, len(train_images) // batch_size)  # Ensure at least 1 step per epoch
    validation_steps = max(1, len(val_images) // batch_size)  # Ensure at least 1 validation step
    
    # Fit the model
    model.fit(
        train_images, 
        train_labels, 
        epochs=16, 
        steps_per_epoch=steps_per_epoch, 
        validation_data=(val_images, val_labels), 
        validation_steps=validation_steps
    )
    
    # Save the trained model
    model.save("dog_breed_classifier.h5")
    print("Model saved as 'dog_breed_classifier.h5'")
