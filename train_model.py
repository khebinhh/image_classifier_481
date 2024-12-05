import tensorflow as tf
import numpy as np
from PIL import Image
import os
import xml.etree.ElementTree as ET
from parse_annotations import load_images_and_annotations, get_image_and_annotation_paths
from sklearn.utils import shuffle
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
        
        size = root.find('size')
        width = int(size.find('width').text) if size is not None and size.find('width') is not None else 0
        height = int(size.find('height').text) if size is not None and size.find('height') is not None else 0
        depth = int(size.find('depth').text) if size is not None and size.find('depth') is not None else 3
        
        objects = []
        for obj in root.findall('object'):
            breed_element = obj.find('name')
            breed = breed_element.text if breed_element is not None else "Unknown"
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) if bbox is not None and bbox.find('xmin') is not None else 0
            ymin = int(bbox.find('ymin').text) if bbox is not None and bbox.find('ymin') is not None else 0
            xmax = int(bbox.find('xmax').text) if bbox is not None and bbox.find('xmax') is not None else 0
            ymax = int(bbox.find('ymax').text) if bbox is not None and bbox.find('ymax') is not None else 0
            objects.append({'breed': breed, 'bbox': (xmin, ymin, xmax, ymax)})
        
        return {'filename': filename, 'size': (width, height, depth), 'objects': objects}
    except Exception as e:
        print(f"Error parsing annotation: {e}")
        return None

def save_breed_mapping(breed_to_index, filepath):
    """
    Save the breed-to-index mapping to a JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(breed_to_index, f, indent=4)

def plot_history(history):
    """
    Plot the training and validation accuracy and loss over epochs.
    """
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def prepare_data(images_dir, annotations_dir):
    """
    Prepare the dataset by loading images and annotations, and converting them to TensorFlow datasets.
    """
    image_paths, annotation_paths = get_image_and_annotation_paths(images_dir, annotations_dir)
    
    images = []
    labels = []
    
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([ 
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])
    
    for img_path, ann_path in zip(image_paths, annotation_paths):
        try:
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue  # Skip if the image file doesn't exist
            if not os.path.exists(ann_path):
                print(f"Annotation not found: {ann_path}")
                continue  # Skip if the annotation file doesn't exist

            breed = parse_annotation(ann_path)  # Parse the breed from annotation
            if breed is None:
                print(f"Invalid annotation: {ann_path}")
                continue  # Skip if there's an error in annotation parsing
            
            labels.append(breed)
            img = Image.open(img_path)
            
            # Apply data augmentation here
            img_data = np.array(img)  # Convert to numpy array for data augmentation
            img_data = data_augmentation(tf.convert_to_tensor(img_data, dtype=tf.float32))  # Apply augmentation
            img_data = preprocess_image(img_data)  # Then preprocess (resize and normalize)
            
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
    Build a CNN model using transfer learning with MobileNetV2 and batch normalization.
    """
    base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the pre-trained layers

    model = tf.keras.models.Sequential([ 
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'DogBreeds')
    images_dir = os.path.join(script_dir, 'Images')
    annotations_dir = os.path.join(script_dir, 'Annotation')
    
    image_paths, annotation_paths = get_image_and_annotation_paths(images_dir, annotations_dir)
    images, annotations = load_images_and_annotations(image_paths, annotation_paths, batch_size=5)
    
    if len(images) == 0 or len(annotations) == 0:
        print("No data loaded. Please check the dataset and paths.")
        exit()

    images = np.reshape(images, (-1, 128, 128, 3))
    labels = [ann['objects'][0]['breed'] for ann in annotations]
    unique_breeds = list(set(labels))
    breed_to_index = {breed: index for index, breed in enumerate(unique_breeds)}
    labels = np.array([breed_to_index[breed] for breed in labels])
    
    images, labels = shuffle(images, labels, random_state=42)
    
    # Train-Test Split (80-20 split)
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    model = build_model(num_classes=len(unique_breeds))
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=10, 
        restore_best_weights=True, 
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        train_images, 
        train_labels, 
        epochs=20, 
        batch_size=16, 
        validation_data=(val_images, val_labels), 
        callbacks=[early_stopping]
    )
    
    # Plot training history
    plot_history(history)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(val_images, val_labels)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_acc}')
    
    # Save the trained model
    model_save_path = "saved_model/dog_breed_classifier.keras"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save breed-to-index mapping
    save_breed_mapping(breed_to_index, 'breed_mapping.json')
