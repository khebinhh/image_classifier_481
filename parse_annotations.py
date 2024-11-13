import os
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import time

# Function to load and process files in batches
def load_images_and_annotations(image_paths, annotation_paths, batch_size=10):
    images = []
    annotations = []
    skipped_files = 0
    
    total_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size > 0 else 0)
    print(f"Total batches to process: {total_batches}")
    
    start_time = time.time()
    
    for i in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[i:i+batch_size]
        batch_annotation_paths = annotation_paths[i:i+batch_size]
        
        for img_path, ann_path in zip(batch_image_paths, batch_annotation_paths):
            # Process annotation file
            try:
                with open(ann_path, 'r') as ann_file:
                    annotation_data = parse_annotation(ann_file)  # Parse the XML annotation file
                    if annotation_data is None:
                        skipped_files += 1
                        continue
            except Exception as e:
                print(f"Error parsing annotation file {ann_path}: {e}")
                skipped_files += 1
                continue
            
            # Process image file
            try:
                with Image.open(img_path) as img:
                    img_data = preprocess_image(img)  # Preprocess the image (resize, normalize, etc.)
            except Exception as e:
                print(f"Error opening image file {img_path}: {e}")
                skipped_files += 1
                continue
            
            images.append(img_data)
            annotations.append(annotation_data)
        
        # Log progress every batch
        elapsed_time = time.time() - start_time
        processed_batches = (i // batch_size) + 1
        print(f"Processed batch {processed_batches}/{total_batches} - Elapsed time: {elapsed_time:.2f} seconds")
    
    print(f"Skipped {skipped_files} files due to errors.")
    return images, annotations

# Example usage
image_base_path = 'C:/Users/arin/Desktop/DogBreeds/Images'
annotation_base_path = 'C:/Users/arin/Desktop/DogBreeds/Annotation'

# Get all image and annotation file paths
image_paths, annotation_paths = load_images_and_annotations(image_base_path, annotation_base_path)

# Load and process images and annotations in batches
images, annotations = load_images_and_annotations(image_paths, annotation_paths, batch_size=5)

# Check the results
print(f"Loaded {len(images)} images and {len(annotations)} annotations.")

# Example output
# Print out the first image data and its annotations
if images and annotations:
    print(f"First image data: {images[0].shape}")
    print(f"First annotation: {annotations[0]}")

# Helper function to parse annotation XML
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




# Function to preprocess image (resize, normalize)
def preprocess_image(image):
    """
    Preprocess the image to resize, convert grayscale to RGB, and normalize the pixel values.
    """
    image = image.resize((128, 128))  # Resize image to 128x128 pixels
    image_data = np.array(image)      # Convert image to numpy array

    # Check if the image is grayscale (2D), then convert to RGB (3 channels)
    if image_data.ndim == 2:
        image_data = np.stack([image_data] * 3, axis=-1)
    elif image_data.ndim == 3 and image_data.shape[2] == 4:
        # Handle RGBA images (4 channels), convert to RGB by removing the alpha channel
        image_data = image_data[..., :3]

    # Normalize pixel values to [0, 1]
    image_data = image_data / 255.0

    # Ensure image has the shape (128, 128, 3) after processing
    if image_data.shape != (128, 128, 3):
        raise ValueError(f"Unexpected image shape: {image_data.shape}, expected (128, 128, 3)")

    return image_data

# Function to get all image and annotation paths from the directories
def get_image_and_annotation_paths(image_base_path, annotation_base_path):
    image_paths = []
    annotation_paths = []
    
    # Convert base paths to absolute paths
    image_base_path = os.path.abspath(image_base_path)
    annotation_base_path = os.path.abspath(annotation_base_path)
    
    print(f"Image base path: {image_base_path}")
    print(f"Annotation base path: {annotation_base_path}")
    
    for breed_folder in os.listdir(image_base_path):
        breed_folder_path = os.path.join(image_base_path, breed_folder)
        annotation_folder_path = os.path.join(annotation_base_path, breed_folder)
        
        if not os.path.isdir(breed_folder_path) or not os.path.isdir(annotation_folder_path):
            continue
        
        print(f"Processing breed folder: {breed_folder}")
        
        for img_filename in os.listdir(breed_folder_path):
            # Only process image files
            if img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):  
                img_path = os.path.join(breed_folder_path, img_filename)
                ann_filename = img_filename.rsplit('.', 1)[0]  # Remove the extension
                ann_path = os.path.join(annotation_folder_path, ann_filename)
                
                # Debugging: Print out paths being processed
                print(f"Processing image: {img_path}")
                print(f"Looking for annotation file: {ann_path}")
                
                # Check if the annotation file exists
                if os.path.exists(ann_path):
                    image_paths.append(img_path)
                    annotation_paths.append(ann_path)
                else:
                    print(f"Annotation file does not exist for: {img_path}")

    print(f"Found {len(image_paths)} images and {len(annotation_paths)} annotations.")
    return image_paths, annotation_paths

# Example usage
image_base_path = 'C:/Users/arin/Desktop/DogBreeds/Images'
annotation_base_path = 'C:/Users/arin/Desktop/DogBreeds/Annotation'

# Get all image and annotation file paths
image_paths, annotation_paths = get_image_and_annotation_paths(image_base_path, annotation_base_path)

# Load and process images and annotations in batches
images, annotations = load_images_and_annotations(image_paths, annotation_paths, batch_size=5)

# Check the results
print(f"Loaded {len(images)} images and {len(annotations)} annotations.")

# Example output
# Print out the first image data and its annotations
if images and annotations:
    print(f"First image data: {images[0].shape}")
    print(f"First annotation: {annotations[0]}")
