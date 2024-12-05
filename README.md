# Dog Breed Classifier

This project implements a deep learning model for predicting the breed of a dog given an image. The model is based on a Convolutional Neural Network (CNN) trained on a dataset of dog breeds. The project uses TensorFlow for model inference and displays the results using Matplotlib.

## Features

- **Breed Prediction**: The model can predict the top 5 breeds for a given dog image.
- **Image Preprocessing**: Images are resized and normalized before being passed through the model.
- **Batch Processing**: You can process and visualize multiple test images at once with paginated results.
- **Custom Image Prediction**: Allows users to input their own image to get breed predictions.
- **Model Inference**: Uses a pre-trained Keras model to predict the breed of dogs in images.
- **Breed Mapping**: A JSON file that maps breed indices to breed names is used to interpret the model's output.

## Installation

To run this project, follow these steps:

### 1. Set Up a Virtual Environment

It is recommended to use a virtual environment to avoid conflicts with other Python projects. You can create and activate a virtual environment by following these steps:

#### On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate

#### On MacOS/Linux:
python3 -m venv venv
source venv/bin/activate


#### 2. Install Packages
pip install tensorflow numpy pillow matplotlib scikit-learn
or
pip install -r requirements.txt




3. Directory Structure
The directory structure should look like this:

bash
Copy code
/dog-breed-classifier
│
├── saved_model/
│   └── dog_breed_classifier.keras  # Trained Keras model
│── Images/  # Folder of Standford Library Dogs
|── Annotations/  # Contains XML files for each dog
├── breed_mapping.json  # Breed-to-index mapping file
├── test_images/        # Directory containing test images
├── breed_classifier.py  # Main script for running the classifier
├── requirements.txt     # List of required packages
└── README.md            # Project documentation
saved_model/dog_breed_classifier.keras: The pre-trained Keras model for breed classification.
breed_mapping.json: A JSON file containing the breed-to-index mapping.
test_images/: A folder where you place your test images for classification.
breed_classifier.py: The main script to run predictions and display results.
requirements.txt: A file containing the list of dependencies for easy installation.



Usage
1. Run the Classifier for Test Images
The script will automatically load the pre-trained model and breed mappings, process all images in the test_images directory, and display the results in a paginated format. The predictions will show the top 5 predicted breeds along with their confidence scores.

To run the classifier, simply execute:

bash
Copy code
python breed_classifier.py
2. Predict Custom Image
After processing the test images, you will be prompted to input a custom image path. If provided, the model will predict the top 5 dog breeds for that image and display the results.

Example input:

vbnet
Copy code
Enter the path to your image file (or press Enter to skip): path/to/your/dog/image.jpg
3. Display Predictions
The predictions for all images will be displayed using Matplotlib, and each image will be shown with its top 5 predicted breeds and their confidence scores. The display will be paginated if there are many images.

Code Overview
Preprocessing: The preprocess_image function resizes and normalizes images to prepare them for the model.
Prediction: The predict_top_breeds function performs breed prediction by passing preprocessed images through the model and extracting the top 5 breeds based on the model's output.
Visualization: The display_predictions function displays the test images along with their breed predictions, paginated if necessary.
Key Functions in the Script
preprocess_image: This function takes an image and preprocesses it by resizing it to 128x128 pixels and normalizing the pixel values to the range [0, 1].

load_breed_mapping: This function loads the breed-to-index mapping from the breed_mapping.json file and creates an index-to-breed mapping.

predict_top_breeds: This function predicts the top k dog breeds for a given image and returns the breed names along with the confidence scores.

display_predictions: This function visualizes the images and their predicted breeds along with the confidence scores using Matplotlib.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
The model used in this project is trained on a large dataset of dog breeds and requires a TensorFlow-compatible environment to run.
Special thanks to all contributors to the dog breed dataset and TensorFlow.
Future Improvements
Fine-tuning the model with more dog breeds for improved accuracy.
Adding more advanced image augmentation techniques to handle a wider variety of input images.
Deploying the model as an API or a web app for easier access and use.
