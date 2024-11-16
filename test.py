import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load the model
model = load_model('D:/DogBreeds/dog_breed_classifier.h5')

# Print model input shape
print(f"Model input shape: {model.input_shape}")

# Load breed labels from the folders in the "Annotation" directory
breed_labels = [
    "n02113023-Pembroke", "n02100236-German_short-haired_pointer", "n02107908-Appenzeller", 
    "n02104029-kuvasz", "n02102480-Sussex_spaniel", "n02088094-Afghan_hound", 
    "n02093256-Staffordshire_bullterrier", "n02112350-keeshond", "n02096051-Airedale", 
    "n02095889-Sealyham_terrier", "n02100877-Irish_setter", "n02088238-basset", "n02105505-komondor", 
    "n02094258-Norwich_terrier", "n02093428-American_Staffordshire_terrier", "n02105056-groenendael", 
    "n02111500-Great_Pyrenees", "n02090721-Irish_wolfhound", "n02091831-Saluki", "n02095314-wire-haired_fox_terrier", 
    "n02096294-Australian_terrier", "n02093991-Irish_terrier", "n02097474-Tibetan_terrier", 
    "n02085620-Chihuahua", "n02110806-basenji", "n02099849-Chesapeake_Bay_retriever", 
    "n02111129-Leonberg", "n02092339-Weimaraner", "n02110185-Siberian_husky", 
    "n02106550-Rottweiler", "n02108551-Tibetan_mastiff", 
    "n02088632-bluetick", "n02099712-Labrador_retriever", "n02113978-Mexican_hairless", 
    "n02105641-Old_English_sheepdog", "n02100583-vizsla", "n02108000-EntleBucher", 
    "n02109961-Eskimo_dog", "n02116738-African_hunting_dog", "n02086910-papillon", 
    "n02111889-Samoyed", "n02101006-Gordon_setter", "n02099429-curly-coated_retriever", "n02088466-bloodhound", 
    "n02110958-pug", "n02096177-cairn", "n02106382-Bouvier_des_Flandres", 
    "n02097298-Scotch_terrier", "n02107142-Doberman", "n02097658-silky_terrier", 
    "n02091032-Italian_greyhound", "n02102973-Irish_water_spaniel", "n02091244-Ibizan_hound", 
    "n02105855-Shetland_sheepdog", "n02109047-Great_Dane", "n02100735-English_setter", 
    "n02096585-Boston_bull", "n02110627-affenpinscher", "n02094114-Norfolk_terrier", 
    "n02111277-Newfoundland", "n02089867-Walker_hound", "n02092002-Scottish_deerhound", 
    "n02097130-giant_schnauzer", "n02091134-whippet", "n02101556-clumber", 
    "n02113186-Cardigan", "n02091467-Norwegian_elkhound", "n02096437-Dandie_Dinmont", 
    "n02107312-miniature_pinscher", "n02099267-flat-coated_retriever", "n02087046-toy_terrier", 
    "n02113624-toy_poodle", "n02093754-Border_terrier", "n02097047-miniature_schnauzer", "n02102318-cocker_spaniel", 
    "n02112137-chow", "n02108422-bull_mastiff", "n02102040-English_springer", "n02102177-Welsh_springer_spaniel", 
    "n02105412-kelpie", "n02101388-Brittany_spaniel", "n02113712-miniature_poodle", 
    "n02091635-otterhound", "n02090622-borzoi", "n02115641-dingo", 
    "n02089973-English_foxhound", "n02112018-Pomeranian", "n02089078-black-and-tan_coonhound", 
    "n02098413-Lhasa", "n02085936-Maltese_dog", "n02113799-standard_poodle", 
    "n02094433-Yorkshire_terrier", "n02098105-soft-coated_wheaten_terrier", "n02090379-redbone", "n02105251-briard", 
    "n02086240-Shih-Tzu", "n02109525-Saint_Bernard", "n02093647-Bedlington_terrier", "n02110063-malamute", 
    "n02107574-Greater_Swiss_Mountain_dog", "n02093859-Kerry_blue_terrier", "n02088364-beagle", "n02099601-golden_retriever", 
    "n02104365-schipperke", "n02085782-Japanese_spaniel", "n02097209-standard_schnauzer", "n02087394-Rhodesian_ridgeback", 
    "n02105162-malinois", "n02086646-Blenheim_spaniel", "n02107683-Bernese_mountain_dog", "n02112706-Brabancon_griffon", 
    "n02095570-Lakeland_terrier", "n02106030-collie", "n02108915-French_bulldog", "n02106166-Border_collie", 
    "n02108089-boxer", "n02115913-dhole", "n02106662-German_shepherd", "n02098286-West_Highland_white_terrier", 
    "n02086079-Pekinese"
]

#print(breed_labels)

# Function to load and process the image
def prepare_image(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    
    # Resize the image to (128, 128), as expected by the model
    img = cv2.resize(img, (128, 128))

    # Convert the image to RGB (OpenCV loads images in BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Expand dimensions to match the input format of the model (batch size, height, width, channels)
    img_array = np.expand_dims(img, axis=0)
    
    # Normalize the image
    img_array = img_array / 255.0

    return img_array

# Get the image path from the command line argument
img_path = 'D:/DogBreeds/' + ' '.join(os.sys.argv[1:])

# Process the image
img_array = prepare_image(img_path)

# Make a prediction
try:
    predictions = model.predict(img_array)
    
    # Sort predictions and get the indices of the top 5 probabilities
    sorted_indices = np.argsort(predictions[0])[::-1][:5]
    
    # Print the top 5 predictions with their probabilities
    print("Top 5 Predictions:")
    for idx in sorted_indices:
        breed = breed_labels[idx]
        probability = predictions[0][idx] * 100
        print(f"{breed}: {probability:.2f}%")
    
    # Get the top prediction
    predicted_class_idx = sorted_indices[0]
    predicted_breed = breed_labels[predicted_class_idx]
    print(f"\nPredicted Dog Breed: {predicted_breed} ({predictions[0][predicted_class_idx]*100:.2f}%)")
except Exception as e:
    print(f"Error processing the image: {e}")
