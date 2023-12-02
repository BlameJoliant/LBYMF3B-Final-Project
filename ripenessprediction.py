import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# Load the trained model
model = load_model('model.keras')

# Dimensions of images
img_width, img_height = 150, 150

# Image data path
basedir = "D:/Desktop/Academics/LBYMF3B/data/test/"

# Rescale factor for pixel values
rescale_factor = 1. / 255

# Class labels
ripeness_labels = ['ripe', 'midripe', 'underripe', 'unripe']

# Manually define class indices
class_indices = {label: i for i, label in enumerate(ripeness_labels)}

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x *= rescale_factor
    x = np.expand_dims(x, axis=0)
    return x

def predict(image_path, model, class_indices):
    x = preprocess_image(image_path)

    # Predict ripeness stage
    raw_preds = model.predict(x)

    # Display the predicted probabilities for each class
    for label, prob in zip(class_indices.keys(), raw_preds[0]):
        print(f'{label}: {prob*100:.2f}%')

    # Get the class with the highest probability
    predicted_class_index = np.argmax(raw_preds)
    predicted_class = list(class_indices.keys())[predicted_class_index]

    print('Predicted ripeness:', predicted_class)

    return raw_preds

# MAIN
for i in range(757): # Iterate through all test images
    image_path = os.path.join(basedir, f"{i}.jpg")
    print(f'Test Sample: {i}, Image Path: {image_path}')
    predict(image_path, model, class_indices)
    print()
