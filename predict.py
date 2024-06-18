import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/model.h5')

# Function to read and preprocess the image file
def read_imagefile(file) -> np.array:
    img = load_img(file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of the image
def predict_image(img_array) -> str:
    prediction = model.predict(img_array)
    class_indices = {0: 'acne', 1: 'redness', 2: 'wrinkles', 3: 'bags'}
    predicted_class = class_indices[np.argmax(prediction)]
    return predicted_class
