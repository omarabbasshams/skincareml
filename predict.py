import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def read_imagefile(file) -> np.ndarray:
    img = Image.open(BytesIO(file))
    return img

def predict(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_indices = {0: "acne", 1: "redness", 2: "wrinkles", 3: "bags"}
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_indices[predicted_class]
