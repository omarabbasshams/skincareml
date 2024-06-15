import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('models/model.h5')

def read_imagefile(file) -> Image.Image:
    """Read and open an image file."""
    image = Image.open(io.BytesIO(file))
    return image

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the image to match the model's input requirements."""
    image = image.resize((224, 224))  # Ensure the image size matches your model input
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image: Image.Image) -> int:
    """Predict the class of the image using the trained model."""
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]
