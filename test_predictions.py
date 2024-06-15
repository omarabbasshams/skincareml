import cv2
import numpy as np
from tensorflow.keras.models import load_model
from load_image import load_data
import os

def predict_image(image_path, model_path, label_encoder, image_size=(224, 224)):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

if __name__ == "__main__":
    image_paths = [
        'image.jpeg',
        'image1.jpg',  # Replace with paths to your test images
        'image2.jpg'
    ]
    model_path = 'best_model.keras'
    
    # Load label encoder from load_images.py
    base_path = 'image/'
    _, _, _, le = load_data(base_path)
    
    for image_path in image_paths:
        predicted_label = predict_image(image_path, model_path, le)
        print(f'Predicted skin issue for {image_path}: {predicted_label}')
