import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_images_from_folder(folder, image_size=(224, 224)):
    images = []
    labels = []
    print(f"Checking directory: {folder}")
    for issue in os.listdir(folder):
        issue_dir = os.path.join(folder, issue)
        if os.path.isdir(issue_dir):
            for split in ['train', 'validate', 'test']:
                split_dir = os.path.join(issue_dir, split, 'images')
                if not os.path.exists(split_dir):
                    print(f"Split directory does not exist: {split_dir}")
                    continue
                print(f"Checking split directory: {split_dir}")
                for file in os.listdir(split_dir):
                    img_path = os.path.join(split_dir, file)
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, image_size)  # Resize the image
                            images.append(img)
                            labels.append(issue)
                            print(f"Loaded image: {img_path} with label: {issue}")
                        else:
                            print(f"Warning: Unable to read image {img_path}")
    print(f"Loaded {len(images)} images from {folder}")
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images / 255.0  # Normalize images
    le = LabelEncoder()
    if len(labels) == 0:
        raise ValueError("No labels found. Ensure the data directories are structured correctly and contain images.")
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)  # One-hot encode labels
    return images, labels, le

def load_data(base_path, image_size=(224, 224)):
    print(f"Loading training data...")
    train_images, train_labels = load_images_from_folder(base_path, image_size)
    print(f"Loading validation data...")
    val_images, val_labels = load_images_from_folder(base_path, image_size)
    print(f"Loading test data...")
    test_images, test_labels = load_images_from_folder(base_path, image_size)
    
    train_images, train_labels, le = preprocess_data(train_images, train_labels)
    val_images, val_labels, _ = preprocess_data(val_images, val_labels)
    test_images, test_labels, _ = preprocess_data(test_images, test_labels)
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), le

if __name__ == "__main__":
    base_path = 'image/'  # Set this to your base path for images
    image_size = (224, 224)  # Desired image size
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), le = load_data(base_path, image_size)
    print(f'Train images shape: {train_images.shape}')
    print(f'Validation images shape: {val_images.shape}')
    print(f'Test images shape: {test_images.shape}')
    print(f'Number of classes: {train_labels.shape[1]}')
