import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from load_images import load_data
from conv_callback import ConvCallback  # Import the custom callback
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Define the model architecture using Functional API
def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the data
base_path = 'image/'
image_size = (224, 224)
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels), le = load_data(base_path, image_size)

# Save the label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

input_shape = train_images.shape[1:]
num_classes = train_labels.shape[1]

# Create the model
model = create_model(input_shape, num_classes)

# Initialize the model by making a prediction with actual data
_ = model.predict(train_images[:1])  # Ensure model layers are initialized with actual data

# Create activation model to get outputs of convolutional layers
conv_layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = Model(inputs=model.input, outputs=conv_layer_outputs)

# Define callbacks
checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
conv_callback = ConvCallback((val_images, val_labels), activation_model, save_dir='conv_matrices', batch_interval=1)

# Train the model
history = model.fit(
    train_images, train_labels, 
    validation_data=(val_images, val_labels), 
    epochs=50, 
    batch_size=32, 
    callbacks=[checkpoint, early_stopping, conv_callback]
)

# Save the model in H5 format
model.save('models/model.h5')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.2f}')

# Plot training & validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
