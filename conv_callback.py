import tensorflow as tf
import numpy as np
import os

class ConvCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, activation_model, save_dir='conv_matrices', batch_interval=1):
        self.val_data = val_data
        self.activation_model = activation_model
        self.save_dir = save_dir
        self.batch_interval = batch_interval
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.batch_interval == 0:
            self.save_conv_matrices(epoch)
    
    def save_conv_matrices(self, epoch):
        x_val, _ = self.val_data
        activations = self.activation_model.predict(x_val[:1])
        
        for layer_name, layer_activation in zip([layer.name for layer in self.activation_model.layers], activations):
            save_path = os.path.join(self.save_dir, f'epoch_{epoch}_{layer_name}.npy')
            np.save(save_path, layer_activation)
            print(f'Saved convolutional matrices for {layer_name} at epoch {epoch}')
