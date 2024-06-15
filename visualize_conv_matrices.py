import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_conv_matrix(epoch, layer_name, save_dir='conv_matrices'):
    file_path = os.path.join(save_dir, f'epoch_{epoch}_{layer_name}.npy')
    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        return
    
    conv_matrix = np.load(file_path)
    print(f'Visualizing convolutional matrices from {file_path}')
    
    # Assuming conv_matrix shape is (1, height, width, num_filters)
    num_filters = conv_matrix.shape[-1]
    
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 5))
    for i in range(num_filters):
        axes[i].imshow(conv_matrix[0, :, :, i], cmap='viridis')
        axes[i].axis('off')
    
    plt.show()

if __name__ == "__main__":
    epoch = 1  # Replace with the desired epoch number
    layer_name = 'conv1'  # Replace with the desired layer name
    visualize_conv_matrix(epoch, layer_name)
