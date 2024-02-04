import numpy as np

def load_data(data_path):
    """Loads the dataset from the specified NPZ file."""
    with np.load(data_path, allow_pickle=True) as data:
        train_images = data['train_images_shuffled']
    return train_images

def calculate_mean_std(images):
    """Calculate the mean and standard deviation for each channel across all images."""
    # Ensure images are in float format and normalize to [0, 1] range for mean/std calculation
    images = images.astype(np.float32) / 255.
    
    # Calculate mean and std
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    
    return mean, std

if __name__ == "__main__":
    data_path = 'C:\\Users\\andi\\Desktop\\xAI_Proj\\pathmnist_shuffled_kaggle.npz'  # Update this path to your dataset location
    train_images = load_data(data_path)
    
    mean, std = calculate_mean_std(train_images)
    
    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std: {std}")
