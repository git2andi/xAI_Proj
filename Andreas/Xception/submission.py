import os
import torch
import csv
import numpy as np
from configuration import Config
from custom_dataset import CustomDataset, preprocessing, data_augmentation
from model_trainer import ModelTrainer
from dataloader import CustomDataLoader

def load_data(data_path):
    """Loads the dataset from the specified NPZ file."""
    with np.load(data_path, allow_pickle=True) as data:
        train_images = data['train_images_shuffled']
        train_labels = data['train_labels_shuffled']
        test_images = data['test_images_shuffled']
    return train_images, train_labels, test_images

if __name__ == "__main__":
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")
    train_images, train_labels, test_images = load_data(config.data_path)
    
    
    dataset = CustomDataset(train_images, train_labels, transform=preprocessing, augment=data_augmentation)
    test_dataset = CustomDataset(test_images, labels=np.zeros(len(test_images)), transform=preprocessing, augment=False)
    data_loader = CustomDataLoader(dataset, config)
    data_loader.prepare_loaders()
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_dataloader = data_loader.get_test_loader(test_dataset)
    
    num_classes = len(np.unique(train_labels))
    print(f"Number of classes: {num_classes}")


    model_trainer = ModelTrainer(config, device, num_classes)

    print("Generating predictions on the test dataset...")
    model_trainer.generate_predictions(test_dataloader)