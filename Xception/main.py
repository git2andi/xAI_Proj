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



    # Prepare CSV file to save the results
    results_path = os.path.join(config.model_data_path, "training_results_" + config.model_name + ".csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # header
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1 Score',
                     'Validation Loss', 'Validation Accuracy', 'Validation Precision', 'Validation Recall', 'Validation F1 Score'])



    model_trainer = ModelTrainer(config, device, num_classes)


    # Training loop
    print("Initializing model training...")
    for epoch in range(config.num_epochs):
        train_accuracy, train_loss, train_metrics = model_trainer.train(train_loader)
        val_accuracy, val_loss, val_metrics = model_trainer.evaluate(val_loader, epoch)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Train Metrics: {train_metrics}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Validation Metrics: {val_metrics}")
        
        # Separate metrics for CSV writing
        train_precision = train_metrics['precision']
        train_recall = train_metrics['recall']
        train_f1_score = train_metrics['f1_score']
        
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_f1_score = val_metrics['f1_score']

        # Open the CSV file and append the results
        with open(results_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write data for the current epoch
            writer.writerow([epoch+1, train_loss, train_accuracy, train_precision, train_recall, train_f1_score,
                             val_loss, val_accuracy, val_precision, val_recall, val_f1_score])

    print("Generating predictions on the test dataset...")
    model_trainer.generate_predictions(test_dataloader)