import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pretrainedmodels
from tqdm import tqdm
import copy
import os
import ssl

class ModelTrainer:
    def __init__(self, config, device, num_classes):
        self.config = config
        self.device = device
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.num_classes = num_classes
        self.model = self.load_model()
        self.criterion = config.criterion
        self.optimizer = self.configure_optimizer()
        self.best_val_loss = -float('inf')
        self.best_val_accuracy = -float('inf')
        self.patience_counter = 0
        self.patience = config.patience
        self.best_model_state = None

    def configure_optimizer(self):
        optimizers = {
            'NAdam': optim.NAdam,
            'Adam': optim.Adam,
            'SGD': optim.SGD,
            'AdamW': optim.AdamW,
            'Adagrad': optim.Adagrad,
            'RMSprop': optim.RMSprop,
        }
        opti_func = optimizers.get(self.config.opti)
        if opti_func is None:
            raise ValueError(f"Optimizer '{self.config.opti}' is not supported.")
        return opti_func(self.model.parameters(), lr=self.config.learning_rate)



    def load_model(self):
        ssl._create_default_https_context = ssl._create_unverified_context # Reset context to allow download (for pretrainedmodels)
        model = pretrainedmodels.__dict__["xception"](pretrained="imagenet")
        #model = timm.create_model('xception', pretrained=True)
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, self.num_classes)
        
        #model = SimpleCNN(num_classes = self.num_classes)

        model.to(self.device)
        return model


    def train(self, train_dataloader):
        self.model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        ground_truth_list = []
        pred_list = []

        
        progress_bar = tqdm(train_dataloader, desc='Training', leave=True)
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device).squeeze(1)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1)
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.size(0)

            labels = labels.detach().cpu().numpy()
            predicted_train = predicted_train.cpu().numpy()
            ground_truth_list.append(labels)
            pred_list.append(predicted_train)

            progress_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
            

        train_accuracy = 100 * correct_train / total_train
        train_loss = total_train_loss / len(train_dataloader)
        metrics = self.calculate_metrics(np.concatenate(ground_truth_list), np.concatenate(pred_list))

        return train_accuracy, train_loss, metrics



    def evaluate(self, validate_dataloader, epoch):
        self.model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        patience_counter = 0
        ground_truth_list = []
        pred_list = []

        with torch.no_grad():
            progress_bar = tqdm(validate_dataloader, desc='Evaluating', leave=True)
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device).squeeze(1)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted_val = torch.max(outputs.data, 1)
                correct_val += (predicted_val == labels).sum().item()
                total_val += labels.size(0)

                labels = labels.detach().cpu().numpy()
                predicted_val = predicted_val.cpu().numpy()
                ground_truth_list.append(labels)
                pred_list.append(predicted_val)

                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        val_accuracy = 100 * correct_val / total_val
        val_loss = total_val_loss / len(validate_dataloader)
        metrics = self.calculate_metrics(np.concatenate(ground_truth_list), np.concatenate(pred_list))


        # Check if 99.6%+
        if val_accuracy >= 99.6:
            torch.save(self.best_model_state, os.path.join(self.model_path, f"996_{self.model_name}.pth"))
            print(f"Model saved with validation accuracy of 99.6% or higher.")


        # Early Stopping and Best Model Save
        if val_accuracy > self.best_val_accuracy:
            patience_counter = 0
            self.best_val_accuracy = val_accuracy
            self.best_model_state = copy.deepcopy(self.model.state_dict())  # Save best model state
            torch.save(self.best_model_state, os.path.join(self.model_path, f"best_{self.model_name}.pth"))
            print("New best model saved with accuracy:", val_accuracy)
    
            # Generate confusion matrix
            best_val_confusion_matrix = confusion_matrix(
            np.concatenate([arr.flatten() for arr in ground_truth_list]),
            np.concatenate([arr.flatten() for arr in pred_list])
            )
            best_matrix_file_path = os.path.join(self.config.model_data_path, f"best_{self.model_name}_CM.npy")
            np.save(best_matrix_file_path, best_val_confusion_matrix)
            print(f"Best model confusion matrix saved as {best_matrix_file_path}")
        else:
            patience_counter += 1


        # Set Early Stop flag
        if patience_counter >= self.patience:
            early_stop = True
        else:
            early_stop = False


        # At the end of all epochs, save the final model and its confusion matrix
        if epoch == (self.config.num_epochs - 1):
            torch.save(self.model.state_dict(), os.path.join(self.model_path, f"full_{self.model_name}.pth"))

            # Generate CM
            full_conf_matrix = confusion_matrix(
                np.concatenate([arr.flatten() for arr in ground_truth_list]),
                np.concatenate([arr.flatten() for arr in pred_list])
            )
            full_matrix_file_path = os.path.join(self.config.model_data_path, f"full_{self.model_name}_CM.npy")
            np.save(full_matrix_file_path, full_conf_matrix)
            print(f"Final model and its confusion matrix saved as {full_matrix_file_path}")


        # Reset lists so labels don't accumulate across epochs
        ground_truth_list = []
        pred_list = []

        return val_accuracy, val_loss, metrics, early_stop, patience_counter
    

    def calculate_metrics(self, ground_truths, predictions):
        precision = precision_score(ground_truths, predictions, average='macro', labels=np.unique(predictions))
        recall = recall_score(ground_truths, predictions, average='macro', labels=np.unique(predictions))
        f1 = f1_score(ground_truths, predictions, average='macro', labels=np.unique(predictions))
        return {'precision': precision, 'recall': recall, 'f1_score': f1}


    def save_model_state(self, filename):
        if self.best_model_state is not None:
            torch.save(self.best_model_state, filename)
        else:
            print("No model state to save.")



    def generate_predictions(self, test_dataloader):
        test_predictions = []
        self.model.eval()
        with torch.no_grad():
            for data, _ in test_dataloader:
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_predictions.extend(pred.cpu().numpy().flatten())

        # Create the submission DataFrame
        submission_file_path = os.path.join(self.config.submission_path, self.model_name + "_submission.csv")
        submission = pd.DataFrame({'ID': range(len(test_predictions)), 'CLASS': test_predictions})
        submission.to_csv(submission_file_path, index=False)
        print("Submission file created successfully.")



class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),            
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)
