import torch
import torch.nn as nn
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
        self.num_classes = num_classes
        self.model = self.load_model()
        self.criterion = config.criterion
        self.optimizer = self.configure_optimizer()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
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
        ssl._create_default_https_context = ssl._create_unverified_context # Reset context to allow download (for pretrained Xception)
        model = pretrainedmodels.__dict__["xception"](pretrained="imagenet")
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, self.num_classes)
        
        #model = timm.create_model('resnet152', pretrained=True)
        #num_ftrs = model.fc.in_features
        #model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        model.to(self.device)
        return model

    #def load_model(self):
        
        # Xception 
        #xception_model = pretrainedmodels.__dict__["xception"](pretrained="imagenet")
        #num_ftrs_xception = xception_model.last_linear.in_features
        #xception_model.last_linear = nn.Linear(num_ftrs_xception, self.num_classes)
        #print("Xception model successfully initialized")

        # ResNet-50
        #resnet_model = timm.create_model('resnet50', pretrained=True)
        #num_ftrs_resnet = resnet_model.fc.in_features
        #resnet_model.fc = nn.Linear(num_ftrs_resnet, self.num_classes)
        #print("ResNet-50 model successfully initialized")

        #class EnsembleModel(nn.Module):
            #def __init__(self, xception_model, resnet_model):
                #super(EnsembleModel, self).__init__()
                #self.xception_model = xception_model
                #self.resnet_model = resnet_model

            #def forward(self, x):
                #xception_output = self.xception_model(x)
                #resnet_output = self.resnet_model(x)
            
                # Ensemble by averaging the predictions
                #output = (xception_output + resnet_output) / 2.0

                #return output

        #ensemble_model = EnsembleModel(xception_model, resnet_model)
    
        #ensemble_model.to(self.device)
        #print(ensemble_model)
        #print("Ensemble model successfully initialized.")

    
        #return ensemble_model

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

        # Early Stopping
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = copy.deepcopy(self.model.state_dict())  # Save best model state
            print("Saved new best model!")
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.config.patience:
            val_confusion_matrix = confusion_matrix(
                np.concatenate([arr.flatten() for arr in ground_truth_list]),
                np.concatenate([arr.flatten() for arr in pred_list])
            )
            print("Patience is reached: created CM")
            print(val_confusion_matrix)
            matrix_file_path = os.path.join(self.config.model_data_path, "ensembleCM")
            np.savetxt(matrix_file_path, val_confusion_matrix, fmt='%d')
            print(f"Confusion matrix saved as {matrix_file_path}")


        if epoch == (self.config.num_epochs - 1):
            conf_matrix = confusion_matrix(
                np.concatenate([arr.flatten() for arr in ground_truth_list]),
                np.concatenate([arr.flatten() for arr in pred_list])
            )
            print("Max Epochs reached: created CM")
            print(conf_matrix)
            matrix_file_path = os.path.join(self.config.model_data_path, self.config.confusion_matrix_name)
            np.savetxt(matrix_file_path, conf_matrix, fmt='%d')
            print(f"Confusion matrix saved as {matrix_file_path}")

        # Reset lists so labels don't accumulate across epochs
        ground_truth_list = []
        pred_list = []

        return val_accuracy, val_loss, metrics
    

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
        submission_file_path = os.path.join(self.config.submission_path, 'xception_2_submission.csv')
        submission = pd.DataFrame({'ID': range(len(test_predictions)), 'CLASS': test_predictions})
        submission.to_csv(submission_file_path, index=False)
        print("Submission file 'submission_2_submission.csv' created successfully.")