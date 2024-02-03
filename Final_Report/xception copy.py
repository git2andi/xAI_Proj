import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold
import argparse
from torchvision.transforms import v2
import pickle
import pretrainedmodels
from pretrainedmodels import xception
import ssl


ssl._create_default_https_context = ssl._create_unverified_context # Reset context to allow download (for pretrained Xception)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


#Setting random seeds
np.random.seed(22)
torch.manual_seed(22)

#########################################
# Paths for saving plots, the model etc.#
#########################################
plot_path = "./plots/"
model_path = "./models/"
model_data_path =   "./model_data/"


####################
# ETC #
####################
balancing = True
data_augmentation = True
split_size = .8
visualise = False

####################
# Hyperparam #
####################
learning_rate = 0.0005
opti = "Adam" # Optimizer
num_epochs = 25
batch_size = 64
patience = 5
model_name = "Xception1"
sceduler_LearningRate = .1
sceduler_LearningRatePatience = 10


####################
# Preprocessing #
####################
preprocessing = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(140),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


####################
# Augmentation #
####################
data_augmentation = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0),
        transforms.RandomAffine(degrees=90),
        transforms.RandomAffine(degrees=180),
        transforms.RandomAffine(degrees=270),
    ]),
    transforms.RandomApply([
        transforms.ColorJitter(),
    ], p=0.1)
])


####################
# Custom Dataset #
####################
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.augment:
            image = data_augmentation(image)

        return image, label


####################
# Visualize #
####################
def visclassdist(dif_cl, occs, desc_s, desc_s_2, colours="deepskyblue"):
    fig, ax = plt.subplots()
    ax.bar(dif_cl, occs, color=colours)
    ax.set_ylim(0, max(occs) + 1000)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Occurences')
    ax.set_title('Class Distribution of the' + desc_s +
                 ' PathMNIST Dataset' + desc_s_2)
    for i, value in enumerate(occs):
        plt.text(i, value + 500, value, ha='center', va='top')

    plt.tight_layout()
    plt.show()


    # printing table of class distribution
    df_dict = {'class_names': dif_cl, 'occurences': occs}
    print(pd.DataFrame(data = df_dict))


###################################
# loading Data into runtime memory#
###################################
data_path = "C:\\Users\\Andi\\Desktop\\xAI_Proj\\pathmnist_shuffled_kaggle.npz"
with np.load(data_path) as data:
    train_images = data["train_images_shuffled"]
    train_labels = data["train_labels_shuffled"]
    test_images = data["test_images_shuffled"]

# Variable for different classes
different_classes = np.unique(train_labels)
num_classes = len(different_classes)    # number of target classes

if visualise:
    # gathering information about class distribution of the data
    dif_cl_strings = list(map(lambda x: str(x), different_classes))
    occurences = []
    for c in different_classes:
        occurences.append((train_labels==c).sum())
    visclassdist(dif_cl_strings, occurences, ' initial', '') # , colours



########################
# Instantiating dataset#
########################
train_dataset = CustomDataset(train_images, train_labels, transform = preprocessing, augment = data_augmentation)
print("Dataset successfully instantiated\n")


#####################################################
# splitting dataset into training and validation set#
#####################################################
len_data = len(train_dataset)
train_set_size = int(split_size*len_data)
val_set_size = len_data - train_set_size

train_set, val_set = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size])
print("Succesful split into training and validation set\n")


t_occurences = [0] * num_classes
if (visualise or balancing):
    for i in train_set.indices:
        t_label = int(train_labels[i])
        t_occurences[t_label] += 1
t_occurences = np.array(t_occurences)

if visualise:
    visclassdist(dif_cl_strings, t_occurences, ' training', ' post split') # , colours


#######################
#Balancing of the data#
#######################
if balancing:
    # Calculate weights for each class
    class_weights = [1 / count for count in t_occurences]

    # Create a list of weights corresponding to each sample in the dataset
    sample_weights = [class_weights[int(train_dataset[label_idx][1])] for label_idx in train_set.indices]

    # Convert the list of weights to a PyTorch tensor
    weights = torch.DoubleTensor(sample_weights)

    # Use WeightedRandomSampler to balance the classes
    sampler = WeightedRandomSampler(weights, len(sample_weights), replacement = True) #len(train_set) target_number * num_classes



####################################
# Loading Data via DataLoader class#
####################################
if balancing:
    train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
else:
    train_dataloader = DataLoader(train_set, batch_size=batch_size)

validate_dataloader = DataLoader(val_set, batch_size=batch_size)
print("Data loaded via DataLoader\n")


# checking the class distribution again; the distribution should be balanced now
if visualise:
    balanced_occurences = np.zeros(9)
    for images, labels in train_dataloader:
        for l in labels:
            balanced_occurences[l] +=1
    # visualise class distribution for training split
    visclassdist(dif_cl_strings, balanced_occurences, ' training',
                 '\npost split post balancing') #, colours


import pretrainedmodels
################
# Loading Model#
################
model = pretrainedmodels.__dict__["xception"](pretrained="imagenet")
num_ftrs = model.last_linear.in_features
model.last_linear = nn.Linear(num_ftrs, num_classes)
model.to(device)


#######################
#Criterion & Optimizer#
#######################
criterion = nn.CrossEntropyLoss()
optimiser_dict = {
    'NAdam':optim.NAdam(model.parameters(), lr = learning_rate),
    'Adam':optim.Adam(model.parameters(), lr = learning_rate),
    'SGD': optim.SGD(model.parameters(), lr = learning_rate),
    'Adagrad': optim.Adagrad(model.parameters(), lr = learning_rate)
        }
optimizer = optimiser_dict[opti]



#Variables for Metrics
train_accuracies, train_losses, train_precisions, train_recalls, train_f1_scores = [], [], [], [], []
val_accuracies, val_losses, val_precisions, val_recalls, val_f1_scores= [], [], [], [], []
ground_truth_list, pred_list = [], []


best_val_loss = float('inf')
best_val_accuracy = 0
best_model_state = None
patience_counter = 0
final_epochs = 0
min_delta = 0.001 # min changes required for improvement

def train(model, train_dataloader, optimizer, criterion):
    model.train()
    correct_train = 0
    total_train = 0
    total_train_loss = 0
    num_train_batches = 0
    ground_truth_list = []
    pred_list = []

    for images, labels in train_dataloader:

        images, labels = images.to(device), labels.to(device).squeeze(1)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        num_train_batches += 1

        # Calculate training accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        labels = labels.detach().cpu().numpy()
        predicted_train = predicted_train.cpu().numpy()

        # For calculating Metrics
        ground_truth_list.append(labels)
        pred_list.append(predicted_train)


    train_accuracy = 100 * correct_train / total_train
    train_loss = total_train_loss / num_train_batches

    ground_truths = np.concatenate([arr.flatten() for arr in ground_truth_list])
    pred_list = np.concatenate([arr.flatten() for arr in pred_list])
    train_precision = precision_score(ground_truths, y_pred = pred_list, labels = different_classes, average = 'macro')
    train_recall = recall_score(ground_truths, y_pred = pred_list, labels = different_classes, average = 'macro')
    train_f1_score = f1_score(ground_truths, y_pred = pred_list, labels = different_classes, average = 'macro')

    return train_accuracy, train_loss, train_precision, train_recall, train_f1_score




def evaluate(model, validate_dataloader, criterion, device, different_classes):
    model.eval()
    correct_val = 0
    total_val = 0
    total_val_loss = 0
    num_val_batches = 0
    ground_truth_list = []
    pred_list = []

    with torch.no_grad():
        for images, labels in validate_dataloader:
            images, labels = images.to(device), labels.to(device).squeeze(1)
            outputs = model(images)
            _, predicted_val = torch.max(outputs.data, 1)

            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss.item()
            num_val_batches += 1

            labels = labels.detach().cpu().numpy()
            predicted_val = predicted_val.cpu().numpy()
            
            # For Confusion Matrix
            ground_truth_list.append(labels)
            pred_list.append(predicted_val)


    val_accuracy = 100 * correct_val / total_val
    val_loss = total_val_loss / num_val_batches


    ground_truths = np.concatenate([arr.flatten() for arr in ground_truth_list])
    pred_list = np.concatenate([arr.flatten() for arr in pred_list])
    val_precision = precision_score(ground_truths, y_pred=pred_list, labels=different_classes, average='macro')
    val_recall = recall_score(ground_truths, y_pred=pred_list, labels=different_classes, average='macro')
    val_f1_score = f1_score(ground_truths, y_pred=pred_list, labels=different_classes, average='macro')

    return val_accuracy, val_loss, val_precision, val_recall, val_f1_score


for epoch in range(num_epochs):
    train_accuracy, train_loss, train_precision, train_recall, train_f1_score = train(model, train_dataloader, optimizer, criterion)
    val_accuracy, val_loss, val_precision, val_recall, val_f1_score = evaluate(model, validate_dataloader, criterion, device, different_classes)

    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1_scores.append(train_f1_score)

    val_accuracies.append(train_accuracy)
    val_losses.append(train_loss)
    val_precisions.append(train_precision)
    val_recalls.append(train_recall)
    val_f1_scores.append(train_f1_score)


    # Check improvement
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        patience_counter = 0  # Reset counter
    else:
        patience_counter += 1  # Increase counter

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, train_precision: {train_precision:.4f}, train_recall: {train_recall:.4f}, train_f1_score: {train_f1_score:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}, val_f1_score: {val_f1_score:.4f}')

    # Early stopping
    if patience_counter >= patience:
        print(f'Early stopping triggered at epoch {epoch+1}')
        break

