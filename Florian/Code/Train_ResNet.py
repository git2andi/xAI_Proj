import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold
import argparse
from torchvision.transforms import v2
import pickle

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
# Reading Arguments#
####################

parser = argparse.ArgumentParser(description='ResNet Parameter Parser')
parser.add_argument('--balancing', type=bool, default=True,
                    help='Default balancing=False')
parser.add_argument('--data_augmentation', type=bool, default=True,
                    help='Default data_augmentation=False')
parser.add_argument('--split_size', type=float, default=0.8,
                    help='Split size, default=0.8')
parser.add_argument('--visualise', type=bool, default=False,
                    help='Default visualise=True')
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='Learning Rate default=0.0005')
parser.add_argument('--optimiser', type=str, default='Adagrad',
                    help='Type of optimiser to use, default=Adam')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='Maximum Number of Epochs as an integer, default=50')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size, default=64')
parser.add_argument('--patience', type=int, default=15,
                    help='Number of patience epochs, default = 7')
parser.add_argument('--model_name', type=str, default="model_152_leakyRelu_test",
                    help='Model name which will be part of the file names')
parser.add_argument('--sceduler_LearningRate', type=str, default=0.25,
                    help='Learning rate Adjustment, default 0.1')
parser.add_argument('--sceduler_LearningRatePatience', type=str, default=8,
                    help='Learning rate Patience, default 10')

args = parser.parse_args()

#################
# Preprocessing #
#################
balancing = args.balancing
data_augmentation = args.data_augmentation
split_size = args.split_size
visualise = args.visualise

#################
#Hyperparameters#
#################
learning_rate = args.learning_rate
arg_optimiser = args.optimiser
num_epochs = args.num_epochs
batch_size = args.batch_size
patience = args.patience
learningRateAdjustment = args.sceduler_LearningRate
learningRatePatience = args.sceduler_LearningRatePatience

########
#Naming#
########
model_name = args.model_name



#necessary preprocessing operations for ResNet18
preprocessing = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.7333, 0.5314, 0.7012], [0.0864, 0.1119, 0.0868]) #(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# augmentations to be applied
augmentations = v2.Compose([
    v2.RandomHorizontalFlip(0.1),
    v2.RandomVerticalFlip(0.1)
])
augmentations_2 = v2.RandomApply(torch.nn.ModuleList(
    [v2.RandomRotation(30),]), p=0.1)
augmentations_3 = v2.RandomApply(torch.nn.ModuleList(
    [transforms.ColorJitter(),]), p=0.1)


# Custom Dataset Class
class CustomMedMNISTDataset(Dataset):
    def __init__(self, images, labels = None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ]) if transform is None else transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx], mode = 'RGB') # L for grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if data_augmentation:
            image = augmentations(image)
            image = augmentations_2(image)
            image = augmentations_3(image)
        return image, label


# Function for visualising balancedness of original HAM 10000 data
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


# colours for visualising class distribution of data
#if visualise:
#    colours = ["red", "orange", "yellow", "lime", "green", "deepskyblue",
#              "blue", "darkviolet", "magenta"]


###################################
# loading Data into runtime memory#
###################################
with np.load('c:\Work\TrainingData\pathmnist_shuffled_kaggle.npz') as data:
    train_images = data["train_images_shuffled"]
    train_labels = data["train_labels_shuffled"]
    test_images = data["test_images_shuffled"]

# Variable for different classes
different_classes = np.unique(train_labels)

if visualise:
    # gathering information about class distribution of the data
    dif_cl_strings = list(map(lambda x: str(x), different_classes))
    occurences = []
    for c in different_classes:
        occurences.append((train_labels==c).sum())
    visclassdist(dif_cl_strings, occurences, ' initial', '') # , colours

############################
# Defining Hyper-Parameters#
############################
num_classes = len(different_classes)    # number of target classes


########################
# Instantiating dataset#
########################
train_dataset = CustomMedMNISTDataset(train_images, train_labels, transform = preprocessing)
print("Dataset successfully instantiated\n")


# information about data size
len_data = len(train_dataset)
train_set_size = int(split_size*len_data)
val_set_size = len_data - train_set_size


#####################################################
# splitting dataset into training and validation set#
#####################################################
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size])
print("Succesful split into training and validation set\n")

# calculate occurences after split
if (visualise or balancing):
    t_occurences = np.zeros(num_classes)
    for i in train_set.indices:
        t_label = int(train_labels[i])
        t_occurences[t_label] += 1
# visualise class distribution for training split
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


# mean = torch.zeros(3)
# std = torch.zeros(3)

# for images, _ in train_dataloader:
#     batch_samples = images.size(0)  # Batch size (the last batch can have smaller size!)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)

# # Final mean and std
# mean /= len(train_dataloader.dataset)
# std /= len(train_dataloader.dataset)

# print(f'Mean: {mean}')
# print(f'Std: {std}')


# checking the class distribution again; the distribution should be balanced now
if visualise:
    balanced_occurences = np.zeros(9)
    for images, labels in train_dataloader:
        for l in labels:
            balanced_occurences[l] +=1
    # visualise class distribution for training split
    visclassdist(dif_cl_strings, balanced_occurences, ' training',
                 '\npost split post balancing') #, colours


######################
#Loading Model########
######################
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer for 9 classes
model.to(device)

###################
#Model & Optimizer#
###################
criterion = nn.CrossEntropyLoss()
optimiser_dict = {
    'NAdam':optim.NAdam(model.parameters(), lr = learning_rate),
    'Adam':optim.Adam(model.parameters(), lr = learning_rate),
    'SGD': optim.SGD(model.parameters(), lr = learning_rate),
    'Adagrad': optim.Adagrad(model.parameters(), lr = learning_rate)
        }
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optimiser_dict[arg_optimiser]

# Add a scheduler to reduce learning rate when a metric has plateaued
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=learningRateAdjustment, patience=learningRatePatience, verbose=True)


#Variables for Metrics
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
train_precisions = []
train_recalls = []
train_f1_scores = []
val_precisions = []
val_recalls = []
val_f1_scores = []


model.train()

best_val_loss = float('inf')
best_val_accuracy = 0
best_model_state = None
patience_counter = 0
final_epochs = 0
ground_truth_list = []
pred_list = []


print("Beginning Training")

# Training Function
for epoch in range(num_epochs):

    #################
    # Training phase#
    #################
    model.train()
    correct_train = 0
    total_train = 0
    total_train_loss = 0
    num_train_batches = 0
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



    ####################
    # Accuracy and Loss#
    ####################
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)
    avg_train_loss = total_train_loss / num_train_batches
    train_losses.append(avg_train_loss)
    #################################################################
    #Calculating Precision, Recall and Training for the entire epoch#
    #################################################################
    ground_truths = np.concatenate([arr.flatten() for arr in ground_truth_list])
    pred_list = np.concatenate([arr.flatten() for arr in pred_list])
    train_precisions.append(precision_score(ground_truths, y_pred = pred_list,
                    labels = different_classes, average = 'macro'))
    train_recalls.append(recall_score(ground_truths, y_pred = pred_list,
                 labels = different_classes, average = 'macro'))
    train_f1_scores.append(f1_score(ground_truths, y_pred = pred_list,
             labels = different_classes, average = 'macro'))

    #Reset lists so labels don't accumulate across epochs
    ground_truth_list = []
    pred_list = []

    ###################
    # Validation phase#
    ###################
    model.eval()
    correct_val = 0
    total_val = 0
    total_val_loss = 0
    num_val_batches = 0

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
    val_accuracies.append(val_accuracy)
    avg_val_loss = total_val_loss / num_val_batches
    val_losses.append(avg_val_loss)

    #################################
    # Precision, Recall and Accuracy#
    #################################
    ground_truths = np.concatenate([arr.flatten() for arr in ground_truth_list])
    pred_list = np.concatenate([arr.flatten() for arr in pred_list])
    val_precisions.append(
        precision_score(y_true = ground_truths, y_pred = pred_list,
                        labels = different_classes, average = 'macro'))
    val_recalls.append(
        recall_score(y_true = ground_truths, y_pred = pred_list,
                     labels = different_classes, average = 'macro'))
    val_f1_scores.append(
        f1_score(ground_truths, y_pred = pred_list,
                 labels = different_classes, average = 'macro'))



     # Early stopping check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = copy.deepcopy(model.state_dict())  # Save the best model state
        print("Saved new best model!")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        final_epochs = epoch
        # Compute Confusion Matrix
        conf_matrix = confusion_matrix(ground_truths, pred_list)
        #Save Model
        torch.save(best_model_state, model_path + model_name + '.pt')
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    if (epoch+1) == num_epochs:
        torch.save(best_model_state, model_path + model_name + '.pt')
        # Compute Confusion Matrix
        conf_matrix = confusion_matrix(ground_truths, pred_list)

    #Reset lists so labels don't accumulate across epochs
    ground_truth_list = []
    pred_list = []

    #tune learning rate
    scheduler.step(avg_val_loss)

    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')



##########################################
#Saving Model Performance#################
model_dict = {
    'train_accuracies': train_accuracies,
    'train_precision': train_precisions[len(train_precisions)-1],
    'train_recall': train_recalls[len(train_recalls)-1],
    'train_f1_score': train_f1_scores[len(train_f1_scores)-1],
    'train_losses': train_losses,
    'val_accuracies': val_accuracies,
    'val_losses': val_losses,
    'val_precision': val_precisions[len(val_precisions)-1],
    'val_recall': val_recalls[len(val_recalls)-1],
    'val_f1_score': val_f1_scores[len(val_f1_scores)-1],
    'confusion_matrix': conf_matrix,
    'learning_rate': learning_rate,
    'max_epochs': num_epochs,
    'optimizer': arg_optimiser,
    'batch_size': batch_size
    }

# save dictionary to pickle file
with open(model_data_path + model_name + '_dict.pkl', 'wb') as file:
    pickle.dump(model_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
#########################################


#################################
# Save Confusion Matrix as plot #
#################################
disp = ConfusionMatrixDisplay(model_dict['confusion_matrix'],
                    display_labels = different_classes)
disp.plot(cmap = plt.cm.RdYlGn)
plt.savefig(plot_path + model_name + '_conf_matrix.png', dpi=240)
plt.show()
#################################

###############################################################################################
# Plotting the training and validation accuracies
def plot_accuracies(plot_path, train_accuracies=train_accuracies, val_accuracies=val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 2), train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(range(1, epoch + 2), val_accuracies, marker='x', label='Validation Accuracy')
    plt.title("Training vs Validation Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, epoch + 2))
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path + model_name + '_accuracies.png', dpi=240)
    plt.show()

plot_accuracies(plot_path)
###############################################################################################


##########################################################################################
#Plotting the training and validation losses
def plot_losses(plot_path, train_losses=train_losses, val_losses=val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 2), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, epoch + 2), val_losses, marker='x', label='Validation Loss')
    plt.title("Training vs Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(range(1, epoch + 2))
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path + model_name + '_losses.png',  dpi=240)
    plt.show()

plot_losses(plot_path)
##########################################################################################
