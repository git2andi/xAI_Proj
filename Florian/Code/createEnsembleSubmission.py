import torch
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import v2
import pandas as pd
import timm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#necessary preprocessing operations for old ResNet18
# preprocessing3 = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

preprocessing = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize(168),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#necessary preprocessing operations for ResNet18
preprocessing2 = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.7333, 0.5314, 0.7012], [0.0864, 0.1119, 0.0868]) #(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
        return image, label

#######################################
# Determine Model architecture ########
#######################################

def load_model(model_path, model_architecture, num_classes):
    model = None
    if model_architecture.startswith('resnet'):
        model = getattr(models, model_architecture)(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_architecture == 'xception':
        model = timm.create_model('legacy_xception', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

    state_dict = torch.load(model_path, map_location=device)

    if model_architecture == 'xception':
        new_state_dict = {}
        for key in state_dict:
            # The Xception model in timm might use 'fc' directly for the final layer
            new_key = key.replace('last_linear', 'fc') if 'last_linear' in key else key
            new_state_dict[new_key] = state_dict[key]
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model

def load_model_with_preprocessing_info(model_path, model_architecture, num_classes, preprocessing_type):
    model = load_model(model_path, model_architecture, num_classes)
    return model, preprocessing_type

# Load your labeled data
# Replace these paths with your actual data file paths
with np.load('c:\Work\TrainingData\pathmnist_shuffled_kaggle.npz') as data:
    labeled_images = data["train_images_shuffled"]
    train_labels = data["train_labels_shuffled"]
    test_images = data["test_images_shuffled"]

# Preprocessing and data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add any normalization or other transforms here.
])


#################################################
# Test Models on subset of training data ########
#################################################

dataset = CustomMedMNISTDataset(labeled_images, train_labels, transform=preprocessing)
dataset2 = CustomMedMNISTDataset(labeled_images, train_labels, transform=preprocessing2)

# Splitting the dataset into training and validation sets
val_size = int(0.2 * len(dataset))  # using 10% of the data for validation
train_size = len(dataset) - val_size
_, val_dataset = random_split(dataset, [train_size, val_size])
_, val_dataset2 = random_split(dataset2, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
val_loader2 = DataLoader(val_dataset2, batch_size=64, shuffle=False)

######################
# model Paths ########
######################
#model_paths = ['./models/model_1_resnet18.pt', './models/model_2_resnet50.pt', './models/model_2.pt', './models/model_101.pt', './models/model_resnet50_80.pt', './models/model_1522.0.pt', './models/model_152_v3_.pt']
models_preprocessing1 = ['./models/model_1_resnet18.pt', './models/model_2_resnet50.pt', './models/model_2.pt', './models/model_101.pt', './models/model_resnet50_80.pt',
                         './models/model_1522.0.pt', './models/model_152_v3_.pt', './models/model_152_v4_.pt', './models/model_152_v5_.pt', './models/model_101_v2_.pt', './models/xception.pth']
num_classes = 9

#################################
# Load Models from paths ########
#################################

local_models = []
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[0], 'resnet18', num_classes, 1))
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[1], 'resnet50', num_classes), 0)
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[2], 'resnet152', num_classes, 0))
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[3], 'resnet101', num_classes, 0))
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[4], 'resnet50', num_classes), 0)
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[5], 'resnet152', num_classes, 0))
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[6], 'resnet152', num_classes, 0))
local_models.append(load_model_with_preprocessing_info(models_preprocessing1[7], 'resnet152', num_classes, 1))
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[8], 'resnet18', num_classes, 1))
#local_models.append(load_model_with_preprocessing_info(models_preprocessing1[9], 'resnet152', num_classes, 1))
local_models.append(load_model_with_preprocessing_info(models_preprocessing1[10], 'xception', num_classes, 0))

# Function to evaluate models on validation data
def evaluate_models(models, loader1, loader2):
    correct = 0
    total = 0
    with torch.no_grad():
        for model, preprocessing_flag in models:
            loader = loader1 if preprocessing_flag == 0 else loader2
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device).squeeze(1)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Evaluate the models
val_accuracy = evaluate_models(local_models, val_loader, val_loader2)
print(f'Validation Accuracy of the ensemble: {val_accuracy:.2f}%')

############################
# Create submission ########
############################

class CustomTestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx], mode='RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Initialize the test dataset with preprocessing
test_dataset = CustomTestDataset(test_images, transform=preprocessing)
test_dataset2 = CustomTestDataset(test_images, transform=preprocessing2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def predict_with_ensemble(models, loader1, loader2):
    model_outputs = []  # To store model outputs

    # Ensure all models are in evaluation mode
    for model, _ in models:
        model.eval()

    # Collect predictions from each model
    for model, preprocessing_type in models:
        loader = loader1 if preprocessing_type == 0 else loader2
        all_outputs = []
        
        for images in loader:  # Assuming loader yields a batch of images
            images = images.to(device)
            
            with torch.no_grad():
                outputs = model(images)
                outputs = outputs.cpu().numpy()  # Convert to numpy array for easier manipulation
                all_outputs.append(outputs)
        
        # Concatenate outputs from all batches
        all_outputs = np.concatenate(all_outputs, axis=0)
        model_outputs.append(all_outputs)
    
    # Average the outputs across all models
    # Convert logits to softmax probabilities for averaging
    avg_outputs = np.mean([softmax(outputs) for outputs in model_outputs], axis=0)
    
    # Convert averaged softmax probabilities to class predictions
    predictions = np.argmax(avg_outputs, axis=1)
    return predictions

# Make predictions
predictions = predict_with_ensemble(local_models, test_loader, test_loader2)

def save_predictions_to_csv(predictions):
    ids = list(range(0, len(predictions)))
    df = pd.DataFrame({'ID': ids, 'CLASS': predictions})
    df.to_csv('./predictions/predictions.csv', index=False)

# Save the predictions
save_predictions_to_csv(predictions)