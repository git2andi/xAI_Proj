import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=None):
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
            image = self.augment(image)

        return image, label


####################
# Preprocessing #
####################
preprocessing = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #transforms.Normalize(mean=[0.27726755, 0.27726755, 0.27726755], std=[0.37245104, 0.28460753, 0.37250024])
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