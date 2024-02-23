from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import numpy as np
from visualize import visclassdist

class CustomDataLoader:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def prepare_loaders(self):
        total_size = len(self.dataset)
        train_size = int(self.config.split_size * total_size)
        val_size = total_size - train_size

        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        # Visualize class distribution before balancing
        if self.config.visualise:
            print("Visualizing class distribution before balancing...")
            self.visualize_class_distribution(self.train_dataset, "pre_balancing")

        if self.config.balancing:
            print("Applying class balancing...")
            sampler = self.get_balanced_sampler(self.train_dataset)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, sampler=sampler)
            if self.config.visualise:
                print("Visualizing class distribution after balancing...")
                self.visualize_post_balancing_distribution("post_balancing")
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)

    def get_balanced_sampler(self, dataset):
        labels = np.array([self.dataset.labels[i] for i in dataset.indices])
        labels = labels.flatten()
        class_counts = np.bincount(labels)
        class_weights = 1. / np.maximum(class_counts, 1)
        weights = class_weights[labels]
        return WeightedRandomSampler(weights, len(weights), replacement=True)


    def visualize_class_distribution(self, dataset, title_suffix):
        labels = np.array([self.dataset.labels[i] for i in dataset.indices])
        unique, counts = np.unique(labels, return_counts=True)
        filename = f"{self.config.model_name}_class_distribution_{title_suffix}.png"
        visclassdist(unique, counts, title_suffix, self.config.plot_path, filename)


    def visualize_post_balancing_distribution(self, title_suffix):
        balanced_occurrences = np.zeros(len(np.unique(self.dataset.labels)))
        for _, labels in self.train_loader:
            for label in labels.numpy():
                balanced_occurrences[label] += 1
        dif_cl_strings = [str(i) for i in range(len(balanced_occurrences))]
        filename = f"{self.config.model_name}_class_distribution_{title_suffix}.png"
        visclassdist(dif_cl_strings, balanced_occurrences, "Post Balancing", self.config.plot_path, filename)

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        return test_loader

