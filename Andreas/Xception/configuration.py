import os
import torch
import torch.nn as nn
import numpy as np

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")

        np.random.seed(22)
        torch.manual_seed(22)

        #########################################
        # Paths for saving plots, the model etc.#
        #########################################
        self.plot_path = "./Andreas/plots/"
        self.model_path = "./Andreas/models/"
        self.model_data_path = "./Andreas/model_data/"
        self.submission_path = "./Andreas/submissions/"

        # Ensure all folders exist
        self.ensure_folders_exist([
            self.plot_path,
            self.model_path,
            self.model_data_path,
            self.submission_path
        ])

        ####################
        # Hyperparam #
        ####################
        self.learning_rate = 0.001
        self.opti = "AdamW" # NAdam, Adam, SGD, Adagrad, AdamW, RMSprop 
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 100
        self.batch_size = 32
        self.patience = 7


        ####################
        # ETC #
        ####################
        self.balancing = True
        self.data_augmentation = True
        self.split_size = .8
        self.visualise = False
        self.save_model = True
        self.model_name = "xception_299"
        self.data_path = "C:\\Users\\Andi\\Desktop\\xAI_Proj\\pathmnist_shuffled_kaggle.npz"


    def ensure_folders_exist(self, paths):
            for path in paths:
                os.makedirs(path, exist_ok=True)
                print(f"Ensured folder exists: {path}")