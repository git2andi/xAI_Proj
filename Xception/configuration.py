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
        self.plot_path = "./Xception/plots/"
        self.model_path = "./Xception/models/"
        self.model_data_path = "./Xception/model_data/"
        self.submission_path = "./Xception/submissions/"


        ####################
        # Hyperparam #
        ####################
        self.learning_rate = 0.0005
        self.opti = "AdamW" # NAdam, Adam, SGD, Adagrad, AdamW 
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 100
        self.batch_size = 32
        self.patience = 5
        self.sceduler_LearningRate = .1
        self.sceduler_LearningRatePatience = 10


        ####################
        # ETC #
        ####################
        self.balancing = True
        self.data_augmentation = True
        self.split_size = .8
        self.visualise = False
        self.save_model = True
        self.model_name = "xception_2"
        self.confusion_matrix_name = "xception_2.txt"
        self.data_path = "C:\\Users\\Andi\\Desktop\\xAI_Proj\\pathmnist_shuffled_kaggle.npz"

