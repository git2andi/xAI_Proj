import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df_all = pd.read_csv('C:\\Users\\Andi\\Desktop\\xAI_Proj\\Andreas\\Xception\\model_data\\training_results_SimpleCNN_all.csv').head(30)
df_noDO = pd.read_csv('C:\\Users\\Andi\\Desktop\\xAI_Proj\\Andreas\\Xception\\model_data\\training_results_SimpleCNN_noDO.csv').head(30)
df_noBN = pd.read_csv('C:\\Users\\Andi\\Desktop\\xAI_Proj\\Andreas\\Xception\\model_data\\training_results_SimpleCNN_noBN.csv').head(30)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)

# Loss plot for all configurations
axs[0].plot(df_all.index + 1, df_all['Validation Loss'], label='All Features Validation Loss', marker='o')
axs[0].plot(df_noDO.index + 1, df_noDO['Validation Loss'], label='No Dropout Validation Loss', marker='x')
axs[0].plot(df_noBN.index + 1, df_noBN['Validation Loss'], label='No BatchNorm Validation Loss', marker='^')
axs[0].set_title('Validation Loss Across Configurations')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

# Accuracy plot for all configurations
axs[1].plot(df_all.index + 1, df_all['Validation Accuracy'], label='All Features Validation Accuracy', marker='o')
axs[1].plot(df_noDO.index + 1, df_noDO['Validation Accuracy'], label='No Dropout Validation Accuracy', marker='x')
axs[1].plot(df_noBN.index + 1, df_noBN['Validation Accuracy'], label='No BatchNorm Validation Accuracy', marker='^')
axs[1].set_title('Validation Accuracy Across Configurations')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy (%)')
axs[1].legend()
axs[1].grid(True)

# Show the plots
plt.show()
