import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('C:\\Users\\Andi\\Desktop\\xAI_Proj\\Andreas\\Xception\\model_data\\training_results_xception_1.csv').head(50)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)

# Loss plot for no Dropout configuration
axs[0].plot(df.index + 1, df['Validation Loss'], label='Validation Loss', marker='x')
axs[0].plot(df.index + 1, df['Train Loss'], label='Train Loss', marker='o')
#axs[0].set_title('Validation and Losses')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

# Accuracy plot for no Dropout configuration
axs[1].plot(df.index + 1, df['Validation Accuracy'], label='Validation Accuracy', marker='x')
axs[1].plot(df.index + 1, df['Train Accuracy'], label='Train Accuracy', marker='o')
#axs[1].set_title('Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy (%)')
axs[1].legend()
axs[1].grid(True)

# Show the plots
plt.show()
