import pandas as pd
import matplotlib.pyplot as plt

# Assuming your CSV file is correctly loaded
df = pd.read_csv('C:\\Users\\Andi\\Desktop\\xAI_Proj\\Andreas\\Xception\\model_data\\training_results_SimpleCNN_all.csv')

# Convert numeric columns to floats for plotting
df[['Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1 Score',
    'Validation Loss', 'Validation Accuracy', 'Validation Precision', 'Validation Recall', 'Validation F1 Score']] = \
    df[['Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1 Score',
        'Validation Loss', 'Validation Accuracy', 'Validation Precision', 'Validation Recall', 'Validation F1 Score']].astype(float)

# Epochs as x-axis
epochs = df['Epoch'].astype(int)

# Fixed axis limits
loss_lim = (0, 1)
accuracy_lim = (70, 100)
precision_recall_f1_lim = (0, 1)

# Plotting
fig, axs = plt.subplots(2, figsize=(15, 10), constrained_layout=True)

# Loss Plot
axs[0].plot(epochs, df['Train Loss'], label='Train Loss')#, marker='o')
axs[0].plot(epochs, df['Validation Loss'], label='Validation Loss', marker='x')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim(loss_lim)
axs[0].legend()
axs[0].grid(True)

# Accuracy Plot
axs[1].plot(epochs, df['Train Accuracy'], label='Train Accuracy')#, marker='o')
axs[1].plot(epochs, df['Validation Accuracy'], label='Validation Accuracy', marker='x')
axs[1].set_title('Training and Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim(accuracy_lim)
axs[1].legend()
axs[1].grid(True)

# Save the plot as a PNG file
plt.savefig('SimpleCNN_all.png', dpi=300)

# Show the plots
plt.show()
