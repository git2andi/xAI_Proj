import pandas as pd
import matplotlib.pyplot as plt

# Load the organized data
df = pd.read_csv('C:\\Users\\Andi\\Desktop\\xAI_Proj\\Xception\\model_data\\training_results_SimpleCNN_all.csv')

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
fig, axs = plt.subplots(3, 2, figsize=(15, 20), constrained_layout=True)

# Loss Plot
axs[0, 0].plot(epochs, df['Train Loss'], label='Train Loss')                # marker='o'
axs[0, 0].plot(epochs, df['Validation Loss'], label='Validation Loss')
axs[0, 0].set_title('Training and Validation Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_ylim(loss_lim)
axs[0, 0].legend()

# Accuracy Plot
axs[0, 1].plot(epochs, df['Train Accuracy'], label='Train Accuracy')
axs[0, 1].plot(epochs, df['Validation Accuracy'], label='Validation Accuracy')
axs[0, 1].set_title('Training and Validation Accuracy')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].set_ylim(accuracy_lim)
axs[0, 1].legend()

# Precision Plot
axs[1, 0].plot(epochs, df['Train Precision'], label='Train Precision')
axs[1, 0].plot(epochs, df['Validation Precision'], label='Validation Precision')
axs[1, 0].set_title('Training and Validation Precision')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Precision')
axs[1, 0].set_ylim(precision_recall_f1_lim)
axs[1, 0].legend()

# Recall Plot
axs[1, 1].plot(epochs, df['Train Recall'], label='Train Recall')
axs[1, 1].plot(epochs, df['Validation Recall'], label='Validation Recall')
axs[1, 1].set_title('Training and Validation Recall')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Recall')
axs[1, 1].set_ylim(precision_recall_f1_lim)
axs[1, 1].legend()

# F1 Score Plot
axs[2, 0].plot(epochs, df['Train F1 Score'], label='Train F1 Score')
axs[2, 0].plot(epochs, df['Validation F1 Score'], label='Validation F1 Score')
axs[2, 0].set_title('Training and Validation F1 Score')
axs[2, 0].set_xlabel('Epochs')
axs[2, 0].set_ylabel('F1 Score')
axs[2, 0].set_ylim(precision_recall_f1_lim) 
axs[2, 0].legend()

# Remove the empty subplot (bottom right)
fig.delaxes(axs[2, 1])

# Save the plot as a PNG file
plt.savefig('CNN1.png', bbox_inches='tight', dpi=300)

# Show the plots
plt.show()
