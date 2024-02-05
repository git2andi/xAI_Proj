import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File path
file_path = 'C:\\Users\\Andi\\Desktop\\xAI_Proj\\Xception\\model_data\\Xception_1_CM.txt'

_, file_extension = os.path.splitext(file_path)

if file_extension == '.npy':
    confusion_matrix = np.load(file_path)
elif file_extension == '.txt':
    with open(file_path, 'r') as file:
        # Read lines, split by space, convert to integers, and form a matrix
        confusion_matrix = np.array([list(map(int, line.split())) for line in file])

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
