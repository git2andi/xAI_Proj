import matplotlib.pyplot as plt

# Data for initial and adapted SimpleCNN
epochs = list(range(1, 16))
initial_val_accuracy = [97.49, 98.16, 98.71, 98.68, 98.76, 98.73, 98.83, 98.94, 98.85, 99.03, 98.90, 98.75, 98.88, 99.04, 99.09]
adapted_val_accuracy = [98.82, 99.01, 99.13, 99.22, 99.29, 99.24, 99.38, 99.19, 99.42, 99.42, 99.46, 99.41, 99.44, 99.38, 99.49]

# Adjusting the figure size to make it more horizontal
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

# Plotting the Validation Accuracy
plt.plot(epochs, initial_val_accuracy, label='Initial SimpleCNN', marker='o')
plt.plot(epochs, adapted_val_accuracy, label='Adapted SimpleCNN', marker='x')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Zoom in on relevant parts for clarity
plt.ylim(97.5, 100)

plt.show()
