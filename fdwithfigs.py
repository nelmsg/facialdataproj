import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE

def read_values(name):
    with open(f"results/{name}", 'r') as file:
        values = [float(line.strip()) for line in file.readlines()]
    return values

epochs = range(1, 101)  # Assuming you have 100 epochs
train_acc = read_values('train_acc.txt')
test_acc = read_values('test_acc.txt')
train_loss = read_values('train_loss.txt')
test_loss = read_values('test_loss.txt')

plt.figure()
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs. Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, test_acc, label="Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Testing Accuracy vs. Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, train_loss, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()

# Testing Loss
plt.figure()
plt.plot(epochs, test_loss, label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Testing Loss over Epochs")
plt.legend()
plt.show()

# train_losses = torch.cat([torch.tensor(l) for l in train_loss]).numpy()
train_losses = torch.tensor(train_loss).numpy()
plt.figure()
plt.hist(train_losses, bins=30, alpha=0.7, label="Training Loss")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.title("Cross Entropy Loss Distribution")
plt.legend()
plt.show()
