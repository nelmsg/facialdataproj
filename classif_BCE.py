import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

def load_mnist():
    transforms = torchvision.transforms.Compose([pil_to_tensor, lambda x: x.squeeze().flatten().float() / 255])

    mnist_train = MNIST(root="./data", train=True, download=True, transform=transforms)
    mnist_test = MNIST(root="./data", train=False, download=False, transform=transforms)

    return mnist_train, mnist_test

mnist_train, mnist_test = load_mnist()
print("\n")

train_cat = torch.cat([x.unsqueeze(0) for x, y in mnist_train if y == 0 or y ==1], dim=0)
labels_cat_train = torch.tensor([y for x, y in mnist_train if y == 0 or y == 1])
# zeros_train = mnist_train.data[mnist_train.targets == 0]
# ones_train = mnist_train.data[mnist_train.targets == 1]
# train_cat = torch.concat([zeros_train, ones_train], dim=0)

test_cat = torch.cat([x.unsqueeze(0) for x, y in mnist_test if y == 0 or y == 1], dim=0)
labels_cat_test = torch.tensor([y for x, y in mnist_test if y == 0 or y == 1])

# zeros_test = mnist_test.data[mnist_test.targets == 0]
# ones_test = mnist_test.data[mnist_test.targets == 1]
# test_cat = torch.concat([zeros_test, ones_test], dim=0)

model = torch.nn.Linear(784, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0, weight_decay=0)

dif = torch.inf  # Placeholder value
current_loss = torch.inf  # Placeholder value
i_loss = torch.inf
print(f"i_loss: {i_loss}")

losses = []
iterations = 0

# while dif > 1e-4 and iterations < 500:
while iterations < 500:
    optimizer.zero_grad()
    predictions = model(train_cat)
    # current_loss = torch.nn.functional.mse_loss(predictions, y_t.view(-1, 1))

    loss_function = nn.CrossEntropyLoss()
    current_loss = loss_function(predictions, labels_cat_train)
    print(f"i_loss: {i_loss}")
    print(f"current_loss: {current_loss}")

    boolean_predictions = (predictions.argmax(dim=1))
    correct = (boolean_predictions == labels_cat_train).sum()
    accuracy = correct / labels_cat_train.size(0)
    print(f"  accuracy: {accuracy}")

    current_loss.backward()
    optimizer.step()
    dif = i_loss - current_loss
    print(f"dif: {dif}")
    i_loss = current_loss

    losses.append(current_loss.item())
    iterations += 1