import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

emotions_train = ImageFolder(root="./archive/train", transform=ToTensor())
emotions_test = ImageFolder(root="./archive/test", transform=ToTensor())

class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

train_counts = [0] * len(class_names)
test_counts = [0] * len(class_names)

train_p_counts = []
test_p_counts = []

for x, y in emotions_train:
    train_counts[y] += 1

for x, y in emotions_test:
    test_counts[y] += 1

print("Training set class counts: ", train_counts)
print("Testing set class counts: ", test_counts)

for i in train_counts:
    train_p_counts.append((100 * (i / sum(train_counts))))
print("Training set percentages: ", train_p_counts)

for i in test_counts:
    test_p_counts.append((100 * (i / sum(test_counts))))
print("Testing set percentages: ", test_p_counts)
