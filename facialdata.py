import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import (Normalize, Compose, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Resize, CenterCrop)
import torch.optim as optim
from pranc_models import resnet20, resnet56
import argparse
from torch.optim.lr_scheduler import StepLR

# device = torch.device('cuda')  # USES A GPU
device = torch.device('cpu')  # USES CPU

parser = argparse.ArgumentParser()  # DEFINING ARGUMENTS
parser.add_argument('--arch', type=str,
                    choices=['resnet18', 'resnet20', 'resnet56'])  # GIVING CMD LINE CHOICES

if parser.parse_args().arch == 'resnet18':  # IF OPTION SELECTED
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'resnet18', pretrained=True)  # LOAD RESNET18
    model.fc = nn.Linear(model.fc.in_features, 7)
elif parser.parse_args().arch == 'resnet20':  # IF OPTION SELECTED
    model = resnet20(num_classes=7)  # LOAD RESNET20
elif parser.parse_args().arch == 'resnet56':  # IF OPTION SELECTED
    model = resnet56(num_classes=7)  # LOAD RESNET56

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # LOADS RESNET18
# model.fc = nn.Linear(model.fc.in_features, 7)  # DEFINES THE EIGHT 'FEATURES' FOR EMOTION CATEGORIZATION
model = model.to(device)  # SENDS THE MODEL TO CPU OR GPU

dataset_counter = ImageFolder(root="./archive/train", transform=ToTensor())  # MINI TRANSFORMS FOR COUNTING
total_pixels = torch.stack([x for x, y in dataset_counter])  # COMBINES TENSORS INTO ONE STACK
pixels_mean = torch.mean(total_pixels)  # MEAN VALUE FOR NORMALIZATION
pixels_std = torch.std(total_pixels)  # STANDARD DEVIATION FOR NORMALIZATION

if model == resnet20(num_classes=7):
    transforms_train = Compose([Resize(256),  RandomCrop(224), ColorJitter(), RandomHorizontalFlip(),
                                ToTensor(), Normalize(pixels_mean, pixels_std)])  # RANDOM TRANSFORMS TRAIN GENERALIZING
    transforms_test = Compose([Resize(256), CenterCrop(224), ToTensor(),
                               Normalize(pixels_mean, pixels_std)])  # COMPOSED TRANSFORMS FOR NORMALIZATION
else:
    transforms_train = Compose([RandomCrop(48, padding=4), ColorJitter(), RandomHorizontalFlip(),
                                ToTensor(), Normalize(pixels_mean, pixels_std)])  # RANDOM TRANSFORMS TRAIN GENERALIZING
    transforms_test = Compose([ToTensor(),
                               Normalize(pixels_mean, pixels_std)])  # COMPOSED TRANSFORMS FOR NORMALIZATION

emotions_train = ImageFolder(root="./archive/train", transform=transforms_train)  # TRANSFORMS EVERY SAMPLE IN DATASET
emotions_test = ImageFolder(root="./archive/test", transform=transforms_test)  # TRANSFORMS EVERY SAMPLE IN DATASET
dataloader_train = DataLoader(emotions_train, batch_size=256, shuffle=True)  # MAKES A SHUFFLING DATALOADER FOR TRAINING
dataloader_test = DataLoader(emotions_test, batch_size=256, shuffle=False)  # MAKES A DATALOADER FOR TESTING

optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4)  # ESTABLISHES SGD OPTIMIZER
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # ESTABLISHES LEARNING RATE DECAY
loss_f = nn.CrossEntropyLoss()  # ESTABLISHES A LOSS FUNCTION BASED ON CEL

test_acc = []  # INITIALIZING VARIABLES
test_loss = []  # INITIALIZING VARIABLES
previous_accuracy = 0  # INITIALIZING VARIABLES
previous_loss = 0  # INITIALIZING VARIABLES

n_epochs = 100  # NUMBER OF TOTAL ITERATIONS
for epoch in range(n_epochs):  # RUNS FOR EVERY EPOCH
    batch_num = 0  # INITIALIZING VARIABLES
    cumulative_loss = 0  # INITIALIZING VARIABLES
    cumulative_test_loss = 0  # INITIALIZING VARIABLES
    total_data_train = 0  # INITIALIZING VARIABLES
    correct_data_train = 0  # INITIALIZING VARIABLES
    model.train()  # REMOVING DROPOUT

    for x, y in dataloader_train:  # RUNS FOR EVERY BATCH
        x = x.to(device)  # SENDS VARIABLE TO CPU OR GPU
        y = y.to(device)  # SENDS VARIABLE TO CPU OR GPU

        total_data_train += len(y)  # ADD BATCH TO TOTAL LENGTH
        Y_prediction = model(x)  # PUTS THE STACK THROUGH THE MODEL FOR THE PREDICTED VALUE
        loss = loss_f(Y_prediction, y)  # PULLS LOSS THROUGH THE CEL FUNCTION
        cumulative_loss += loss.detach().cpu()  # BUILDS TOTAL LOSS
        acc = torch.mean((Y_prediction.argmax(dim=1) == y).float())  # ACCURACY
        correct_data_train += torch.sum((Y_prediction.argmax(dim=1) == y).float())  # ADD

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # PROGRESS
        batch_num += 1  # COUNTING

        print(f"    Epoch: {epoch}  Batch: {batch_num}  Accuracy: {acc}  BS: {len(y)}")  # BATCH-WISE REPORT
    cumulative_train_accuracy = correct_data_train / total_data_train
    print(f"Cumulative Loss: {cumulative_loss}   "
          f"Cumulative Training Accuracy: {cumulative_train_accuracy}")  # EPOCH-WISE TRAINING REPORT

    with torch.no_grad():
        model.eval()  # REMOVES DROPOUT
        datapoints = 0  # INITIALIZING VARIABLES
        correct_datapoints = 0  # INITIALIZING VARIABLES
        for x, y in dataloader_test:  # GOES THROUGH ALL TESTING DATA
            x = x.to(device)  # SENDS VARIABLE TO CPU OR GPU
            y = y.to(device)  # SENDS VARIABLE TO CPU OR GPU

            datapoints += len(y)  # COUNTS TOTAL TESTING POINTS
            y_prediction_test = model(x)  # PREDICTS CLASS
            loss_t = loss_f(y_prediction_test, y)  # FINDS LOSS
            cumulative_test_loss += loss_t.detach().cpu()  # BUILDS TOTAL LOSS
            correct_datapoints += torch.sum((y_prediction_test.argmax(dim=1) == y))
        acc_t = correct_datapoints / datapoints  # CALCULATES AVERAGE ACCURACY

    if acc_t < previous_accuracy:
        accuracy_annotation = "\033[1;31;40m [Negative Change.]"
    else:
        accuracy_annotation = "\033[1;32;40m [Improvement!]"
    if cumulative_test_loss > previous_loss:
        loss_annotation = "\033[1;31;40m [Negative Change.] \033[1;37;40m"
    else:
        loss_annotation = "\033[1;32;40m [Improvement!] \033[1;37;40m"
    print(f"\033[1;37;40m Epoch: {epoch}  Total Test Loss: {cumulative_test_loss} {loss_annotation}  "
          f"Test Accuracy: {acc_t} {accuracy_annotation}")  # EPOCH-WISE TOTAL REPORT
    previous_loss = cumulative_test_loss  # DEFINES PREVIOUS TESTING LOSS
    previous_accuracy = acc_t  # DEFINES PREVIOUS TESTING ACCURACY
    scheduler.step()
