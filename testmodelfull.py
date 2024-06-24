import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(0.5077, 0.2550)])
device = torch.device('cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 7)

state_dict = torch.load('resnet18_model_file_mixup_on_orig.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
model.to(device)

emotions_test = ImageFolder(root="./archive/test",transform=transform)  # TRANSFORMS EVERY SAMPLE IN DATASET
dataloader_test = DataLoader(emotions_test, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

datapoints = 0  # INITIALIZING VARIABLES
correct_datapoints = 0  # INITIALIZING VARIABLES

with torch.no_grad():
    for x, y in dataloader_test:  # GOES THROUGH ALL TESTING DATA
        x = x.to(device)  # SENDS VARIABLE TO CPU OR GPU
        y = y.to(device)  # SENDS VARIABLE TO CPU OR GPU

        datapoints += len(y)  # COUNTS TOTAL TESTING POINTS
        y_prediction_test = model(x)  # PREDICTS CLASS
        correct_datapoints += torch.sum((y_prediction_test.argmax(dim=1) == y))

        print(f"Accuracy: {100 * torch.sum((y_prediction_test.argmax(dim=1) == y)) / len(y)}%")

    print(f"Final Accuracy: {100 * correct_datapoints / datapoints}%")
