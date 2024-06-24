import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import (Normalize, Compose, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Resize, CenterCrop, RandomRotation)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # LOAD RESNET18
model.fc = nn.Linear(model.fc.in_features, 7)  # DEFINE NUMBER OF CLASSES
state_dict = torch.load('resnet18_model_file_mixup_on_orig.pth', map_location='cpu')  # LOADING WEIGHTS
model.load_state_dict(state_dict)  # LOADING THE MODEL WITH SET WEIGHTS
model.fc = nn.Identity()

device = torch.device('cpu')  # USES A GPU
model = model.to(device)  # SENDS THE MODEL TO CPU OR GPU

transform = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(0.5077, 0.2550)])

emotions_test = ImageFolder(root="./archive/train", transform=transform)  # TRANSFORMS EVERY SAMPLE IN DATASET
dataloader_test = DataLoader(emotions_test, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)


def embed_space(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            features = model(x).detach().cpu().numpy()
            embeddings.append(features)
            labels.append(y.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure()
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", s=1, alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.title("2D Embedding Space Visual")
    plt.show()


embed_space(model, dataloader_test, device)
