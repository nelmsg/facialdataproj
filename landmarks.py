import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import (Normalize, Compose, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Resize, CenterCrop, RandomRotation)
import torch.optim as optim
from pranc_models import resnet20, resnet56
import argparse
from torch.optim.lr_scheduler import StepLR
from timm.data import Mixup

from fmd.ds300vw import DS300VW

# Set the path to the dataset directory.
DS300VW_DIR = "/home/robin/data/facial-marks/300VW"

# Construct a dataset.
ds = DS300VW("300Vw")

# Populate the dataset with essential data
ds.populate_dataset(DS300VW_DIR)

# See what we have got.
print(ds)

sample = ds.pick_one()
