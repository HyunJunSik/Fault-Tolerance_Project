import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ID_OOD_dataloader(bz=64):
    ood_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    
        ])

    id_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    svhn_test = datasets.SVHN(root="./data", split='test', download=True, transform=ood_transform)
    cifar_test = datasets.CIFAR100(root='./../data', train=False, download=True, transform=id_transform)
    # Define data loaders
    ood_loader = DataLoader(svhn_test, batch_size=bz, shuffle=False)
    in_loader = DataLoader(cifar_test, batch_size=bz, shuffle=False)
    print(f"ID and OOD dataset load complete.")
    return in_loader, ood_loader
