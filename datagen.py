import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms

import os
import tarfile
import imageio
import tqdm
import numpy as np



def get_classes(target, labels):
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices



def load_cifar10(split=True):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    path = 'data_c/'
    if split:
        train_idx = [0, 1, 2, 3, 4, 5]
        test_idx = [6, 7, 8, 9]

    else:
        train_idx = list(range(10))
        test_idx = list(range(10))

    trainset = datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, train_idx))
    train_loader = torch.utils.data.DataLoader(train_hidden,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True)

    valset = datasets.CIFAR10(
            path,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
    val_hidden = torch.utils.data.Subset(valset, get_classes(valset, train_idx))
    val_loader = torch.utils.data.DataLoader(val_hidden,
            batch_size=16,
            shuffle=True,
            **kwargs)

    testset = datasets.CIFAR10(path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, test_idx))
    test_loader = torch.utils.data.DataLoader(test_hidden,
            batch_size=16,
            shuffle=False,
            **kwargs)

    return train_loader, test_loader, val_loader


