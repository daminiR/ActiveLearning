from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from pprint import pprint
from math import log

transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = os.getcwd()
root = os.path.join(path, 'VOCdevkit\VOC2012')
data = datasets.ImageFolder(root= root, transform=transforms)
loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)

def uncertain_samples(model, loader):
    for idx, (data, label) in enumerate(loader):
        model.eval()
        data = data.to(device)
        outputs = model(data)
        softmax = nn.Softmax(dim=1)
        #TODO: do we need an alg for num class proprtion to the choice of uncertain value for i.e 0.5 and below verus 0.3 and below

        pred = softmax(outputs)
        print(pred)
        uncertainty = entropy(pred)
        print(uncertainty)
        break

def entropy(prediction):
    return torch.div(torch.sum(-prediction * torch.log2(prediction),dim=1), log(4,2))

if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 4)#todo: make dataset and consider class size right now it doesnt matter
    uncertain_samples(net.to(device), loader=loader)



