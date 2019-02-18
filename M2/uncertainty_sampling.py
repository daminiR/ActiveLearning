from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from math import log
import matplotlib.pyplot as plt

BATCH_SIZE = 8

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
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)


def uncertain_samples(model, loader):
    uncertainty_dict = {}
    for idx, (data, label) in enumerate(loader):
        visualize_image(data[0])
        model.eval()
        data = data.to(device)
        softmax = nn.Softmax(dim=1)
        visualize_image(data)

        outputs = model(data)
        pred = softmax(outputs)
        uncertainty_dict = update_uncertainty_dict(uncertainty_dict, idx, pred)
        break

    uncertainty_dict = sorted(uncertainty_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return uncertainty_dict


def update_uncertainty_dict(uncertainty_dict, idx, pred):
    uncertainty = entropy(pred)
    for key, value in enumerate(uncertainty):
        index = str(key + (idx*BATCH_SIZE))
        uncertainty_dict[index] = float(value)
    return uncertainty_dict


def entropy(prediction):
    return torch.div(torch.sum(-prediction * torch.log2(prediction), dim=1), log(4,2))


def visualize_image(image):
    plt.imshow(image.permute(2, 1, 0))
    plt.show()


if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 4)#todo: make dataset and consider class size right now it doesnt matter
    uncertain_samples(net.to(device), loader=loader)



