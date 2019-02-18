from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from math import log
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 8

transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CIFAR10WithID(datasets.CIFAR10):
    def __getitem__(self, index):
        return super(CIFAR10WithID, self).__getitem__(index), index


def uncertain_samples(model):
    dataset = CIFAR10WithID(root=os.path.join(os.getcwd(), "CIFAR10"), train=True, transform=transforms, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    uncertainty_dict = {}

    for idx, value in enumerate(loader):
        (data, label), index = value
        model.eval()
        data = data.to(device)
        softmax = nn.Softmax(dim=1)

        outputs = model(data)
        pred = softmax(outputs)
        uncertainty_dict = update_uncertainty_dict(uncertainty_dict, index, pred)
        if idx == 10:
            break

    uncertainty_dict = sorted(uncertainty_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(uncertainty_dict)
    visualize_image(dataset[next(iter(uncertainty_dict))[0]][0][0], 'Most Uncertain Image, Entropy = %f' % next(iter(uncertainty_dict))[1])
    return uncertainty_dict


def update_uncertainty_dict(uncertainty_dict, index, pred):
    uncertainty = entropy(pred)
    for key, value in enumerate(uncertainty):
        uncertainty_dict[int(index[key:key+1])] = float(value)
    return uncertainty_dict


def visualize_image(image, title):
    image = np.swapaxes(image, 0, 2)
    plt.imshow(image)
    plt.title(title)
    plt.show()


def entropy(prediction):
    return torch.div(torch.sum(-prediction * torch.log2(prediction), dim=1), log(10, 2))


if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 10)#todo: make dataset and consider class size right now it doesnt matter
    uncertain_samples(net.to(device))


