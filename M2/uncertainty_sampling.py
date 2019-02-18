from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from math import log
import matplotlib.pyplot as plt
import numpy as np

# Parameters
SAMPLE_SIZE = 8
BATCH_SIZE = 16

transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CIFAR10WithID(datasets.CIFAR10):
    def __getitem__(self, index):
        return super(CIFAR10WithID, self).__getitem__(index), index


class CustomDataset(datasets.ImageFolder):
    def __init__(self, data, label, uncertainty_list, transforms=None):
        self.image_data = data
        self.id = [data[0] for data in uncertainty_list]
        self.entropy = [data[1] for data in uncertainty_list]
        self.label = label
        self.length = len(uncertainty_list)
        self.transforms = transforms

    def __getitem__(self, index):
        data = self.image_data[index]
        label = self.label[index]
        # if self.transforms is not None:
        #     data = self.transforms(data)
        return data, label, self.id[index], self.entropy[index]

    def __len__(self):
        return self.length


def uncertain_samples(model):
    dataset = CIFAR10WithID(root=os.path.join(os.getcwd(), "CIFAR10"), train=True, transform=transforms, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=SAMPLE_SIZE, shuffle=False)

    uncertainty_dict = {}

    for idx, value in enumerate(loader):
        (data, label), index = value
        model.eval()
        data = data.to(device)
        softmax = nn.Softmax(dim=1)

        outputs = model(data)
        pred = softmax(outputs)
        uncertainty_dict = update_uncertainty_dict(uncertainty_dict, index, pred)
        if idx == 1:
            break

    uncertainty_dict = sorted(uncertainty_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(uncertainty_dict)
    visualize_image(dataset[next(iter(uncertainty_dict))[0]][0][0], 'Most Uncertain Image, Entropy = %f' % next(iter(uncertainty_dict))[1])
    selected_dataset = extract_data(dataset, uncertainty_dict)
    return uncertainty_dict, selected_dataset

def extract_data(dataset, uncertainty_list):
    data = []
    label = []
    for key, value in enumerate(uncertainty_list):
        data.append(dataset[key][0][0])
        label.append(dataset[key][0][1])
    selected_data = CustomDataset(data, label, uncertainty_list, transforms=False)
    return selected_data



def update_uncertainty_dict(uncertainty_dict, index, pred):
    uncertainty = entropy(pred)
    for key, value in enumerate(uncertainty):
        if uncertainty_dict.__len__() < BATCH_SIZE:
            uncertainty_dict[int(index[key:key+1])] = float(value)
        else:
            minimum_key = min(uncertainty_dict, key=uncertainty_dict.get)
            minimum_value = uncertainty_dict[minimum_key]
            if value > minimum_value:
                del uncertainty_dict[minimum_key]
                uncertainty_dict[int(index[key:key + 1])] = float(value)
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
    uncertainty_list, selected_dataset = uncertain_samples(net.to(device))

    for key, value in enumerate(selected_dataset):
        print(value)

