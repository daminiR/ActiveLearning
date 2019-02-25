from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from math import log
import matplotlib.pyplot as plt
import numpy as np


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


class UncertaintySampler:

    def __init__(self, threshold=0.5, sample_size=8, verbose=True, iteration=1):
        self.threshold = threshold
        self.sample_size = sample_size
        self.verbose = verbose
        self.iteration = iteration

    def calculate_uncertainty(self, model, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.sample_size, shuffle=False)
        uncertainty_dict = {}
        model.eval()

        for idx, value in enumerate(loader):
            (data, label), index = value
            data = data.to(device)
            softmax = nn.Softmax(dim=1)
            outputs = model(data)
            pred = softmax(outputs)
            uncertainty_dict = self._update_uncertainty_dict(uncertainty_dict, index, pred)
            if idx == self.iteration:
                break

        uncertainty_list = sorted(uncertainty_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        if self.verbose:
            print(uncertainty_list)
            self._visualize_image(dataset[next(iter(uncertainty_list))[0]][0][0],
                                  'Most Uncertain Image, Entropy = %.3f' % next(iter(uncertainty_list))[1])
        return uncertainty_list

    def _update_uncertainty_dict(self, uncertainty_dict, index, prediction):
        uncertainty = self._entropy(prediction)
        for key, value in enumerate(uncertainty):
            if value > self.threshold:
                uncertainty_dict[int(index[key:key + 1])] = float(value)
        return uncertainty_dict

    @staticmethod
    def _visualize_image(image, title):
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 0, 1)
        plt.imshow(image)
        plt.title(title)
        plt.show()

    @staticmethod
    def _entropy(prediction):
        return torch.div(torch.sum(-prediction * torch.log2(prediction), dim=1), log(10, 2))


if __name__ == '__main__':
    SAMPLE_SIZE = 8
    NUM_CLASSES = 10

    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CIFAR10WithID(root=os.path.join(os.getcwd(), "CIFAR10"), train=True, transform=transforms, download=True)
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    M2 = UncertaintySampler(iteration=10)
    uncertainty = M2.calculate_uncertainty(net.to(device), dataset)

