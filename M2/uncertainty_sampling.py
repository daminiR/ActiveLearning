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


class UnlabelledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, transforms=None):
        available_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
        self.unlabelled_index = []
        self.transforms = transforms
        if dataset_name in available_datasets:
            if dataset_name == 'MNIST':
                self.dataset_train = datasets.MNIST(root=os.path.join(os.getcwd(), dataset_name), train=True, download=True)
                self.dataset_test = datasets.MNIST(root=os.path.join(os.getcwd(), dataset_name), train=False, download=True)
            if dataset_name == 'FashionMNIST':
                self.dataset_train = datasets.FashionMNIST(root=os.path.join(os.getcwd(), dataset_name), train=True, download=True)
                self.dataset_test = datasets.FashionMNIST(root=os.path.join(os.getcwd(), dataset_name), train=False, download=True)
            if dataset_name == 'CIFAR10':
                self.dataset_train = datasets.CIFAR10(root=os.path.join(os.getcwd(), dataset_name), train=True, download=True)
                self.dataset_test = datasets.CIFAR10(root=os.path.join(os.getcwd(), dataset_name), train=False, download=True)
            if dataset_name == 'CIFAR100':
                self.dataset_train = datasets.CIFAR100(root=os.path.join(os.getcwd(), dataset_name), train=True, download=True)
                self.dataset_test = datasets.CIFAR100(root=os.path.join(os.getcwd(), dataset_name), train=False, download=True)
        else:
            path = os.path.join(os.getcwd(), dataset_name)
            self.dataset_train = datasets.ImageFolder(root=os.path.join(path, 'train'))
            self.dataset_test = datasets.ImageFolder(root=os.path.join(path, 'test'))

    def __getitem__(self, index):
        data, label = self.dataset_train[index]
        if self.transforms:
            data = self.transforms(data)
        return data, label, index

    def _x_iter__(self):
        return (self.indices[i] for i in self.unlabelled_inde)

    def __len__(self):
        return len(self.dataset_train)

    def mark(self, index):
        self.unlabelled_index.append(index)


class UncertaintySampler:

    def __init__(self, threshold=0.5, sample_size=8, verbose=True, iteration=1):
        self.threshold = threshold
        self.sample_size = sample_size
        self.verbose = verbose
        self.iteration = iteration
    #     TODO: How to set max iterations

    def calculate_uncertainty(self, model, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.sample_size, shuffle=False)
        uncertainty_dict = {}
        model.eval()

        for idx, value in enumerate(loader):
            data, label, index = value
            print(data.shape)
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
        image = image.numpy()
        print(image.shape)
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
    dataset = UnlabelledDataset('CIFAR10', transforms=transforms)
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    M2 = UncertaintySampler(iteration=1)
    uncertainty = M2.calculate_uncertainty(net.to(device), dataset)

