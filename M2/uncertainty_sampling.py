from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from math import log
import matplotlib.pyplot as plt
import numpy as np


class UnlabelledDataset(torch.utils.data.Dataset):
    """
    Dataset class that can load either from PyTorch Library or a local folder.

    Arguments:
        1. dataset_name (String): Name of the dataset user wish to load. The local image folder should be in the
           following format: cwd/dataset_name/train and cwd/dataset_name/test
        2. transforms_train (Callable): A function that takes in a train image and return the transformed version.
        3. transforms_test (Callable): A function that takes in a test image and return the transformed version.
        4. num_classes (int): Only fill this if you are using a custom dataset that's not in PyTorch
    """

    def __init__(self, dataset_name, transform_train=None, transform_test=None, num_classes=0):
        available_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
        self.transform_train = transform_train
        if dataset_name in available_datasets:
            if dataset_name == 'MNIST':
                self.dataset_train = datasets.MNIST(root=os.path.join(os.getcwd(), dataset_name),
                                                    train=True, download=True)
                self.dataset_test = datasets.MNIST(root=os.path.join(os.getcwd(), dataset_name),
                                                   train=False, download=True, transform=transform_test)
                self.num_classes = 10
            if dataset_name == 'FashionMNIST':
                self.dataset_train = datasets.FashionMNIST(root=os.path.join(os.getcwd(), dataset_name),
                                                           train=True, download=True)
                self.dataset_test = datasets.FashionMNIST(root=os.path.join(os.getcwd(), dataset_name),
                                                          train=False, download=True, transform=transform_test)
                self.num_classes = 10
            if dataset_name == 'CIFAR10':
                self.dataset_train = datasets.CIFAR10(root=os.path.join(os.getcwd(), dataset_name),
                                                      train=True, download=True)
                self.dataset_test = datasets.CIFAR10(root=os.path.join(os.getcwd(), dataset_name),
                                                     train=False, download=True, transform=transform_test)
                self.num_classes = 10
            if dataset_name == 'CIFAR100':
                self.dataset_train = datasets.CIFAR100(root=os.path.join(os.getcwd(), dataset_name),
                                                       train=True, download=True)
                self.dataset_test = datasets.CIFAR100(root=os.path.join(os.getcwd(), dataset_name),
                                                      train=False, download=True, transform=transform_test)
                self.num_classes = 100
        else:
            path = os.path.join(os.getcwd(), dataset_name)
            self.dataset_train = datasets.ImageFolder(root=os.path.join(path, 'train'))
            self.dataset_test = datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform_test)
            self.num_classes = num_classes
        self.labelled_index = np.ones(len(self.dataset_train))

    def __getitem__(self, index):
        data, label = self.dataset_train[index]
        if self.transform_train:
            data = self.transform_train(data)
        return data, label, index

    def __len__(self):
        return len(self.dataset_train)

    """
    Function to set a particular data as labelled.

    Arguments:
        1. index (list-like): index of the current set of images sent to be labelled. 
    
    Return:
        Void
    """
    def mark(self, index):
        for i in index:
            self.labelled_index[i] = 0



class SequentialSubsetSampler(torch.utils.data.Sampler):
    """
    Sampler class that samples a subset dataset sequentially

    Arguments:
        1. dataset_name (sequence): a sequence of indices of desired subset
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class UncertaintySampler:
    """
    Sampler class that returns a list of tuples with indices and entropy value of a dataset.

    Arguments:
        1. threshold (float): The threshold level where entropy < threshold will not be returned.
        2. sample_size (int): The number of images to be sent to the GPU at one iteration.
        3. verbose (boolean): If True, calling calculate_uncertainty will print the entropy list and
                              display the image with the highest uncertainty.
        4. iteration (int)  : Number of sampling. To get full dataset, set iteration to None
    """

    def __init__(self, threshold=0.5, sample_size=8, verbose=True, iteration=10):
        self.threshold = threshold
        self.sample_size = sample_size
        self.verbose = verbose
        self.iteration = iteration

    """
    Function to calculate entropy value of a dataset.

    Arguments:
        1. model (Model)              : Current model
        2. dataset (UnlabelledDataset): The target domain dataset
    
    Return:
        1. List of tuples in the format [(index, (entropy, tensor), (index, (entropy, tensor)),..]
    """

    def calculate_uncertainty(self, model, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.sample_size,
                                             sampler=SequentialSubsetSampler(np.where(dataset.labelled_index)[0]))
        uncertainty_dict = {}
        model.eval()
        num_classes = dataset.num_classes

        for idx, value in enumerate(loader):
            data, label, index = value
            data = data.to(device)
            softmax = nn.Softmax(dim=1)
            with torch.no_grad():
                outputs = model(data)
                #print(torch.max(outputs))
                # TODO: ensure results are stable
                pred = softmax(outputs)
            uncertainty_dict = self._update_uncertainty_dict(uncertainty_dict, index, pred, num_classes)
            # if idx+1 == self.iteration:
            #     break

        uncertainty_list = sorted(uncertainty_dict.items(), key=lambda kv: (kv[1][0], kv[0]), reverse=True)
        if self.verbose:
            print(uncertainty_list)
            imageData, _ = (next(iter(uncertainty_list))[1])
            self._visualize_image(dataset[next(iter(uncertainty_list))[0]][0],
                                  'Most Uncertain Image, Entropy = %.3f' % imageData)
        return uncertainty_list

    def _update_uncertainty_dict(self, uncertainty_dict, index, prediction, num_classes):
        uncertainty = self._entropy(prediction, num_classes)
        for key, value in enumerate(uncertainty):
            if value > self.threshold:
                uncertainty_dict[int(index[key:key + 1])] = (float(value), prediction[key])
        return uncertainty_dict

    @staticmethod
    def _visualize_image(image, title):
        image = image.numpy()
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 0, 1)
        plt.imshow(image)
        plt.title(title)
        plt.show()

    def _entropy(self, prediction, num_classes):
        return torch.div(torch.sum(-prediction * torch.log2(prediction), dim=1), log(num_classes, 2))


if __name__ == '__main__':
    SAMPLE_SIZE = 4
    NUM_CLASSES = 10

    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UnlabelledDataset('CIFAR10', transform_train=transforms)
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    M2 = UncertaintySampler(sample_size=SAMPLE_SIZE, iteration=2)

    # Send image number 1 to be labelled
    labelled_index = [1]
    dataset.mark(labelled_index)
    uncertainty1 = M2.calculate_uncertainty(net.to(device), dataset)

    # Send image number 4, 6 to be labelled
    labelled_index = np.array([4, 6])
    dataset.mark(labelled_index)
    uncertainty2 = M2.calculate_uncertainty(net.to(device), dataset)

