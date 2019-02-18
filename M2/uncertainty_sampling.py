from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from math import log
import re

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


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPath, self).__getitem__(index), self.imgs[index]


def uncertain_samples(model):
    uncertainty_dict = {}
    data = ImageFolderWithPath(root=root, transform=transforms)
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    for idx, value in enumerate(loader):
        (data, label), (path, _) = value
        model.eval()
        data = data.to(device)
        softmax = nn.Softmax(dim=1)

        outputs = model(data)
        pred = softmax(outputs)
        uncertainty_dict = update_uncertainty_dict(uncertainty_dict, path, pred)
        if idx == 2:
            break

    uncertainty_dict = sorted(uncertainty_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(uncertainty_dict)
    return uncertainty_dict


def extract_path (path, index):
    regex = r'(?<=JPEGImages\\)(?s)(.*$)'
    file_name = re.findall(regex, path[index])
    file_name = ", ".join(file_name)
    file_name = file_name.split(".")[0]
    return file_name


def update_uncertainty_dict(uncertainty_dict, path, pred):
    uncertainty = entropy(pred)
    for key, value in enumerate(uncertainty):
        index = extract_path(path, key)
        uncertainty_dict[index] = float(value)
    return uncertainty_dict


def entropy(prediction):
    return torch.div(torch.sum(-prediction * torch.log2(prediction), dim=1), log(4,2))


if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 4)#todo: make dataset and consider class size right now it doesnt matter
    uncertain_samples(net.to(device))


