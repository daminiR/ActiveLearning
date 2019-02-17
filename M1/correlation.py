#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.optim as lr_scheduler
from torchvision import models
import torchvision.transforms as transforms
import os, sys
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas
import gzip
import cca_core
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


# transform1 = transforms.Compose([
#         transforms.Resize(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#

def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def compute_out_size(in_size, mod):
    """Compute output size of Module `mod` given an input with size `in_size`.

    Parameters
    ----------
    in_size : type
        Description of parameter `in_size`.
    mod : type
        Description of parameter `mod`.

    Returns
    -------
    type
        Description of returned object.

    """

    f = mod.forward(Variable(torch.Tensor(1, *in_size)))
    return int(np.prod(f.size()[1:]))


def _get_feature_map_size(model, layer, image):
    size = [0]

    def forward_hook(layer, input, output):
        print('get size')
        size[0] = output.size()


    hook = layer.register_forward_hook(forward_hook)

    model(image)
    hook.remove()
    return size[0]



def get_feature_map(model, layer, image):
    """Short summary.

    Parameters
    ----------
    model : type
        Description of parameter `model`.
    layer : type
        Description of parameter `layer`.
    image : type
        Description of parameter `image`.
    size : type
        Description of parameter `size`.

    Returns
    -------
    type
        Description of returned object.

    """

    feature_map = torch.zeros()

    def forward_hook(layer, input, output):
        feature_map = torch.zeros(output.size())
        feature_map.copy_(output.data)

    hook = layer.register_forward_hook(forward_hook)

    model(image)
    hook.remove()

    return feature_map


if __name__ == '__main__':
    img_path ='Data/dog.jpg'
    img = Image.open(img_path)
    totensor = transforms.ToTensor()
    scaler = transforms.Resize((224, 224))
    transformed_img = Variable(totensor(scaler(img)))
    net = models.alexnet(pretrained=True)
    net.eval()
    # size = compute_out_size(transformed_img.size(), net)
    layer1 = net._modules.get('features')[0]
    print(_get_feature_map_size(net, layer1, transformed_img.unsqueeze(0)))
    # print(get_feature_map(net, layer1, transformed_img.unsqueeze(0)))
