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


def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def compute_out_size(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """

    f = mod.forward(Variable(torch.Tensor(1, *in_size)))
    return int(np.prod(f.size()[1:]))


def get_feature_map(model, layer, image, size):
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
    feature_map = torch.zeros(size)

    def forward_hook(layer, input, output):
        feature_map.copy_(output.data)

    hook = layer.register_forward_hook(forward_hook)

    model(image)
    hook.remove()

    return feature_map


if __name__ == '__main__':
    img_path ='Data/dog.jpg'
    img = Image.open(img_path)
    totensor = transforms.ToTensor()
    scaler = transforms.Scale((224, 224))
    # t_img = Variable((to_tensor(scaler(img)).unsqueeze(0))
    transformed_img = Variable(totensor(scaler(img)))
    net = models.resnet18(pretrained=True)
    compute_out_size(transformed_img.size(), net))

    # print(net)
    # layer1 = net._modules.get('layer1')
    # print(layer1)
