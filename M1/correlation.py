#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.optim as lr_scheduler
from torchvision import models
import os, sys
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas
import gzip
import cca_core
from PIL import Image
from torch.autograd import Variable

def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()




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

    model(image)git add -A
    hook.remove()

    return feature_map


if __name__ == '__main__':
    img_path ='Data/dog.jpg'
    img = Image.open(img_path)
    trans = transform.ToPIL
    transformed_img = Variable(torch.tensor(img))
    net = models.resnet18(pretrained=True)
    layer = net.conv3()
    print(layer, sep=' ', end='n', file=sys.stdout, flush=False)
