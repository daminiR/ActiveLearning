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
from cca_core import get_cca_similarity


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






def get_feature_map(model, layer, image):
    """Returns the feature map .

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

    result = [None]

    def forward_hook(layer, input, output):
        feature_map = torch.zeros(output.size())
        feature_map.copy_(output.data)
        result[0] = feature_map

    hook = layer.register_forward_hook(forward_hook)

    model(image)
    hook.remove()

    return result[0]

def correlate_layers(feature_map1, feature_map2):
    """Short summary.

    Parameters
    ----------
    feature_map1 : type
        Description of parameter `feature_map1`.
    feature_map2 : type
        Description of parameter `feature_map2`.

    Returns
    -------
    type
        Description of returned object.

    """
    print(get_cca_similarity(feature_map1, feature_map2, epsilon=0., threshold=0.98, compute_coefs=True, compute_dirns=False, verbose=True))




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
    # print(_get_feature_map_size(net, layer1, transformed_img.unsqueeze(0)))
    # print(get_feature_map(net, layer1, transformed_img.unsqueeze(0)))
    featureA = get_feature_map(net, layer1, transformed_img.unsqueeze(0))
    featureB = get_feature_map(net, layer1, transformed_img.unsqueeze(0))
    print("what")
    print(featureA.shape)
    correlate_layers(featureA, featureB)
