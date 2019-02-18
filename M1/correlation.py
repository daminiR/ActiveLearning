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
from pprint import pprint as pp



# transform1 = transforms.Compose([
#         transforms.Resize(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#

def _plot_helper(arr, xlabel, ylabel, title, output_file):
    x_range = np.arange(len(arr)); pp(x_range)
    baseline_correlation = np.mean(arr)

    plt.ylim([0, 1.2])
    plt.plot(np.arange(len(arr)), arr)
    plt.hlines(baseline_correlation, xmin = 0, xmax=len(arr))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(0, baseline_correlation + 0.01, str(baseline_correlation))
    plt.grid()
    print(arr[-1])
    plt.savefig(output_file)



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

def correlate_layers(feature_map1, feature_map2, title, output_file):
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

    num_imgs, channels, h, w = feature_map1.size()
    features_i = feature_map1.view(-1, channels).numpy()  #numpy bridge

    num_imgs, channels, h, w = feature_map2.size()
    features_j = feature_map2.view(-1, channels).numpy()   #numpy bridge

    print(features_i.shape)
    print(features_j.shape)
    # print(fourier_ccas(feature_map1, featureB, return_coefs=True, compute_dirns=True,verbose=False))

    result = get_cca_similarity(features_i.T, features_j.T, epsilon=1e-10,
                                                         compute_coefs=True,
                                                         compute_dirns=False,
                                                         verbose=True)
    # print(result["cca_coef1"])
    _plot_helper(result["cca_coef1"], "CCA Coef idx", "coef value",title,  output_file)
    pp(result["cca_coef1"])


# python3 correlation.py >> log.txt



if __name__ == '__main__':
    img_i_path ='Data/dog.jpg'
    img_j_path ='Data/dog.jpg'

    img_i = Image.open(img_i_path)
    img_j = Image.open(img_j_path)

    totensor = transforms.ToTensor()
    scaler   = transforms.Resize((224, 224))
    transformed_img_i = Variable(totensor(scaler(img_i)))
    transformed_img_j = Variable(totensor(scaler(img_j)))
    net = models.alexnet(pretrained=True)
    net.eval()
    # size = compute_out_size(transformed_img.size(), net)
    layer1 = net._modules.get('features')[0]
    # print(_get_feature_map_size(net, layer1, transformed_img.unsqueeze(0)))
    # print(get_feature_map(net, layer1, transformed_img.unsqueeze(0)))
    featureA = get_feature_map(net, layer1, transformed_img_i.unsqueeze(0))
    featureB = get_feature_map(net, layer1, transformed_img_j.unsqueeze(0))
    print("what")
    print(featureA.shape)
    correlate_layers(featureA, featureB, title = 'dogVSdog', output_file='Plots/dogVSdog.jpg')
