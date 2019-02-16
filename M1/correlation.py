#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.optim as lr_scheduler
from torchvision import models


if name == '__main__':

    net = models.resnet18(pretrained=True)
