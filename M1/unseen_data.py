#load trained function
#creata new CNN from base network
#add Class layers
import torch
from torchvision import models
import torch.nn as nn
from torchvision import  transforms, datasets
from torch.autograd import  Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import copy
import torchvision
import numpy as np
import sys
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
import os
import os.path
import sys
from PIL import Image


class MyImageFolder(datasets.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None):
        super(MyImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.labelled_index = np.ones(len(self.samples))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

    def mark(self, index):
        for i in index:
            self.labelled_index[i] = 0


sys.path.append("..")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.set_printoptions(threshold=sys.maxsize)

from M2.uncertainty_sampling import UnlabelledDataset, SequentialSubsetSampler
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# dataloader = torch.utils.data.DataLoader(
#             datasets.CIFAR100('../CIFAR100_data', train=True, download=True,
#                               transform=transforms.Compose(
#                                   [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])),
#             batch_size=4, shuffle=True, num_workers=5)

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

class access_super_loader():
    def __init__(self, unseen_set_indices_list, batch_size_set):
        self.idx = unseen_set_indices_list
        self.size_batch = batch_size_set
    def get_loader(self):
        subset_loader = torch.utils.data.DataLoader(dataset, batch_size=self.size_batch, sampler=SubsetRandomSampler(self.idx))
        return subset_loader.dataset


def acceptance_layer(idx, predictions, threshold):#happens during testnig
    """

    Parameters
    ----------
    idx -  at batch index ids
    predictions - tensor of (N,C)  #TODO : works for 1 batch class
    set_of_accepted - (N
    threshold

    Returns
    -------

    """
    # return torch.gt(predictions, threshold).nonzero()[:,0].unique()
    set_of_rejected = list()
    #for each image
    for image_ID in idx:
        #for each nueron
        # print(predictions[image_ID % 8])#8 is the batch size
        for single_output in predictions[image_ID % 8]:
           if(single_output > threshold):
                img_reject = image_ID
                set_of_rejected.append(img_reject)
                break
    # print("set rejected {}".format(set_of_rejected))
    return set_of_rejected



#combine all testing functions in to one


def accept_target_data_m1(net, loader):
    # accuracy = 0
    # epoch = 0
    threshold = 0.2
    # max_epochs = 10
    set_of_rejected = list()
    with torch.no_grad():#save memeory
        # while accuracy < 0.7 or epoch < max_epochs:
        for idx_batch, (inputs, labels, index) in enumerate(loader):
            progbar(idx_batch, len(loader.dataset)/16, 20)
            inputs = Variable(inputs)
            labels = Variable(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = net(inputs)
            #TODO: WARNING ADDED SOFTMAX TO GET PROBABLITY VALUE ONCE TRAINED ON NEW bce LOSS - THIS IS NOT NEEDE. sOFTMAX USED FOR TESTING CODE.
            # softmax = nn.Softmax(dim=1)
            # predictions = softmax(predictions)
            rejected_data = acceptance_layer(index, predictions, threshold)
            set_of_rejected.append(rejected_data)
            # print(set_of_rejected)
            if(idx_batch == len(loader.dataset)/(2*64)):
                print("... half way  done ..")
        flat_list = [item for sublist in set_of_rejected for item in sublist]
        all = np.arange(len(loader.dataset))
        accepted_data = np.setdiff1d(all, np.array(flat_list))
    torch.set_printoptions(profile='full')
    # print("accepted")
    # print(accepted_data)
    #TODO: ERROR JUT FOR THE SAKE OF TESTING WE ARE SENING REJECTED == IT SHOULD BE THE ACACEPTED LIST!!!
    # return set_of_rejected
    return accepted_data


def find_unseen_dataset(size_batch, dataset):
    model_path = "../../bloss_cifar"
    net = models.vgg16(pretrained=True)
    #todo: make functionfor this

    # net.classifier = nn.Sequential(
    #     nn.Linear(in_features=25088, out_features=4096, bias=True),
    #     nn.ReLU()
    # )

    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # train_model(net, optimizer, exp_lr_scheduler, 2)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    net.classifier[-1] = nn.Linear(in_features=4096, out_features=10) #todo: still need to train and fine-tune model
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    # dataset = UnlabelledDataset("CIFAR10", transform_train=transform)
    sample_from = np.where(np.ones(len(dataset))[0])
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=SequentialSubsetSampler(range(len(dataset))))
    # loader = torch.utils.data.DataLoader(dataset, batch_size=8,
    #                                      sampler=SequentialSubsetSampler(range(len(dataset))))
    data_inds = accept_target_data_m1(net, loader)
    print(data_inds)
    subset_loader = torch.utils.data.DataLoader(dataset, batch_size=size_batch,sampler=SubsetRandomSampler(data_inds))
    unseen_dataset = subset_loader.dataset
    print(unseen_dataset)
    return unseen_dataset

