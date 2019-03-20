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

from M2.uncertainty_sampling import UnlabelledDataset, SequentialSubsetSampler
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../CIFAR100_data', train=True, download=True,
                              transform=transforms.Compose(
                                  [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])),
            batch_size=4, shuffle=True, num_workers=5)




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
    set_of_rejected = list()
    #for each image
    for image_ID in idx:
        #for each nueron
        print(predictions[image_ID])
        for single_output in predictions[image_ID]:
           if(single_output > threshold):
                img_reject = image_ID
                set_of_rejected.append(img_reject)
                break
    print("set rejected {}".format(set_of_rejected))
    return set_of_rejected



#combine all testing functions in to one


def accept_target_data_m1(net, loader):
    accuracy = 0
    epoch = 0
    threshold = 0.2
    max_epochs = 10
    set_of_rejected = list()
    with torch.no_grad():#save memeory
        while accuracy < 0.7 or epoch < max_epochs:
            for idx_batch, (inputs, labels, index) in enumerate(loader):
                inputs = Variable(inputs)
                labels = Variable(labels)
                predictions = net(inputs)
                #TODO: WARNING ADDED SOFTMAX TO GET PROBABLITY VALUE ONCE TRAINED ON NEW bce LOSS - THIS IS NOT NEEDE. sOFTMAX USED FOR TESTING CODE.
                softmax = nn.Softmax(dim=1)
                predictions = softmax(predictions)
                rejected_data = acceptance_layer(index, predictions, threshold)
                set_of_rejected.append(rejected_data)
                break #temp to check
            all = np.arange(len(loader))
            accepted_data = np.setdiff1d(all, np.array(rejected_data))
            break
    # return accepted_data





if __name__ == "__main__":
    net = models.vgg16(pretrained=True)

    #todo: make functionfor this

    # net.classifier = nn.Sequential(
    #     nn.Linear(in_features=25088, out_features=4096, bias=True),
    #     nn.ReLU()
    # )

    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # train_model(net, optimizer, exp_lr_scheduler, 2)
    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    net.classifier[-1] = nn.Linear(in_features=4096, out_features=10) #todo: still need to train and fine-tune model
    UnlabelledDataset
    input = torch.rand((2, 2))
    labels = torch.randint(2, size=(2,))
    print(input)
    print(labels)
    print(one_v_all_sigmoid_loss(input, labels))
    dataset = UnlabelledDataset("CIFAR10", transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=SequentialSubsetSampler(np.where(dataset.labelled_index)[0]))

    accept_target_data_m1(net, loader)


