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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#TODO: right now figure sumation
def one_v_all_sigmoid_loss(predicts, targets):#target is (batch x num)_classes dims
    """
    Parameters
    ----------
    predicts - tensor of shape (batch, num_classes)
    targets - tensor of shape (batch, num_classes)
    num_classes -

    Returns
    -------

    """

    y = torch.zeros(predicts.shape[0], predicts.shape[1])
    #one hot encoding
    y[range(targets.shape[0]), targets] = 1
    # objective_loss_function = nn.CrossEntropyLoss()
    # loss = criterion(outputs, labels)
    objective_loss_function = nn.BCEWithLogitsLoss()
    return objective_loss_function(predicts, y.to(device))


def train_model(model,laoder,  optimizer, scheduler, num_epochs=25):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train()


        running_loss = 0.0
        running_corrects = 0

        # iterate over data
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = one_v_all_sigmoid_loss(outputs, labels)
                # criterion = nn.CrossEntropyLoss()
                # loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        #
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_corrects / len(dataloader)
        torch.save(model.state_dict(), "M1")
    print('loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print('Best val acc: {:4f}'.format(best_acc))
    return model


if __name__ == "__main__":
    net = models.vgg16(pretrained=False)

    # todo: make functionfor this

    # net.classifier = nn.Sequential(
    #     nn.Linear(in_features=25088, out_features=4096, bias=True),
    #     nn.ReLU()
    # )

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    net.classifier[-1] = nn.Linear(in_features=4096, out_features=10)  # todo: still need to train and fine-tune model
    net = net.to(device)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms)
    loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
    train_model(net, loader, optimizer, exp_lr_scheduler, 2)





