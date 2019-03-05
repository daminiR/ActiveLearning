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
from M2 import UnlabelledDataset
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../CIFAR100_data', train=True, download=True,
                              transform=transforms.Compose(
                                  [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])),
            batch_size=4, shuffle=True, num_workers=5)


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
    objective_loss_function = nn.BCEWithLogitsLoss(reduction='sum')
    return objective_loss_function(predicts, y)




def acceptance_layer(idx, predictions, set_of_accepted, threshold):#happens during testnig
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
    for single_output in predictions:  # per batch
        accepted = (idx for each_pred in single_output if (each_pred <= threshold))
        set_of_accepted.append(idx)
        return set_of_accepted

#combine all testing functions in to one


def accept_target_data_m1(net, loader):
    accuracy = 0
    epoch = 0
    threshold = 0.5
    class_dominant = 0
    max_epochs = 10
    set_of_accepted = list()
    loader = torch.utils.data.DataLoader(dataset, batch_size=self.sample_size,
                                         sampler=SequentialSubsetSampler(np.where(dataset.labelled_index)[0]))
    with torch.no_grad():#save memeory
        while accuracy < 0.7 or epoch < max_epochs:
            for idx, (inputs, labels) in enumerate(loader):
                inputs = Variable(inputs)
                labels = Variable(labels)

                predictions = net(inputs)
                accepted_data = acceptance_layer(idx, predictions, set_of_accepted, threshold)
    return accepted_data



def train_model(model, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train()


        running_loss = 0.0
        running_corrects = 0

        # iterate over data
        for inputs, labels in dataloader:
            inputs = inputs
            labels = labels

            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                print(outputs)
                _, preds = torch.max(outputs, 1)
                loss = one_v_all_sigmoid_loss(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        #
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_corrects / len(dataloader)

    print('loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print('Best val acc: {:4f}'.format(best_acc))
    return model



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
    UnlabelledDataset
    input = torch.rand((2, 2))
    labels = torch.randint(2, size=(2,))
    print(input)
    print(labels)
    print(one_v_all_sigmoid_loss(input, labels))

