
from __future__ import print_function,division 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])


data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data_dir = 'data/hymenoptera_data'
test_set = {x: datasets.ImageFolder(os.path.join(test_data_dir, x),
                                          test_transforms)
                   for x in ['test']
                  }

test_dataloader = {x: torch.utils.data.DataLoader(test_set[x], batch_size=4,
                                             shuffle=False, num_workers=4)
              for x in ['test']}
test_dataset_sizes = {x: len(test_set[x]) for x in ['test']}
test_class_names = test_set['test'].classes
print(test_class_names)
print(test_dataset_sizes)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            test_acc_cont  = []
            batch_cont = []
            batchNo = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                #calculate test accuracy after calculating of each batch. 

                if epoch < num_epochs - 3 and phase == 'train':

                    test_acc = test_model(model) #calling test_model function every time a batch in training data set is used for training. 
                    test_acc_cont.append(test_acc)
                    batchNo = batchNo + 1
                    batch_cont.append(batchNo) #current batch number 

                    print(test_acc)

                    #plotting the test accuracy for a batch no for the epochs we want. 

            if epoch < num_epochs - 3 and phase == 'train': 
                #plot x and y 
                print('len')
                print(len(batch_cont))
                print(len(test_acc_cont))
                fig = plt.figure()
                plt.plot(batch_cont,test_acc_cont)
                plt.ylabel('Accuracy')
                plt.xlabel('BatchNo')
                plt.title('Epoch #')



            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                print('validation phase')
                #for i in range(len(class_names)):
                    #accuracyClass[i] = 100 * (running_correctClass[i]/class_total[i]) 
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    #for i in range(len(class_names)):
        #print(accuracyClass[i])

    # load best model weights
    model.load_state_dict(best_model_wts)

    # #plot bar graph accuracy vs labels
    # fig2 = plt.figure()
    # index = np.arange(len(class_names))
    # plt.bar(index,accuracyClass)
    # plt.xlabel('Classes')
    # plt.ylabel('Best Val Accuracy (%)')
    # plt.yticks(np.arange(0,100, step = 5))
    # plt.xticks(index,class_names)
    # plt.title('Class vs Accuracy')
    # #plt.ioff()
    # #plt.show()

    return model


def test_model(model):
    model.eval()
    test_acc = 0.0
    total_test = 0

    for i, (inputs, labels) in enumerate(test_dataloader['test']): #going through all the data in the test data folder and calculating the accuracy for the current model wieghts. 

        inputs = inputs.to(device)
        labels = labels.to(device)


        # Predict classes using images from the test set
        outputs = model(inputs)
        _, prediction = torch.max(outputs.data, 1)
        
        test_acc += torch.sum(prediction == labels.data) #prediction == labels.data gives a tensor of batch_size with 0's and 1's, 1 for when condn is true and then when you do torch.sum you get the total correct values. 
       # print(test_acc)
        total_test = total_test+1*inputs.size()[0] #calculating the total number of images in test data set == len(test_set)
        #print(total_test)

    # Compute the average acc and loss over all 10000 test images
    #print('Average Loss:')
    test_acc = (test_acc.double() / total_test)*100
    #print('Best test Acc: {:4f}'.format(test_acc))

    return test_acc


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Without freezing some of the layers:
#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
#visualize_model(model_ft)

#model_conv = torchvision.models.resnet18(pretrained=True)
model_conv = torchvision.models.alexnet(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default

#ALEXANET
num_ftrs = model_conv.classifier[6].in_features
model_conv.classifier[6] = nn.Linear(num_ftrs,2)

for param in model_conv.classifier[6].parameters():
    param.requires_grad = True

#RESNET
#num_ftrs = model_conv.fc.in_features
#model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.

#RESNET
#optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

#ALEXANET

optimizer_conv = optim.SGD(model_conv.classifier[6].parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#Freezing all layers except the last layer:
model_conv = train_model(model_conv, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=5)

plt.ioff()
plt.show()
visualize_model(model_conv)

plt.ioff()
plt.show()

