# -*- coding: utf-8 -*-
"""
Training alexnet on CIFAR10

#sources: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

"""
# pytorch installation instructions at pytorch.org
# Other dependencies in requirements.txt
from torchvision import models, transforms, datasets, utils
from torch.autograd import Variable
import torch
import torchvision
from torch import utils as u
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import alexnet as ax
import time
start = time.time()

#Get data and transform to fit alexnet ----------------------------------------
batch_size = 64

x = [] #batch number 
y = [] #accuracy

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

resize = transforms.Resize((227, 227))
transform = transforms.Compose([resize, transforms.ToTensor(), normalize])

fashion_train = datasets.CIFAR10("/home/shay/a/ighodgao/CAM2", 
        train=True, transform=transform, target_transform=None, download=True)
fashion_test = datasets.CIFAR10("/home/shay/a/ighodgao/CAM2", 
        train=False, transform=transform, target_transform=None, download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# WHOLE DATASET
# train_loader = torch.utils.data.DataLoader(dataset=fashion_train,
#                                           batch_size=batch_size,
#                                           shuffle = True)
# test_loader = torch.utils.data.DataLoader(da#taset=fashion_test,
#                                          batch_size=batch_size,
#                                          shuffle = True)
           
classes = ("plane", "automobile","bird", "cat", "deer", 
         "dog", "frog",  "horse",  "ship", "truck")
#------------------------------------------------------------------------------


num_classes = 10

#VGG

al = models.vgg16(pretrained=True)
al.classifier[-1] = nn.Linear(in_features=4096, out_features=10)
al = al.to(device)
f = open("accuracies_pretrained_vgg.txt", "w")

#RESNET
'''
al = models.resnet18(pretrained=True)
al.fc= nn.Linear(in_features= 2048, out_features=10)
al = al.to(device)
f = open("accuracies_pretrained_resnet.txt", "w")
'''

#ALEXNET
'''
al = models.alexnet(pretrained=True)
al.classifier[-1] = nn.Linear(in_features=4096, out_features=10)
al = al.to(device)
f = open("accuracies_pretrained_alexnet.txt", "w")
'''

train_loader = torch.utils.data.DataLoader(dataset=fashion_train, batch_size=batch_size, sampler = u.data.RandomSampler(fashion_train))
test_loader = torch.utils.data.DataLoader(dataset=fashion_test,
                                         batch_size=batch_size,
                                         shuffle = True)


training_dataset = enumerate(train_loader, 0)
testing_dataset = enumerate(test_loader, 0)

al.eval()
print("in Testing mode")
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = al(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100*correct/total
print(accuracy)
#------------------------------------------------------------------------------
x.append(0)
y.append(accuracy)

f.write("Batch: %d Accuracy %lf\n" %(0, accuracy))

#Training ---------------------------------------------------------------------
al.train()
print("in Training mode")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(al.parameters(), lr=0.001, momentum=0.9)


running_loss = 0.0
for i, data in enumerate(train_loader, 0):
#for num in range (0, 100):
    #i, data = next(training_dataset)
    al.train()
    print(i)
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = al(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print('Finished finetuning on one batch')
    print(loss)
    #Test accuracy of all testing images overall ----------------------------------
    al.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = al(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100*correct/total
    print("accuracy: ", accuracy)
    #------------------------------------------------------------------------------
    x.append(i+1)
    y.append(accuracy)
    f.write("Batch: %d Accuracy %lf\n" %(i+1, accuracy))
    f.flush()

end = time.time()
f.close()
print("time: ", end - start)
print(x)
print(y)
plt.plot(x, y)
plt.axis([0, len(x)-1, 0, 100])
plt.show()
plt.savefig('scratch.png')#------------------------------------------------------------------------------
