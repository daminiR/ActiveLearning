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
from torch import utils as u
import matplotlib
matplotlib.use("webagg")
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


# WHOLE DATASET
# train_loader = torch.utils.data.DataLoader(dataset=fashion_train,
#                                           batch_size=batch_size,
#                                           shuffle = True)
# test_loader = torch.utils.data.DataLoader(da#taset=fashion_test,
#                                          batch_size=batch_size,
#                                          shuffle = True)

train_loader = torch.utils.data.DataLoader(dataset=fashion_train,
                                           batch_size=batch_size,
                                           sampler = u.data.SubsetRandomSampler(list(range(1,40000))))
test_loader = torch.utils.data.DataLoader(dataset=fashion_test,
                                          batch_size=batch_size)
#print("train loader size ", len(train_loader))
#print("test loader size ", len(test_loader))
                
classes = ("plane", "automobile","bird", "cat", "deer", 
         "dog", "frog",  "horse",  "ship", "truck")
#------------------------------------------------------------------------------


# Initialize the pre-trained model
al = ax.alexnet(pretrained=False)

#Training ---------------------------------------------------------------------
al = al.train()
print("in Training mode")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(al.parameters(), lr=0.1, momentum=1)

running_loss = 0.0
for i, data in enumerate(train_loader, 0):
    #print("i")
    inputs, labels = data
    optimizer.zero_grad()
    outputs = al(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    
print('Finished Training for Train from scratch')
# #------------------------------------------------------------------------------


train_loader = torch.utils.data.DataLoader(dataset=fashion_train,
                                           batch_size=batch_size,
                                           sampler = u.data.SubsetRandomSampler(list(range(1,10000))))

al = al.eval()
print("in Testing mode")
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = al(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100*correct/total
#------------------------------------------------------------------------------
x.append(0)
y.append(accuracy)

#Training ---------------------------------------------------------------------
al = al.train()
print("in Training mode")


running_loss = 0.0
for i, data in enumerate(train_loader, 0):
    al.train()
    inputs, labels = data
    optimizer.zero_grad()
    outputs = al(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print('Finished finetuning on one batch')

    #Test accuracy of all testing images overall ----------------------------------
    al = al.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = al(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100*correct/total
    #------------------------------------------------------------------------------
    x.append(i+1)
    y.append(accuracy)

    # #Test accuracy per class ------------------------------------------------------
#    class_correct = list(0. for i in range(10))
 #   class_total = list(0. for i in range(10))
  #  with torch.no_grad():
   #     for data in test_loader:
    #        images, labels = data
     #       outputs = al(images)
      #      _, predicted = torch.max(outputs, 1)
       #     c = (predicted == labels).squeeze()
        #    for i in range(9):
         #       label = labels[i]
          #      class_correct[label] += c[i].item()
           #     class_total[label] += 1

   # for i in range(10):
       # if class_total[i] == 0:
            #print('Accuracy of %5s : %2d %%' % (classes[i], 0))
       # else:
           # print('Accuracy of %5s : %2d %%' % (
           # classes[i], 100 * class_correct[i] / class_total[i]))
    # #------------------------------------------------------------------------------
end = time.time()
print("time: ", end - start)

plt.plot(x, y)
plt.axis([0, len(x)-1, 0, 100])
plt.show()
plt.savefig('scratch.png')#------------------------------------------------------------------------------
