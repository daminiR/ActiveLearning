import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import numpy as np
from scipy import stats
from uncertainty_sampling import *
from klDivergence import *

start_time = time.time()

phases = ['train', 'test']

def train_model(model, criterion, optimizer, scheduler, num_epochs=50, num_iterations=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    M2 = UncertaintySampler(sample_size=batch_size, verbose=False)

    f = open("a.out", "w")
    f.write("loss (every 20 instances), acc (every 20 instances)\n")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if 'train' in phases:
            scheduler.step()
            #model.train()

            running_loss = 0.0
            running_corrects = 0

            for iteration in range(num_iterations):
                uncertainty = M2.calculate_uncertainty(model, dataset, device)
                uncertainty = uncertainty[0:(2 * batch_size)]
                uncertainty = zip(*uncertainty)
                index, val = uncertainty
                _, pred = zip(*val)
                
                batch_index = calculate_KL_batch(pred, batch_size, device)
                index = np.array(index)[np.array(list(batch_index))]
                
                inputs_labelled = []
                labels_labelled = []
                for ind in index:
                    data, label, _ = dataset[ind]
                    inputs_labelled.append(data)
                    labels_labelled.append(label)
                dataset.mark(index)

                # convert list to tensor
                inputs_labelled = torch.stack(inputs_labelled)
                labels_labelled = torch.tensor(labels_labelled)

                model.train()

                inputs = inputs_labelled.to(device)
                labels = labels_labelled.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (num_iterations * batch_size) # dataset_sizes['train']
            epoch_acc = running_corrects.float() / (num_iterations * batch_size) # dataset_sizes['train']

            print('train loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if 'val' in phases:
            model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes['val']
            epoch_acc = running_corrects.float() / dataset_sizes['val']

            print('val loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if 'test' in phases:
            model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloader_test:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size_test
            epoch_acc = running_corrects.float() / dataset_size_test

            print('test loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))
            f.write("{:.4f}, {:.4f}\n".format(epoch_loss, epoch_acc))

        print()

    f.close()

    f = open("b.out", "w")
    for i in np.where(dataset.labelled_index == 0)[0]:
        f.write("{}\n".format(i))
    f.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    if 'val' not in phases:
        return model

    # laod best model weights
    model.load_state_dict(best_model_wts)
    return model

# inverse normalization
inv_normalize = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Imagenet
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # CIFAR10
    ]),

    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Imagenet
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # CIFAR10
    ])
}
batch_size = 128
#data_dir = '/home/data/ilsvrc/ILSVRC/ILSVRC2012_Classification'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in phases}
dataset = UnlabelledDataset('CIFAR10', transform_train=data_transforms['train'], transform_test=data_transforms['test'])
#class_names = image_datasets['train'].classes
class_names = dataset.dataset_train.classes
num_class = len(list(class_names))
dataloader_test = torch.utils.data.DataLoader(dataset.dataset_test, batch_size = batch_size, shuffle=False)
dataset_size_test = len(dataset.dataset_test)
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=False, num_workers=4) for x in phases}
#dataset_sizes = {x: len(image_datasets[x]) for x in phases}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vgg16 = models.vgg16(pretrained=True)

ct = 0
for child in vgg16.children():
    ct += 1
    if ct < 17:
        for param in child.parameters():
            param.requires_grad = False

num_ftrs = vgg16.classifier[-1].in_features
vgg16.classifier[-1] = nn.Linear(num_ftrs, num_class)

vgg16 = vgg16.to(device)

criterion = nn.CrossEntropyLoss()

# observe that all parameters are being optimized
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

# decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

vgg16 = train_model(vgg16, criterion, optimizer, exp_lr_scheduler, num_epochs=10)

print('Execution time: {}s'.format(time.time() - start_time))
