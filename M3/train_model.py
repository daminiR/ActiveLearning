import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from scipy import stats
from uncertainty_sampling import * 

start_time = time.time()

phases = ['train']

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    M2 = UncertaintySampler(sample_size=batch_size, verbose=False)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        f = open("a.out", "a")

        if 'train' in phases:
            scheduler.step()
            model.train()

            running_loss = 0.0
            running_corrects = 0

            uncertainty = M2.calculate_uncertainty(model, dataset, device)
            uncertainty = uncertainty[0:(batch_size-1)]
            uncertainty = zip(*uncertainty)
            index, _ = uncertainty

            inputs_labelled = []
            labels_labelled = []
            for ind in index:
                data, label, _ = dataset[ind]
                inputs_labelled.append(data)
                labels_labelled.append(label)
                f.write("{}\n".format(ind))
            dataset.mark(index)

            # convert list to tensor
            inputs_labelled = torch.stack(inputs_labelled)
            labels_labelled = torch.tensor(labels_labelled)

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
        f.close()
        epoch_loss = running_loss / batch_size # dataset_sizes['train']
        epoch_acc = running_corrects / batch_size # dataset_sizes['train']

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
            epoch_acc = running_corrects / dataset_sizes['val']

            print('val loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
batch_size = 4
data_dir = '/home/data/ilsvrc/ILSVRC/ILSVRC2012_Classification'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in phases}
dataset = UnlabelledDataset('CIFAR10', transforms=data_transforms['train'])
#class_names = image_datasets['train'].classes
class_names = dataset.dataset_train.classes
num_class = len(list(class_names))
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=False, num_workers=4) for x in phases}
#dataset_sizes = {x: len(image_datasets[x]) for x in phases}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vgg16 = models.vgg16(pretrained=True)
vgg16 = vgg16.to(device)

criterion = nn.CrossEntropyLoss()

# observe that all parameters are being optimized
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

# decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

vgg16 = train_model(vgg16, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

print('Execution time: {}s'.format(time.time() - start_time))
