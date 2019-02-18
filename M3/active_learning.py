import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from torch.autograd import Variable

start_time = time.time()
phases = ["train", "test"]

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

data_dir = '/home/nobelletay/centers'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms["train"]) for x in phases}
class_names = image_datasets['train'].classes
num_class = len(list(class_names))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=False, num_workers=4) for x in phases}
dataset_sizes = {x: len(image_datasets[x]) for x in phases}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, dropout=False, weightDecay=0):
        super(Net, self).__init__()
        '''
        self.fc1 = nn.Linear(11, 11) # 2 Input noses, 50 in middle layers
        self.do1 = nn.Dropout(p=0.2)
        self.rl1 = nn.Sigmoid()
        self.fc2 = nn.Linear(11, 3)
        self.do2 = nn.Dropout(p=0.2)
        self.smout = nn.Softmax(dim=1)
        '''

        self = models.vgg16(pretrained=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0)
        self.dropout=dropout

        self.cuda()

        self.trainOverTimeAccuracy = []
        self.trainOverTimeLoss = []
        self.testOverTimeAccuracy = []
        self.testOverTimeLoss = []

    def forward(self, x):
        x = self(x)
        return x

    def log(self, epoch, train, test):
        if epoch % REPORT_RATE == 0:
            self.trainOverTimeAccuracy.append(self.accuracy(train))
            #self.trainOverTimeLoss.append(self.loss(train))
            self.testOverTimeAccuracy.append(self.accuracy(test))
            #self.testOverTimeLoss.append(self.loss(test))

    def epochTrain(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        self.loss = self.criterion(outputs, torch.max(labels, 1)[1])
        self.loss.backward()
        self.optimizer.step()


    def train(self, numEpochs, train_set, train, test):
        inputs, labels = train

        inputs = Variable(torch.FloatTensor(inputs).cuda())
        labels = Variable(torch.FloatTensor(labels).cuda())

        for epoch in range(numEpochs):
            self.epochTrain(inputs, labels)
            self.log(epoch, train, test)
                
    def activetrain(self, activeUpdateRate, batchSize, numEpochs, train, test):
        trainings = int(numEpochs/activeUpdateRate)
        currentEpoch = 0

        for i in range(trainings):
            activecriterion = nn.CrossEntropyLoss(reduction='none')
            
            inputs, labels = train
            
            inputs = Variable(torch.FloatTensor(inputs).cuda())
            labels = Variable(torch.FloatTensor(labels).cuda())
            
            outputs = self(inputs)

            loss = activecriterion(outputs, torch.max(labels, 1)[1]).detach().cpu().numpy()
            sortIndexs = np.argsort(loss)[::-1]
            dynamicTrainSet = np.array(copy.deepcopy(train))[sortIndexs]
            dynamicTrainSet = dynamicTrainSet[0:batchSize]

            inputs, labels = dataloaders[dynamicTrainSet]
        
            inputs = Variable(torch.FloatTensor(inputs).cuda())
            labels = Variable(torch.FloatTensor(labels).cuda())
            
            for epoch in range(activeUpdateRate):
                currentEpoch += 1
                self.epochTrain(inputs, labels)
                self.log(currentEpoch, train, test)
            
    def randomtrain(self, batchSize, numEpochs, train, test):
        shuffledTrainSet = np.array(copy.deepcopy(train))
        for epoch in range(numEpochs):
            np.random.shuffle(shuffledTrainSet)

            inputs, labels = shuffledTrainSet[0:batchSize]

            inputs = Variable(torch.FloatTensor(inputs).cuda())
            labels = Variable(torch.FloatTensor(labels).cuda())
            
            self.epochTrain(inputs, labels)
            self.log(epoch, train, test)

    def loss(self, test_set):
        inputs, labels = test_set
        inputs = Variable(torch.FloatTensor(inputs).cuda())
        labels = Variable(torch.FloatTensor(labels).cuda())
        result = self(inputs)
        loss = self.criterion(result, torch.max(labels, 1)[1])

        return loss.item()

    def accuracy(self, test_set):
        inputs, labels = test_set
        result = self(Variable(torch.FloatTensor(inputs)).cuda())
        inputs_max = np.argmax(result.detach().cpu().numpy(), axis=1)
        labels_max = np.argmax(np.array(labels), axis=1)
        correct = np.sum(inputs_max == labels_max)

        return correct/len(test_set)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.statedict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    print(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.statedict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    if 'val' not in phases:
        return model

    # laod best model weights
    model.load_state_dict(best_model_wts)
    return model

vgg16 = Net()
vgg16.activetrain(1, 4, 100, dataloaders["train"], dataloaders["test"])

with open("train_acc.txt", "r") as f:
    for item in vgg16.trainOverTimeAccuracy:
        f.write("{}\n".format(item))

with open("train_loss.txt", "r") as f:
    for item in vgg16.trainOverTimeLoss:
        f.write("{}\n".format(item))

with open("test_acc.txt", "r") as f:
    for item in vgg16.testOverTimeAccuracy:
        f.write("{}\n".format(item))

with open("test_loss.txt", "r") as f:
    for item in vgg16.testOverTimeLoss:
        f.write("{}\n".format(item))

print('Execution time: {}s'.format(time.time() - start_time))
