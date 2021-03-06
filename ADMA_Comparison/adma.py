import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from scipy import stats

start_time = time.time()

phases = ['train','val']
batch_size = 4

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def hook_centers_a(module, inputs, outputs):
    centers_a.append(outputs.view(-1))


def hook_centers_b(module, inputs, outputs):
    centers_b.append(outputs.view(-1))


def hook_instances_a(module, inputs, outputs):
    instances_a.copy_(outputs.view(list(outputs.size())[0], -1).unsqueeze(1))


def hook_instances_b(module, inputs, outputs):
    instances_b.copy_(outputs.view(list(outputs.size())[0], -1).unsqueeze(1))


def extract_feature(model, inputs, layer_ind):
    classifier = list(model.classifier.children())[:-1]
    model.classifier = nn.Sequential(*classifier)
    modules = modules + list(model.classifier)

    out = modules[layer_ind](inputs)
    print(out.size())
    return inputs


def modify_model(model, layer_ind=-1, module='classifier'):
    if module == 'classifier':
        new_model = type(model)(model.features)
        state = model.state_dict()
        new_model.load_state_dict(model.state_dict())
        new_classifier = nn.Sequential(*list(new_model.classifier.children())[:layer_ind])
        new_model.classifier = new_classifier
    elif module == 'features':
        new_model = type(model)(model.features)
        state = model.state_dict()
        new_model.load_state_dict(model.state_dict())
        new_features = nn.Sequential(*list(new_model.features.children())[:layer_ind])
        new_model.features = new_features
    else:
        return model
    new_model.eval()
    for param in new_model.parameters():
        param.requires_grad = False
    return new_model, state


def compute_mean(model, dataloader, num_class, layer_ind=-1):
    #    classifier = list(model.classifier.children())[:-1]
    #    model.classifier = nn.Sequential(*classifier)

    #    img_size = (batch_size, 3, 224, 224)
    #    inputs = torch.randn(img_size)
    #    features = extract_feature(model, inputs, layer_ind)
    model.eval()
    feature_size = model.classifier[layer_ind].out_features
    #    mean_feature_size = torch.mean(features, 0).size()
    zero = torch.zeros(feature_size)
    zero = zero.to(device)
    sums = {x: zero for x in range(num_class)}
    num_images = {x: 0 for x in range(num_class)}
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        features = model(inputs)
        while len(features) != 0:
            class_ind = int(labels[0])
            same_class = (labels == class_ind)
            diff_class = (labels != class_ind)
            cur_features = features[same_class]
            sums[class_ind] += torch.sum(cur_features, 0)
            num_images[class_ind] += len(cur_features)
            features = features[diff_class]
            labels = labels[diff_class]
    # compute the mean of features for every class
    means = {x: torch.div(sums[x], num_images[x]) for x in range(num_class)}
    return means


def find_center(model, dataloader, num_class, class_names, means):
    model.eval()
    img_size = (3, 224, 224)
    min_dist = {x: float("inf") for x in range(num_class)}
    # ind = {x: 0 for x in range(num_class)}
    center_img = {x: torch.zeros(img_size) for x in range(num_class)}
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        features = model(inputs)
        while len(features) != 0:
            class_ind = int(labels[0])
            same_class = (labels == class_ind)
            diff_class = (labels != class_ind)
            cur_features = features[same_class]
            cur_inputs = inputs[same_class]
            # calculate the distances between features of instances and mean
            dist = torch.sum((cur_features - means[class_ind]) ** 2, 1)
            # determine the minimum distance in the batch and its index
            min_batch_dist, min_ind = torch.min(dist, 0)
            # update the instance that has minimum distance and its distance value
            if min_dist[class_ind] > min_batch_dist:
                min_dist[class_ind] = min_batch_dist
                # center[class_ind] = cur_features[min_ind]
                center_img[class_ind] = cur_inputs[min_ind]
                # torchvision.utils.save_image(inv_normalize(center_img[class_ind]), '{}_center{}.jpg'.format(class_names[class_ind], ind[class_ind]))
                # ind[class_ind] += 1
            features = features[diff_class]
            inputs = inputs[diff_class]
            labels = labels[diff_class]
    return center_img

def sortFunction(val):
    return val[0]

def train_model(model, criterion, optimizer, scheduler,distList, num_epochs=25):
    since = time.time()

    for phase in ['train']:
      if phase=='train':
          scheduler.step()
          model.train()
      else:
          model.eval()

      running_loss = 0.0
      running_corrects = 0
      trainIterations = 0
      lambdac= 0.8
      storeImages = []
      trainIlist = []
      runningAccList = []
        #repeat until certain training loss or accuracy is reached? 

        #select active learning instances using critertion score 
      if phase == 'train':
          for inputs,labels in dataloaders['train']:
              #print('thisis how it should look')
              #print(inputs)
              #print(labels)
              for ind in range(batch_size):
                  storeImages.append((inputs[ind],labels[ind]))

          #train until accuracy of training is <= 0.7
          running_acc = 0.0 
          while(running_acc <= 0.7):
              #calculate critertion score after every you train with a batch:

              criterionScores = []


              for imageNo in range(len(storeImages)):
                  distinctiveness = distList[imageNo]
                  imagecriterionScore = (1 - lambdac*trainIterations*distinctiveness,imageNo)
                  criterionScores.append(imagecriterionScore)
                                    
              #sort criterionScore list by first val
              criterionScores.sort(key = sortFunction)

              #choose the last batch_size in the training set 
              activeLearningInputs = []
              activeLearningLabels = []
              for ind in range(batch_size):
                  _, alImageNo = criterionScores.pop()
                  alInput,alLabel = storeImages.pop(alImageNo)
                  #print(alInput)
                  #print(alLabel)
                  activeLearningInputs.append(alInput)
                  activeLearningLabels.append(alLabel)
              
              #train the data:
              trainIterations = trainIterations+1 

              activeLearningInputsT = torch.stack(activeLearningInputs)
              #print(activeLearningInputsT)
              #print(device); 
              activeLearningInputsT = activeLearningInputsT.to(device) 
              activeLearningLabels = torch.stack(activeLearningLabels)
              activeLearningLabels =  activeLearningLabels.to(device)


              # zero the parameter gradients
              optimizer.zero_grad()

              # forward
              # track history if only in train
              with torch.set_grad_enabled(phase == 'train'):
                  outputs = model(activeLearningInputsT)
                  #print(outputs)
                  _, preds = torch.max(outputs, 1)
                  loss = criterion(outputs, activeLearningLabels)

              loss.backward()
              optimizer.step()

              # statistics
              running_loss += loss.item() * inputs.size(0)
              running_corrects += torch.sum(preds == activeLearningLabels.data)
              running_acc = running_corrects/trainIterations*batch_size
              print(running_acc)
              print(trainIterations)
              trainIlist.append(trainIterations)
              runningAccList.append(running_acc)

            #statistic outside while loop 
          epoch_loss = running_loss / dataset_sizes[phase]
          epoch_acc = running_corrects / dataset_sizes[phase]

          print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
          print()

          #plotting training accuracy graph:
          fig = plt.figure()
          plt.plot(trainIlist,runningAccList)
          plt.ylabel('Training Accuracy')
          plt.xlabel('Batch No')
          plt.title('Training Accuracy after a each batch')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # return model
    return model

# inverse normalization
inv_normalize = transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     [1 / 0.229, 1 / 0.224, 1 / 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

}

data_dir_pretrained = '/home/min/a/nrajanee/centers'
image_datasets_pretrained = {x: datasets.ImageFolder(os.path.join(data_dir_pretrained, x), data_transforms[x]) for x in ['train']}
class_names_pretrained = image_datasets_pretrained['train'].classes
num_class_pretrained = len(list(class_names_pretrained))
dataloaders_pretrained = {
x: torch.utils.data.DataLoader(image_datasets_pretrained[x], batch_size=batch_size, shuffle=False, num_workers=4) for x
in ['train']}
dataset_sizes_pretrained = {x: len(image_datasets_pretrained[x]) for x in ['train']}

# data_dir = 'voc'
data_dir = '/home/min/a/nrajanee/CAM2ActiveLearning/data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in phases}
class_names = image_datasets['train'].classes
num_class = len(list(class_names))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=False, num_workers=4) for x in
               phases}
dataset_sizes = {x: len(image_datasets[x]) for x in phases}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vgg16_pretrained = models.vgg16(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

for param in vgg16_pretrained.parameters():
    param.requires_grad = False

vgg16_pretrained = vgg16_pretrained.to(device)
vgg16_pretrained.eval()
vgg16 = vgg16.to(device)

criterion = nn.CrossEntropyLoss()

# observe that all parameters are being optimized
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

# decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# vgg16 = train_model(vgg16, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

# means = compute_mean(vgg16_pretrained, dataloaders_pretrained['train'], num_class_pretrained)
# center_img = find_center(vgg16_pretrained, dataloaders_pretrained['train'], num_class_pretrained, class_names_pretrained, means)
center_img = {}
for data in dataloaders_pretrained['train']:
    inputs, labels = data
    for ind in range(list(inputs.size())[0]):
        center_img[int(labels[ind])] = inputs[ind]

centers_a = []
centers_b = []
find_centers_a = vgg16_pretrained.features[-3].register_forward_hook(hook_centers_a)
find_centers_b = vgg16_pretrained.classifier[-4].register_forward_hook(hook_centers_b)

for ind in range(num_class_pretrained):
    # save center images
    # torchvision.utils.save_image(inv_normalize(center_img[ind]), '{}_center.jpg'.format(class_names[ind]))

    # extract outputs of center images at layer A and B as the landmarks
    center = center_img[ind].to(device).unsqueeze(0)
    vgg16_pretrained(center)

find_centers_a.remove()
find_centers_b.remove()

# convert list to tensor
centers_a = torch.stack(centers_a)
centers_b = torch.stack(centers_b)

relatives_a = []
relatives_b = []

for ind in range(num_class_pretrained):
    # compute relative representations at layer A and B based on the landmarks
    # calculate the distance between representation of center image at specific layer of one class and that of all classes
    relative_a = torch.sum((centers_a[ind] - centers_a) ** 2, 1)
    relative_b = torch.sum((centers_b[ind] - centers_b) ** 2, 1)
    relatives_a.append(relative_a)
    relatives_b.append(relative_b)

# convert list to tensor
relatives_a = torch.stack(relatives_a)
relatives_b = torch.stack(relatives_b)

# compute feature transformation pattern between each class from layer A to B
patterns_ab = relatives_a - relatives_b

instances_a = torch.zeros((4, 1, 100352))
instances_b = torch.zeros((4, 1, 4096))
find_instances_a = vgg16_pretrained.features[-3].register_forward_hook(hook_instances_a)
find_instances_b = vgg16_pretrained.classifier[-4].register_forward_hook(hook_instances_b)

print('Distinctiveness')

# print(patterns_ab.size())
# print()

# TODO: set the condition of getting out of while loop
while (1):
    distList = [0]*len(image_datasets['train'])
    imageNo = 0
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        weights = vgg16_pretrained(inputs)
        instances_a = instances_a.to(device)
        instances_b = instances_b.to(device)
        relatives_instances_a = torch.sum((instances_a - centers_a) ** 2, 2)
        relatives_instances_b = torch.sum((instances_b - centers_b) ** 2, 2)
        patterns_instances_ab = relatives_instances_a - relatives_instances_b
        # print(patterns_instances_ab.size())
        # print()
        weights.transpose_(0, 1)
        approx_patterns_instances_ab = patterns_ab @ weights
        approx_patterns_instances_ab.transpose_(0, 1)
        # print(approx_patterns_instances_ab.size())
        for ind in range(batch_size):
            tau, _ = stats.kendalltau(patterns_instances_ab[ind].cpu(), approx_patterns_instances_ab[ind].cpu())
            distinctiveness = (1 - tau) / 2
            distList[imageNo] = distinctiveness
            imageNo = imageNo + 1
            print(distinctiveness)


            # TODO: calculate uncertainty and criterion score, then select the instances with highest criterion score to train the model continuously
            # printing distinctiveness to stdout to analyze the metrics
    break

train_model(vgg16,criterion,optimizer,exp_lr_scheduler,distList,num_epochs = 1)
print('Execution time: {}s'.format(time.time() - start_time))


