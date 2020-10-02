#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from os.path import join, isdir
from os import listdir, makedirs, getcwd, remove
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
import random
from collections import deque

use_cuda = torch.cuda.is_available()
use_cuda


# In[ ]:


manualSeed = 999
def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
fixSeed(manualSeed)


# In[ ]:


import os

def find_classes(fullDir):
    classes = [d for d in os.listdir(fullDir) if os.path.isdir(os.path.join(fullDir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    num_to_class = dict(zip(range(len(classes)), classes))
    
    train = []
    for index, label in enumerate(classes):
      path = fullDir + label + '/'
      for file in listdir(path):
        train.append(['{}/{}'.format(label, file), label, index])
    
    df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])
    
    return classes, class_to_idx, num_to_class, df


# In[ ]:


classes, class_to_idx, num_to_class, df = find_classes("../input/train/")

X, y = df.drop('category_id', axis=1), df['category_id']

print(f'X shape: {X.shape}, y shape: {y.shape}')
X.head()


# In[ ]:


class SeedlingDataset(Dataset):
    def __init__(self, filenames, labels, root_dir, subset=False, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        fullname = join(self.root_dir, self.filenames.iloc[idx])
        image = Image.open(fullname).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels.iloc[idx]


# # Train Validation Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X, y = df.drop(['category_id', 'category'], axis=1), df['category_id']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# In[ ]:


print(f'train size: {X_train.shape}, validation size: {X_val.shape}, test size: {X_test.shape}')


# # Train and Test Methods

# In[ ]:


def train(train_loader, model, optimizer, criterion):
    model.train()
    
    losses, accuracies = [], []
    for batch_idx, (data, target) in (enumerate(train_loader)):
        correct = 0
        
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data) # forward
        loss = criterion(output, target)
        loss.backward()
        # Calculate Accuracy
        pred = output.data.max(1)[1] # max probability
        correct += pred.eq(target.data).cpu().sum() 
        accuracy = 100. * correct / len(data)
        optimizer.step()
        
        losses.append(loss.data.item())
        accuracies.append(accuracy)
        
    return np.mean(losses), np.mean(accuracies)


def test(test_loader, criterion, model):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum().item()
    
    test_loss /= len(test_loader) # loss function already averages over batch size
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def _should_early_stop(recent_validation_losses, validation_loss, early_stopping_rounds):
    recent_validation_losses.append(validation_loss)
    if early_stopping_rounds < len(recent_validation_losses):
        recent_validation_losses.popleft()
        return all(np.diff(recent_validation_losses) > 0)
    return False
    
def run_train_process(epochs, loaders, model, optimizer, criterion, early_stopping_rounds=10):
    epoch_to_results = {}
    recent_validation_losses = deque()
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train(loaders['train'], model, optimizer, criterion)
        validation_loss, validation_accuracy = test(loaders['validation'], criterion, model)
        epoch_to_results[epoch] = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_accuracy,
        }

        # Debug
        if epoch % 5 == 0:
            print('[%s] Train Accuracy: %.5f , Validation Accuracy: %.5f' % (epoch, train_accuracy, validation_accuracy))
        
        # Early Stop
        should_early_stop = _should_early_stop(recent_validation_losses, validation_loss, early_stopping_rounds)
        if should_early_stop:
            print(f'Train epoch {epoch} EARLY STOPPING - validation loss has not improved in the last {early_stopping_rounds} rounds')
            break
    
    df_epoch_to_results = pd.DataFrame.from_dict(epoch_to_results).T
    _, test_accuracy = test(loaders['test'], criterion, model)
    
    return df_epoch_to_results, test_accuracy


# # Lenet

# In[ ]:


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook


# In[ ]:


class LeNet(nn.Module):
    def __init__(self, num_classes=12, num_rgb=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_rgb, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), -1)
        out = self.fc3(out)
        return out


# # Experiment 2

# In[ ]:


learning_rate = 1e-3
lenet = LeNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)

image_size = (224, 224)
train_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor()
])
validation_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor()
])

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

batch_size = 8
loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'validation': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
epochs = 3

df_experiment2_result, test_accuracy_experiment2 = run_train_process(epochs, loaders, lenet, optimizer, criterion)
df_experiment2_result


# In[ ]:


ax = df_experiment2_result[['train_accuracy', 'validation_accuracy']].plot(figsize=(16,9),)
df_experiment2_result[['train_loss', 'validation_loss']].plot(ax=ax.twinx())
ax.set_title(f'[Experiment 2] Test Accuracy: {test_accuracy_experiment2:.2f}')


# # Experiment 3

# ## Normalize Images

# In[ ]:


image_size = (224, 224)
train_set = SeedlingDataset(
    X_train.file, y_train, '../input/train/',
    transform=transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor()]))

means = []
means_sq = []

for img, _ in tqdm_notebook(train_set):
    means.append(np.asarray(img, dtype='float32').mean(axis=(1,2)))
    means_sq.append((np.asarray(img, dtype='float32') ** 2).mean(axis=(1,2)))

mean_img = np.mean(means, axis=0)
std_img = np.sqrt(np.mean(means_sq, axis=0) - (mean_img ** 2))


# In[ ]:


mean_img, std_img


# In[ ]:


learning_rate = 1e-3
lenet = LeNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)

image_size = (224, 224)
train_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])
validation_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

batch_size = 8
loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'validation': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
epochs = 3

df_experiment3_part1_result, test_accuracy_experiment3_part1 = run_train_process(epochs, loaders, lenet, optimizer, criterion)
df_experiment3_part1_result


# In[ ]:


ax = df_experiment3_part1_result[['train_accuracy', 'validation_accuracy']].plot(figsize=(16,9),)
df_experiment3_part1_result[['train_loss', 'validation_loss']].plot(ax=ax.twinx())
ax.set_title(f'[Experiment 3 Part 1] Test Accuracy: {test_accuracy_experiment3_part1:.2f}')


# ## Augmentations

# In[ ]:


image_size = (224, 224)
batch_size = 8

train_trans = transforms.Compose([
    transforms.transforms.RandomHorizontalFlip(),
    transforms.transforms.RandomRotation(180),
    transforms.transforms.RandomVerticalFlip(),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img),
])
validation_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])

learning_rate = 1e-3
lenet = LeNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'validation': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
epochs = 3

df_experiment3_part2_result, test_accuracy_experiment3_part2 = run_train_process(epochs, loaders, lenet, optimizer, criterion)
df_experiment3_part2_result


# In[ ]:


ax = df_experiment3_part2_result[['train_accuracy', 'validation_accuracy']].plot(figsize=(16,9),)
df_experiment3_part2_result[['train_loss', 'validation_loss']].plot(ax=ax.twinx())
ax.set_title(f'[Experiment 3 Part 2] Test Accuracy: {test_accuracy_experiment3_part2:.2f}')


# # Experiment 4

# ## Change Kernel Size to 3

# In[ ]:


class LeNetWithKernelSize3(LeNet):
    def __init__(self, num_classes=12, num_rgb=3):
        super(LeNetWithKernelSize3, self).__init__(num_classes, num_rgb)
        self.conv1 = nn.Conv2d(num_rgb, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(46656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)


# In[ ]:


image_size = (224, 224)
batch_size = 8

train_trans = transforms.Compose([
    transforms.transforms.RandomHorizontalFlip(),
    transforms.transforms.RandomRotation(180),
    transforms.transforms.RandomVerticalFlip(),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img),
])
validation_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

learning_rate = 1e-3
lenet = LeNetWithKernelSize3().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

batch_size = 8
loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'validation': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
epochs = 3

df_experiment4_part1_result, test_accuracy_experiment4_part1 = run_train_process(epochs, loaders, lenet, optimizer, criterion)
df_experiment4_part1_result


# In[ ]:


ax = df_experiment4_part1_result[['train_accuracy', 'validation_accuracy']].plot(figsize=(16,9),)
df_experiment4_part1_result[['train_loss', 'validation_loss']].plot(ax=ax.twinx())
ax.set_title(f'[Experiment 4 Part 1] Test Accuracy: {test_accuracy_experiment4_part1:.2f}')


# ## Change Batch size to 32

# In[ ]:


image_size = (224, 224)
batch_size = 32

train_trans = transforms.Compose([
    transforms.transforms.RandomHorizontalFlip(),
    transforms.transforms.RandomRotation(180),
    transforms.transforms.RandomVerticalFlip(),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img),
])
validation_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

learning_rate = 1e-3
lenet = LeNetWithKernelSize3().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'validation': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
epochs = 100

df_experiment4_part2_result, test_accuracy_experiment4_part2 = run_train_process(epochs, loaders, lenet, optimizer, criterion)
df_experiment4_part2_result


# In[ ]:


ax = df_experiment4_part2_result[['train_accuracy', 'validation_accuracy']].plot(figsize=(16,9),)
df_experiment4_part2_result[['train_loss', 'validation_loss']].plot(ax=ax.twinx())
ax.set_title(f'[Experiment 4 Part 2] Test Accuracy: {test_accuracy_experiment4_part2:.2f}')


# # Experiment 5

# ## Change Activation Function to Sigmoid

# In[ ]:


class LeNetWithSigmoidActivation(LeNet):
    def forward(self, x):
        out = F.sigmoid(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.sigmoid(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        out = out.view(out.size(0), -1)
        out = self.fc3(out)
        return out


# In[ ]:


image_size = (224, 224)
batch_size = 32

train_trans = transforms.Compose([
    transforms.transforms.RandomHorizontalFlip(),
    transforms.transforms.RandomRotation(180),
    transforms.transforms.RandomVerticalFlip(),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img),
])
validation_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

learning_rate = 1e-3
lenet = LeNetWithSigmoidActivation().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'validation': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
epochs = 100

df_experiment5_part1_result, test_accuracy_experiment5_part1 = run_train_process(epochs, loaders, lenet, optimizer, criterion)
df_experiment5_part1_result


# In[ ]:


ax = df_experiment5_part1_result[['train_accuracy', 'validation_accuracy']].plot(figsize=(16,9),)
df_experiment5_part1_result[['train_loss', 'validation_loss']].plot(ax=ax.twinx())
ax.set_title(f'[Experiment 5 Part 1] Test Accuracy: {test_accuracy_experiment5_part1:.2f}')


# ## Change Optimization Function to Adam

# In[ ]:


image_size = (224, 224)
batch_size = 32

train_trans = transforms.Compose([
    transforms.transforms.RandomHorizontalFlip(),
    transforms.transforms.RandomRotation(180),
    transforms.transforms.RandomVerticalFlip(),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img),
])
validation_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

learning_rate = 1e-3
lenet = LeNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters(), lr=learning_rate)

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'validation': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
epochs = 100

df_experiment5_part2_result, test_accuracy_experiment5_part2 = run_train_process(epochs, loaders, lenet, optimizer, criterion)
df_experiment5_part2_result


# In[ ]:


ax = df_experiment5_part2_result[['train_accuracy', 'validation_accuracy']].plot(figsize=(16,9),)
df_experiment5_part2_result[['train_loss', 'validation_loss']].plot(ax=ax.twinx())
ax.set_title(f'[Experiment 5 Part 2] Test Accuracy: {test_accuracy_experiment5_part2:.2f}')


# # Experiment 8 - Transfer Learning

# In[ ]:


# Based on: https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)
print(f'Number of output fearurs of pretrained vgg is: {vgg16.classifier[6].out_features}') # 1000

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
print(f'Last layer of vgg16 has {num_features} input features.')
print(f'Removing layer: {list(vgg16.classifier.children())[-1]}')
classifier_layers = list(vgg16.classifier.children())[:-1] # Remove last layer
layer_to_add = nn.Linear(num_features, len(np.unique(y)))
print(f'Adding layer: {layer_to_add}')
classifier_layers.extend([layer_to_add]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*classifier_layers) # Replace the model classifier


# In[ ]:


image_size = (224, 224)
batch_size = 32

train_trans = transforms.Compose([
    transforms.transforms.RandomHorizontalFlip(),
    transforms.transforms.RandomRotation(180),
    transforms.transforms.RandomVerticalFlip(),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img),
])
validation_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

learning_rate = 1e-3
vgg = vgg16.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(vgg.parameters(), lr=learning_rate)

train_set = SeedlingDataset(X_train.file, y_train, '../input/train/', transform=train_trans)
valid_set = SeedlingDataset(X_val.file, y_val, '../input/train/', transform=validation_trans)
test_set = SeedlingDataset(X_test.file, y_test, '../input/train/', transform=validation_trans)

loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'validation': DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
epochs = 30

df_experiment8_result, test_accuracy_experiment8 = run_train_process(epochs, loaders, vgg, optimizer, criterion)
df_experiment8_result


# In[ ]:


ax = df_experiment8_result[['train_accuracy', 'validation_accuracy']].plot(figsize=(16,9),)
df_experiment8_result[['train_loss', 'validation_loss']].plot(ax=ax.twinx())
ax.set_title(f'[Experiment 8] Test Accuracy: {test_accuracy_experiment8:.2f}')


# # Combine Results

# In[ ]:


df_experiment2_result['experiment']       = 'experiment2'
df_experiment3_part1_result['experiment'] = 'experiment3_part1'
df_experiment3_part2_result['experiment'] = 'experiment3_part2'
df_experiment4_part1_result['experiment'] = 'experiment4_part1'
df_experiment4_part2_result['experiment'] = 'experiment4_part2'
df_experiment5_part1_result['experiment'] = 'experiment5_part1'
df_experiment5_part2_result['experiment'] = 'experiment5_part2'
df_experiment8_result['experiment']       = 'experiment8'

df_all_expirement_results = pd.concat([
    df_experiment2_result,
    df_experiment3_part1_result,
    df_experiment3_part2_result,
    df_experiment4_part1_result,
    df_experiment4_part2_result,
    df_experiment5_part1_result,
    df_experiment5_part2_result,
    df_experiment8_result
]).rename_axis('epoch').reset_index()

df_all_expirement_results


# In[ ]:


df_all_expirement_results.to_csv('df_all_expirement_results.csv', index=False, header=True)


# # Kaggle Sumbission

# In[ ]:


data_dir = '../input/'
sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

image_size = (224, 224)
test_trans = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_img, std_img)
])

kaggle_train_set = SeedlingDataset(
    sample_submission.file,
    sample_submission.species,
    '../input/test/',
    transform=test_trans)

kaggle_test_loader = DataLoader(kaggle_train_set, batch_size=8, shuffle=False, num_workers=0)

def predict(sample_submission, kaggle_test_loader, model):
    predictions = []
    for data, _ in kaggle_test_loader:
        if use_cuda:
            data = data.cuda() 
            data = Variable(data)
            output = model(data)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            predictions.extend(pred.tolist())

    df_predictions = pd.DataFrame.from_dict({
        'file': sample_submission.file,
        'species': [num_to_class[p] for p in predictions]
    })
    return df_predictions


# In[ ]:


df_predictions = predict(sample_submission, kaggle_test_loader, lenet)
df_predictions.to_csv('predictions_lenet.csv', index=False, header=True)


# In[ ]:


df_predictions = predict(sample_submission, kaggle_test_loader, vgg)
df_predictions.to_csv('predictions_vgg.csv', index=False, header=True)

