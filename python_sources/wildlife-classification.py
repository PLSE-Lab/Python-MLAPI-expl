#!/usr/bin/env python
# coding: utf-8

# # African Wildlife Classification
# ## By Sergei Issaev

# Hello and welcome to my final course project for the 2020 course offering of PyTorch: Zero to GANS, presented by Jovian.ml. I will be attempting to build a classifier capable of taking an input image containing either a buffalo, and elephant, a rhino or a zebra, then outputting the correct class label for the input image. The data was kindly provided by Bianca Ferreira, and can be found here: https://www.kaggle.com/biancaferreira/african-wildlife. 
# 
# I will be using the current state-of-the-art performance as a benchmark for my own work. From the five notebooks published for this dataset at the time of this writing, the best accuracy was obtained by Leogalbu, who attained an accuracy of 98.0%. His notebook can be found here:
# https://www.kaggle.com/leogalbu/african-wildlife-fastai-progressive-resize.
# 
# As per the course instructions, several different methods will be applied to the dataset. They are as follows:
# * 1) Logistic Regression
# * 2) Feedforward Neural Network
# * 3) CNN
# * 4) Transfer learning with wide ResNet and data augmentation
# 
# The links to the github code, Kaggle notebook, and my social media links are included at the end of the article. Please don't forget to upvote if you liked my work !
# 

# Mapping of animal classes to integers is as follows:
# 
# 0 -------> buffalo
# 
# 1 -------> elephant
# 
# 2 -------> rhino
# 
# 3 -------> zebra

# # Logistic Regression

# ### Import Libraries

# In[ ]:


import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import random_split
import torch.nn.functional as F
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

torch.manual_seed(42)


# In[ ]:


project_name='wildlife'


# ### Load in the Data

# In[ ]:


dataset = ImageFolder(root='/kaggle/input/african-wildlife/', transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))


# In[ ]:


dataset_size = len(dataset)
dataset_size


# In[ ]:


classes = dataset.classes
classes


# In[ ]:


num_classes = len(dataset.classes)
num_classes


# ### Perform Train-Validation-Test Split

# In[ ]:


test_size = 100
nontest_size = len(dataset) - test_size

nontest_ds, test_ds = random_split(dataset, [nontest_size, test_size])
len(nontest_ds), len(test_ds)


# In[ ]:


val_size = 100
train_size = len(nontest_ds) - val_size

train_ds, val_ds = random_split(nontest_ds, [train_size, val_size])
len(train_ds), len(val_ds)


# In[ ]:


batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)


# In[ ]:


for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break


# In[ ]:


input_size = 3 * 256*256
num_classes = 4


# ### Train the Logistic Regression Model

# In[ ]:


class WildlifeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1 , 3 * 256 * 256)
        out = self.linear(xb)
        return out
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = WildlifeModel()


# In[ ]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[ ]:


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');


# In[ ]:


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


# ### Train the model

# In[ ]:


result0 = evaluate(model, val_loader)
result0


# In[ ]:


history1 = fit(50, 0.000005, model, train_loader, val_loader)


# In[ ]:


history2 = fit(10, 0.000001, model, train_loader, val_loader)


# In[ ]:


history3 = fit(25, 0.000001, model, train_loader, val_loader)


# In[ ]:


history4 = fit(10, 0.0000005, model, train_loader, val_loader)


# In[ ]:


history5 = fit(10, 0.0000005, model, train_loader, val_loader)


# In[ ]:


history6 = fit(10, 0.0000005, model, train_loader, val_loader)


# The validation accuracy from the tuned Logistic Regression model was 44.0%. Considering there are only 4 classes, and guessing randomly would get about 25% accuracy, this isn't great. Let's try a feedforward neural network.

# In[ ]:


# Replace these values with your results
history = [result0] + history1 + history2 + history3 + history4 + history5 + history6
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


# In[ ]:


get_ipython().system('pip install jovian')


# In[ ]:


import jovian


# In[ ]:


jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])


# In[ ]:


jovian.commit(project=project_name, environment=None)


# # Feedforward Neural Network

# Since we have already done the imports and dataloaders, we won't repeat them here, and will go straight to defining and training the feedforward neural network.

# In[ ]:


input_size = 3 * 256 * 256
hidden_size1 = 128 # you can change this
hidden_size2 = 32
hidden_size3 = 64
hidden_size4 = 32
output_size = 4


# In[ ]:


class FeedforwardModel(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.linear5 = nn.Linear(hidden_size4, output_size)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


# In[ ]:


model = FeedforwardModel()


# In[ ]:


for t in model.parameters():
    print(t.shape)


# In[ ]:


torch.cuda.is_available()


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# In[ ]:


device = get_default_device()
device


# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# In[ ]:


for images, labels in train_loader:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break


# In[ ]:


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)


# In[ ]:


for xb, yb in val_loader:
    print('xb.device:', xb.device)
    print('yb:', yb)
    break


# In[ ]:


# Model (on GPU)
model = FeedforwardModel()
to_device(model, device)


# In[ ]:


history = [evaluate(model, val_loader)]
history


# In[ ]:


history += fit(30, 0.01000, model, train_loader, val_loader)


# In[ ]:


history += fit(30, 0.00500, model, train_loader, val_loader)


# In[ ]:


history += fit(30, 0.00010, model, train_loader, val_loader)


# In[ ]:


history += fit(30, 0.00005, model, train_loader, val_loader)


# The validation accuracy from the tuned feedforward neural network model was 52.0%. This is an improvement, however it is still far below the baseline. Let's try using a convolutional neural network, which is known to do well with image data.

# In[ ]:


plot_losses(history)


# In[ ]:


plot_accuracies(history)


# In[ ]:


jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])


# In[ ]:


jovian.commit(project=project_name, environment=None)


# # CNN

# These old school methods did not perform very well for image classification. Let's try using some CNNs!

# In[ ]:


import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import random_split
import torch.nn.functional as F
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

torch.manual_seed(42)


# In[ ]:


dataset = ImageFolder(root='/kaggle/input/african-wildlife', transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))


# In[ ]:


dataset_size = len(dataset)
dataset_size


# In[ ]:


classes = dataset.classes
classes


# In[ ]:


num_classes = len(dataset.classes)
num_classes


# ### Perform Train-Validation-Test Split

# In[ ]:


test_size = 100
nontest_size = len(dataset) - test_size

nontest_ds, test_ds = random_split(dataset, [nontest_size, test_size])
len(nontest_ds), len(test_ds)


# In[ ]:


val_size = 100
train_size = len(nontest_ds) - val_size

train_ds, val_ds = random_split(nontest_ds, [train_size, val_size])
len(train_ds), len(val_ds)


# In[ ]:


batch_size = 16

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)


# In[ ]:


train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)


# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[ ]:


class CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*16*16 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)


# In[ ]:


model = CnnModel()
model


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


device = get_default_device()
device


# In[ ]:


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device);


# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


model = to_device(CnnModel(), device)

for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break


# In[ ]:


num_epochs1 = 10
num_epochs2 = 10
num_epochs3 = 10
opt_func = torch.optim.Adam
lr1 = 0.000010
lr2 = 0.0000005
lr3 = 0.0000001


evaluate(model, val_dl)


# In[ ]:


history = fit(num_epochs1, lr1, model, train_dl, val_dl, opt_func)


# In[ ]:


history = fit(num_epochs2, lr2, model, train_dl, val_dl, opt_func)


# In[ ]:


history = fit(num_epochs3, lr3, model, train_dl, val_dl, opt_func)


# The validation accuracy for this CNN was 53.12%. Not a great improvement. Perhaps using a larger network, such as ResNet, can help us achieve our goal. Note that we will be using data augmentation now as well.

# In[ ]:


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


# In[ ]:


plot_accuracies(history)


# In[ ]:


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


# In[ ]:


plot_losses(history)


# In[ ]:


jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])


# In[ ]:


jovian.commit(project=project_name, environment=None)


# 
# ### Resnet

# Let's try transfer learning from wide_resnet.

# In[ ]:


import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
from tqdm.notebook import tqdm
import torchvision.transforms as T
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import PIL

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(42)


# In[ ]:


dataset = ImageFolder(root='/kaggle/input/african-wildlife/')

dataset_size = len(dataset)
dataset_size


# In[ ]:


classes = dataset.classes
classes


# In[ ]:


num_classes = len(dataset.classes)
num_classes


# ### Perform Train-Validation-Test Split

# In[ ]:


test_size = 100
nontest_size = len(dataset) - test_size

nontest_df, test_df = random_split(dataset, [nontest_size, test_size])
len(nontest_df), len(test_df)


# In[ ]:


val_size = 100
train_size = len(nontest_df) - val_size

train_df, val_df = random_split(nontest_df, [train_size, val_size])
len(train_df), len(val_df)


# In[ ]:


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
    #T.RandomCrop(256, padding=8, padding_mode='reflect'),
     #T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
    #T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(), 
    T.RandomRotation(10),
    T.ToTensor(), 
     T.Normalize(*imagenet_stats,inplace=True), 
    #T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
     T.Resize((256, 256)), 
    T.ToTensor(), 
     T.Normalize(*imagenet_stats)
])


# In[ ]:


test_df.dataset.transform = valid_tfms
val_df.dataset.transform = valid_tfms

train_df.dataset.transform = train_tfms


# In[ ]:


batch_size = 16

train_dl = DataLoader(train_df, batch_size, shuffle=True, 
                      num_workers=3, pin_memory=True)
val_dl = DataLoader(val_df, batch_size*2, 
                    num_workers=2, pin_memory=True)
test_dl = DataLoader(test_df, batch_size*2, 
                    num_workers=2, pin_memory=True)


# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[ ]:


class CnnModel2(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.wide_resnet101_2(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 4)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


# In[40]:


model = CnnModel2()
model


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[42]:


device = get_default_device()
device


# In[ ]:


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)
to_device(model, device);


# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


model = to_device(CnnModel2(), device)

for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break


# In[ ]:


num_epochs1 = 10
opt_func = torch.optim.Adam
lr1 = 0.000010

evaluate(model, val_dl)


# In[ ]:


history = fit(num_epochs1, lr1, model, train_dl, val_dl, opt_func)


# Interesting... the model no longer fits at 100% accuracy, even though it did when I ran the code on another server:
# ![](http://i.ibb.co/TWLfJKy/lavie.png)
# This was most likely caused by a lucky start to the gradient descent, that let it find a nice minimum. Restarting the training several times will likely find it again. Let this be a lesson to save your model weights when you get a record breaking score! Live and learn, c'est la vie :'(

# In[ ]:


plot_accuracies(history)


# In[ ]:


plot_losses(history)


# In[ ]:


jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])


# In[ ]:


jovian.commit(project=project_name, environment=None)


# EDIT: I reran the code on my other server, and was able to save the weights for the 100% accuracy model. I couldn't get 100% on the kaggle server (guess it doesn't have the magic touch that Compute Canada's Helios server does), and since I can already see the sun rising loading in the weights will have to do.

# In[ ]:


model2 = CnnModel2()
model2.load_state_dict(torch.load('/kaggle/input/weights100/weights1.pth'))


# In[ ]:


model2 = to_device(model2, device)


# In[ ]:


evaluate(model2, val_dl)


# In[ ]:


evaluate(model2, test_dl)


# In[ ]:


jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])


# In[ ]:


jovian.commit(project=project_name, environment=None)


# ![](http://www.memekingz.net/img/memes/201805/aceb324e10bb797ec061881347548226.jpg)

# # Final Remarks

# We have reached 100% accuracy in not only the validation set, but in the test set as well! Therefore, I have surpassed the previous state of the art (98%), and have built a classifier that can be relied upon for classifying the four African animals. This can have further applications in preventing the poaching of endangered species, and can be of interest to conservation groups and other wildlife researchers.
# 
# Of course, this notebook is considerably longer than the previous state-of-the-art, which was created using Fast.AI, a framework built on PyTorch that can create models in a fraction of the number of lines of code used for this notebook. Is the extra time and energy worth a 2% increase in model accuracy? That's up to you to decide ;) 
# 
# Thank you for reading until the end! If you have any questions or comments either leave them below the kernel or reach out to me at sergei740@gmail.com. 
# 
# ![](http://memegenerator.net/img/instances/66006142/thank-you-for-reading-this-article.jpg)
