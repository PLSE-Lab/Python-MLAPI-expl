#!/usr/bin/env python
# coding: utf-8

# # Indian Currency Classification
# 
# ## By: Sergei Issaev

# ### Introduction
# In this notebook I will be using image data of different types of Indian currency to build a classifier. I will be using data provided by Gaurav Sahani at https://www.kaggle.com/gauravsahani/indian-currency-notes-classifier.
# This dataset is very small by computer vision standards, with a training size of 153 images and a test size of 42. As a result, the validation error varies greatly from run to run. I have obtained 100% validation accuracy several times, but this error rate is inconsistent due to the small data size. Perhaps using external data could be a next step.

# ### Import Libraries

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


# ### Load in the Data

# In[ ]:


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
    #T.RandomCrop(256, padding=8, padding_mode='reflect'),
     #T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(), 
    T.RandomRotation(10),
    T.ToTensor(), 
     T.Normalize(*imagenet_stats,inplace=True), 
    T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
     T.Resize((256, 256)), 
    T.ToTensor(), 
     T.Normalize(*imagenet_stats)
])


# In[ ]:


dataset = ImageFolder(root='/kaggle/input/indian-currency-notes-classifier/Train/')

dataset_size = len(dataset)
dataset_size


# In[ ]:


testdataset = ImageFolder(root='/kaggle/input/indian-currency-notes-classifier/Test/', transform = valid_tfms)

testdataset_size = len(testdataset)
testdataset_size


# In[ ]:




classes = dataset.classes
classes


# In[ ]:


num_classes = len(dataset.classes)
num_classes


# ### Perform Train Validation Split

# In[ ]:


val_size = 16
train_size = len(dataset) - val_size

train_df, val_df = random_split(dataset, [train_size, val_size])
len(train_df), len(val_df)


# In[ ]:



val_df.dataset.transform = valid_tfms

train_df.dataset.transform = train_tfms


# In[ ]:




batch_size = 16

train_dl = DataLoader(train_df, batch_size, shuffle=True, 
                      num_workers=3, pin_memory=True)
val_dl = DataLoader(val_df, batch_size*2, 
                    num_workers=2, pin_memory=True)
test_dl = DataLoader(testdataset, batch_size*2, 
                    num_workers=2, pin_memory=True)


# ### Define the Models

# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[ ]:


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
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


# In[ ]:


class CnnModel2(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.wide_resnet101_2(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 7)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


# In[40]:


model = CnnModel2()


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
test_dl = DeviceDataLoader(test_dl, device)
to_device(model, device);


# In[ ]:




model = to_device(CnnModel2(), device)

for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break


# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# ### Train the Models

# In[ ]:


history = [evaluate(model, val_dl)]
history


# In[ ]:


epochs = int(np.random.choice([3, 5, 7, 9, 11, 13, 15]))
max_lr = np.random.choice([1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6])
grad_clip = np.random.choice([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
weight_decay = np.random.choice([1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5])
opt_func = torch.optim.Adam
print('epoch = ', epochs, 'lr = ', max_lr, 'grad is ', grad_clip, 'weights = ', weight_decay)


# In[ ]:


torch.cuda.empty_cache()


history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)


# In[ ]:


all = []
pop = evaluate(model, val_dl)['val_acc']
all.append(pop)
torch.cuda.empty_cache()
all


# In[ ]:



model = to_device(CnnModel2(), device)


# In[ ]:


epochs = 11
max_lr = 5e-4
grad_clip = 0.3
weight_decay = 5e-4
opt_func = torch.optim.Adam
print('epoch = ', epochs, 'lr = ', max_lr, 'grad is ', grad_clip, 'weights = ', weight_decay)


# In[ ]:



get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, \n                             grad_clip=grad_clip, \n                             weight_decay=weight_decay, \n                             opt_func=opt_func)\n')


# In[ ]:


pop = evaluate(model, val_dl)['val_acc']
all.append(pop)
all


# ### Final Results

# Two fits were made - one with randomly selected hyperparameters (1) and one with hyperparameters that have shown the ability to attain 100% accuracy for validation (2). However, due to the small dataset size these numbers are still highly variable. 

# In[ ]:


evaluate(model, val_dl)['val_acc']


# In[ ]:


evaluate(model, test_dl)['val_acc']


# Thank you for reading, and please upvote if you enjoyed! 
# ![](http://i.redd.it/dfzo0lwp49951.jpg)

# In[ ]:



