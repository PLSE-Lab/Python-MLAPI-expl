#!/usr/bin/env python
# coding: utf-8

# This tut is based on pytorch tut for transfer learning. 
# Many issues are sovled. Like correct use of data set & data loader. Usage of multiple size input. Usage of cuda. etc.
# 
# check input.

# In[ ]:


get_ipython().system('ls ../input')


# import libs

# In[ ]:


# import libs
import glob, pylab, pandas as pd
import pydicom, numpy as np
import random
import json
import time
import copy
import pydicom
import torchvision
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont

from matplotlib import patches, patheffects

from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from pathlib import Path


# create class to index dict see how to use it, https://github.com/pytorch/vision/blob/d6c7900d06c3388bf814cecbe90f91a9afecbefb/torchvision/datasets/folder.py#L54

# In[ ]:


PATH = Path('../input')
class_df = pd.read_csv(PATH/'labels.csv')
class_df.head()
class_to_idx = {x:i for i,x in enumerate(class_df.breed.unique())}
idx_to_class = {i:x for i,x in enumerate(class_df.breed.unique())}
device = torch.cuda.set_device(0)


# We create some helper func to load & show images
# We implement Dataset class to load and give image. We are centering the odd resized image and pasting on black background.

# In[ ]:


def show_img(im, figsize=None, ax=None):
    if not ax: 
        fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_outline(o, lw):
  o.set_path_effects([patheffects.Stroke(
      linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)

def black_background_thumbnail(image, thumbnail_size=(224,224)):
    background = Image.new('RGB', thumbnail_size, "black")    
    source_image = image.convert("RGB")
    source_image.thumbnail(thumbnail_size)
    (w, h) = source_image.size
    background.paste(source_image, (int((thumbnail_size[0] - w) / 2), int((thumbnail_size[1] - h) / 2) ))
    return background

class CDataset(Dataset):
    def __init__(self, ds, img_dir, class_df, class_to_idx, transform=None,): 
        self.ds = ds
        self.img_dir = img_dir
        self.class_df = class_df
        self.class_to_idx = class_to_idx
        self.transform = transform if transform else None
        
    def __len__(self): 
        return len(self.ds)
    
    def read_image(self, loc):
        img_arr = Image.open(loc.as_posix())
        return img_arr.convert('RGB')
        
    def __getitem__(self, i):
        img = self.read_image(self.ds[i])
        img = black_background_thumbnail(img)
        if self.transform:
            img = self.transform(img)
        label = self.ds[i].name.split('.')[0]
        kls = self.class_df[self.class_df['id'] == label]
        return img, self.class_to_idx[kls.iloc[0].breed]


# https://github.com/pytorch/vision/blob/master/torchvision/utils.py

# Next create data loader. We have removed normalization in transformation because transformed image would give -ve value. Normalisation values should be calculated on the dataset.

# In[ ]:


img_dir = PATH/'train'
# sample = random.sample(list(img_dir.iterdir()), 400) # sample
sample = list(img_dir.iterdir())
train, test = train_test_split(sample)
batch_size=58
sz=224

# if adding normalize with given value will cause transformed image range value in -ve to +ve.
# avoid it.
transform = transforms.Compose([
        transforms.CenterCrop(sz),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


train_ds = CDataset(train, img_dir, class_df, class_to_idx, transform=transform)
test_ds = CDataset(test, img_dir, class_df, class_to_idx, transform=transform)
train_dl = DataLoader(train_ds, batch_size=batch_size,)
test_dl = DataLoader(test_ds, batch_size=batch_size,)


# Lets plot some images

# In[ ]:


image, klass = next(iter(train_dl))
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=image[i].numpy().transpose((1, 2, 0))
    b = idx_to_class[klass[i]]
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0,0), b)
plt.tight_layout()


# Lets create a trainer

# In[ ]:


use_gpu = torch.cuda.is_available()
dataloaders = {'train': train_dl, 'val':test_dl}
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        start = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            data_loader = dataloaders[phase]
            for data in tqdm_notebook(data_loader):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda(),)
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects / len(data_loader.dataset)
            epoch_time = time.time() - start
            tqdm.write('{} Loss: {:.4f} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, epoch_acc, epoch_time // 60, epoch_time % 60))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        tqdm.write('')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# We are using resnet152. 
# We are using `requeires_grad=False` to tell trainer not to train those layers.
# Lets train it. 

# In[ ]:


model_ft = torchvision.models.resnet152(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_to_idx))
# for param in model_ft.parameters():
#     print(param.requires_grad)
model_ft = model_ft.cuda(0)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[ ]:


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)


# I stopped because validation accuracy is not improving. 

# In[ ]:




