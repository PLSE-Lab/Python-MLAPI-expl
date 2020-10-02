#!/usr/bin/env python
# coding: utf-8

# # Dog Breed Classification

# In this project, we will try to classify 120 different dog species from over 10,000 images
# 
# We run a resnet34 model using Pytorch and achieve 75+% accuracy in around 30 minutes of training
# 
# I detail out the steps and try to define the steps. 
# 
# Hope this helps!

# # Import Libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import cv2
import random
from random import randint
import time


import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from scipy import ndimage

import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder

from tqdm.notebook import tqdm

from sklearn.metrics import f1_score


# ## Preparing the Data

# Define the data directories

# In[ ]:


DATA_DIR = '../input/dog-breed-identification'


TRAIN_DIR = DATA_DIR + '/train'                           
TEST_DIR = DATA_DIR + '/test'                             

TRAIN_CSV = DATA_DIR + '/labels.csv'                     
TEST_CSV = DATA_DIR + '/submission.csv' 


# Read the files

# In[ ]:


data_df = pd.read_csv(TRAIN_CSV)
data_df.head(10)


# Create a label dictionary

# In[ ]:


labels_names=data_df["breed"].unique()
labels_sorted=labels_names.sort()

labels = dict(zip(range(len(labels_names)),labels_names))
labels 


# Add the numberical labels and path to the dataframe

# In[ ]:



lbl=[]
path_img=[]

for i in range(len(data_df["breed"])):
    temp1=list(labels.values()).index(data_df.breed[i])
    lbl.append(temp1)
    temp2=TRAIN_DIR + "/" + str(data_df.id[i]) + ".jpg"
    path_img.append(temp2)

data_df['path_img'] =path_img  
data_df['lbl'] = lbl

data_df.head()


# Lets check the number of files and classes (dog breeds) in the dataset

# In[ ]:


num_images = len(data_df["id"])
print('Number of images in Training file:', num_images)
no_labels=len(labels_names)
print('Number of dog breeds in Training file:', no_labels)


# Are images equally distributed between all dog breeds?
# 
# Let's plot a graph and see!

# In[ ]:


bar = data_df["breed"].value_counts(ascending=True).plot.barh(figsize = (30,120))
plt.title("Distribution of the Dog Breeds", fontsize = 20)
bar.tick_params(labelsize=16)
plt.show()


# In[ ]:


data_df["breed"].value_counts(ascending=False)


# We observe that the distribution is not equal. Scottish deerhound has 126 images while eskimo dog and briard breeds have 66 images

# # Image Analysis

# Let us display 20 picture of the dataset with their labels

# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data_df.path_img[i]))
    ax.set_title(data_df.breed[i])
plt.tight_layout()
plt.show()


# What do you observe?
# 
# All images are of differnt sizes
# 
# The backgrounsd vary- some have humans, and other items in the backgrounds
# 
# Also some images are not vertical - e.g., the lakeland terrier in the lower night

# # Image Transforms using Pytorch

# In[ ]:


class DogDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['id'], row['lbl']
        img_fname = self.root_dir + "/" + str(img_id) + ".jpg"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, img_label


# Lets perform image transforms the same using PyTorch
# 
# for a 
# [Beginner's Guide: Image Augmentation & Transforms click here](https://www.kaggle.com/kmldas/beginner-s-guide-image-augmentation-transforms)

# In[ ]:


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
    T.Resize((300,300)),
#    T.CenterCrop(256),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#    T.RandomCrop(32, padding=4, padding_mode='reflect'),
    T.RandomHorizontalFlip(), 
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(*imagenet_stats,inplace=True), 
#    T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
    T.Resize((300,300)),
    #T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
])


# In[ ]:


np.random.seed(42)
msk = np.random.rand(len(data_df)) < 0.8

train_df = data_df[msk].reset_index()
val_df = data_df[~msk].reset_index()


# In[ ]:


train_ds = DogDataset(train_df, TRAIN_DIR, transform=train_tfms)
val_ds = DogDataset(val_df, TRAIN_DIR, transform=valid_tfms)
len(train_ds), len(val_ds)


# In[ ]:


def show_sample(img, target, invert=True):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', labels[target])


# # View Sample Images after Transform

# We view images with inverted colours and normal colours

# In[ ]:


show_sample(*train_ds[241])


# In[ ]:


show_sample(*train_ds[419], invert=False)


# # Data holders

# In[ ]:


batch_size = 128


# In[ ]:


train_dl = DataLoader(train_ds, batch_size, shuffle=True, 
                      num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, 
                    num_workers=3, pin_memory=True)


# In[ ]:


def show_batch(dl, invert=True):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break


# # View Batch images

# We view images with inverted colours and normal colours

# In[ ]:


show_batch(train_dl, invert=True)


# In[ ]:


show_batch(train_dl, invert=False)


# # Model - Transfer Learning

# We define accuracy as number of pictures correctly classified or predicted to belong to the accurate class
# 

# In[ ]:


def accuracy(output, label):
    _, pred = torch.max(output, dim=1)
    return torch.tensor(torch.sum(pred == label).item() / len(pred))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, targets) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, targets)   # Calculate loss
        acc = accuracy(out, targets)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.8f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


# We will use a Resnet34 model. We use a pretrained model

# In[ ]:


resnet34 = models.resnet34()
resnet34


# In[ ]:


class DogResnet34(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs,120)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True


# We recommend using a CUDA or GPU is available;
# 
# if not this may be run using a CPU but will take a longer time

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


# # Training

# Now lets get into training the model
# 
# We will use one cycle fit which is now the state of the art for fitting the model.

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
        for batch in tqdm(train_loader):
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


# We load the mdodel in to the device

# In[ ]:


model = to_device(DogResnet34(), device)


# We see what the default accuracy is

# In[ ]:


history = [evaluate(model, val_dl)]
history


# The default accuracy is around 1% (0.01) as there are 120 breeds

# In[ ]:


model.freeze()


# We use the following parameters for the model
# 
# This is what you should focus on. Please change the parameters and see how that improves or decreases the accuracy.
# 
# Understanding the impact of the number of epochs, maximum learning rate, grad clip and weight decay will help you understand how to tune this and other models
# 
# 

# In[ ]:


epochs = 5
max_lr = 0.0001
grad_clip = 0.5
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[ ]:


get_ipython().run_cell_magic('time', '', 'starttime= time.time()\nhistory += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# We store the values
# and unfreeze the model

# In[ ]:


model.unfreeze()


# We run the model again
# 
# 
# 
# I am only changing the max lr to a tenth.  You may change the different parameters and even the model here

# In[ ]:


get_ipython().run_cell_magic('time', '', 'max_lr = max_lr/10\n\n#epochs = epochs-1  \n#grad_clip = grad_clip/5\n#weight_decay = weight_decay/10\n\nhistory += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# You may tun the model a third time as well

# In[ ]:


#model.unfreeze()


# In[ ]:


#%%time
#max_lr = max_lr/10

#history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
#                         grad_clip=grad_clip, 
#                         weight_decay=weight_decay, 
#                         opt_func=opt_func)


# Note the time to model. This model by default should give you 75% accuracy in around 30 minutes of training with GPU

# In[ ]:


endtime=time.time()

duration=endtime-starttime
train_time=time.strftime('%M:%S', time.gmtime(duration))
train_time


# Lets plot charts on the progress of some parameters 

# In[ ]:


def plot_scores(history):
    scores = [x['val_acc'] for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy vs. No. of epochs');


# In[ ]:


plot_scores(history)


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


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


# In[ ]:


plot_lrs(history)


# # Save and Commit

# In[ ]:


weights_fname = 'dog-resnet.pth'
torch.save(model.state_dict(), weights_fname)


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


jovian.reset()
jovian.log_hyperparams(arch='resnet34', 
                       epochs=3*epochs, 
                       lr=max_lr*10, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)


# In[ ]:


jovian.log_metrics(val_loss=history[-1]['val_loss'], 
                   val_score=history[-1]['val_acc'],
                   train_loss=history[-1]['train_loss'],
                   time=train_time)


# In[ ]:


project_name='dog-breed-classification'


# In[ ]:


jovian.commit(project=project_name, environment=None, outputs=[weights_fname])

