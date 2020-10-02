#!/usr/bin/env python
# coding: utf-8

# # IDENTIFY THE DANCE FORM
# 
# **Identify the dance form is a machine learning competition hosted by hackerearth: a platform where you can learn coding and machine learning. I'll use the dataset of this competition for my course project 'Deep Learning with Pytorch: Zero to GANs' : a free course provided by jovian.ml and freecode camp. For more information visit.** 

# ## Problem statement
# 
# This International Dance Day, an event management company organized an evening of Indian classical dance performances to celebrate the rich, eloquent, and elegant art of dance. Post the event, the company planned to create a microsite to promote and raise awareness among the public about these dance forms. However, identifying them from images is a tough nut to crack.
# 
# You have been appointed as a Machine Learning Engineer for this project. Build an image tagging Deep Learning model that can help the company classify these images into eight categories of Indian classical dance.
# 
# **The dataset consists of 364 images belonging to 8 categories, namely manipuri, bharatanatyam, odissi, kathakali, kathak, sattriya, kuchipudi, and mohiniyattam.**
# 
# Dataset link : https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-identify-dance-form/

# **In this notebook I will usedifferent artchiture given below and measure their effectiveness:**
# 1. Use feedword neural network
# 2. Transfer learning

# # Importing the libraries

# In[ ]:


get_ipython().system('pip install fastai2')


# In[ ]:


from fastai2.vision.all import *
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.transforms as T
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


project_name = 'Dance-Classifier'


# # Exploring The Dataset

# In[ ]:


DATA_DIR = '../input/indian-danceform-classification/dataset'

TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
TEST_DIR = DATA_DIR + '/test'                             # Contains test images

TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV =  DATA_DIR + '/test.csv'                        # Contains dummy labels for test image


# All the given training images are at TRAIN_DIR directory. Let's look at few: 

# In[ ]:


os.listdir(TRAIN_DIR)[:5]


# In[ ]:


Image.open(TRAIN_DIR+'/234.jpg')


# In[ ]:


Image.open(TRAIN_DIR+'/287.jpg')


# **Q: What is the total no of images?**

# In[ ]:


len(os.listdir(TRAIN_DIR))


# **To train a classifier we need labels which is given in TRAIN_CSV file. Each images is mapped with their labels in this file. Let's look at this file.**

# In[ ]:


train_df = pd.read_csv(TRAIN_CSV)
train_df.head()


# **Q: what is the total no of images belongs to each class?**

# In[ ]:


train_df.target.value_counts()


# # Dataset and Dataloader
# 
# I will use fastai datablock to make dataset and dataloader which I've learn recently through fastbook. If you want to learn Machine Learning this is the best resourse. Visit [fast.ai](http://fast.ai) for more information.

# In[ ]:


def get_x(r): return DATA_DIR+'/train/'+r['Image']  # Image Directory
def get_y(r): return r['target']                    # Getting the label
dblock = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    splitter=RandomSplitter(),
    get_x = get_x,
    get_y = get_y,
    item_tfms = Resize(330),
    batch_tfms=aug_transforms(mult=2))

dls = dblock.dataloaders(train_df)

train_dl = dls.train
valid_dl = dls.valid


# **Let's look a batch of dataset**

# In[ ]:


dls.show_batch()


# In[ ]:


dls.train.show_batch()


# In[ ]:


dls.valid.show_batch()


# In[ ]:


# Let's save the work to jovian
get_ipython().system('pip install jovian --upgrade -q')
import jovian
jovian.commit(project = project_name, environment=None)


# # Model
# Let's create a base model class, which contains everything except the model architecture i.e. it wil not contain the __init__ and __forward__ methods. We will later extend this class to try out different architectures. In fact, you can extend this model to solve any image classification problem.

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
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


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


# # 1. Feedforward Network

# In[ ]:


class DanceModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 512)  # first linear layer
        self.linear2 = nn.Linear(512, 128)          # second linear layer
        self.linear3 = nn.Linear(128, output_size)  # third linear layer

        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.linear1(out)
        out = F.relu(out)
        
        out = self.linear2(out)
        out = F.relu(out)
        
        out = self.linear3(out)
        return out


# # Model on GPU

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


# In[ ]:


device = get_default_device()
print(device)

# image size is 224x224x3 
# output class = 8
input_size = 330*330*3
output_size = 8
model = to_device(DanceModel(), device)


# Before you train the model, it's a good idea to check the validation loss & accuracy with the initial set of weights.

# In[ ]:


history = [evaluate(model, valid_dl)]
history


# **Train the model using the fit function to reduce the validation loss & improve accuracy.**

# In[ ]:


history += fit(5, 1e-3, model, train_dl, valid_dl)


# In[ ]:


history += fit(5, 1e-2, model, train_dl, valid_dl)


# **Let us also define a couple of helper functions for plotting the losses & accuracies.**

# In[ ]:


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');
    
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    


# In[ ]:


plot_losses(history)


# In[ ]:


plot_accuracies(history)


# ## Recoding your results
# 
# As we perform multiple experiments, it's important to record the results in a systematic fashion, so that we can review them later and identify the best approaches that we might want to reproduce or build upon later. 

# **list of artchitures**

# In[ ]:


arch1 = "4 layers (1024, 512, 128, 8)"
arch2 = '3 layers (512, 128, 8)'
arch = [arch1, arch2]


# **List of learning rates**

# In[ ]:


lrs1 = [1e-2, 1e-3]
lrs2 = [1e-2, 1e-3]
lrs = [lrs1, lrs2]


# **No of epoch used while training**

# In[ ]:


epoch1 = [5, 5]
epoch2 = [5, 5]
epochs = [epoch1, epoch2]


# **Final validation accuracy and loss**

# In[ ]:


valid_acc = [14.8, 24]
valid_loss = [2.10, 2.10]


# In[ ]:


torch.save(model.state_dict(), 'dance-feed-forward.pth')


# In[ ]:


# Clear previously recorded hyperparams & metrics
jovian.reset()


# In[ ]:


jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)


# In[ ]:


jovian.log_metrics(valid_loss=valid_loss, valid_acc=valid_acc)


# In[ ]:


jovian.commit(project=project_name, outputs=['dance-feed-forward.pth'], environment=None)


# # 2. Transfer Learning

# In[ ]:


class DanceResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
#         self.network = models.resnet34(pretrained=True)
        self.network = models.resnet50(pretrained=True)
        
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)
    
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


# # Training

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


# # Model on GPU

# In[ ]:


device = get_default_device()
print(device)
model = to_device(DanceResnet(), device)


# **Train the model using the fit function to reduce the validation loss & improve accuracy.**

# In[ ]:


history = [evaluate(model, valid_dl)]
history


# In[ ]:


model.freeze()


# In[ ]:


epochs = 5
max_lr =  1e-3
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# In[ ]:


model.unfreeze()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(5, 1e-4, model, train_dl, valid_dl, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# In[ ]:


plot_losses(history)


# In[ ]:


plot_accuracies(history)


# ## Recoding your results
# 
# As we perform multiple experiments, it's important to record the results in a systematic fashion, so that we can review them later and identify the best approaches that we might want to reproduce or build upon later. 

# **list of artchitures**

# In[ ]:


arch1 = 'resnet 34'
arch2 = 'resnet 50'
arch3 = 'resnet 50: replaced RandomResized224 to Resize224'
arch3 = 'resnet 50: replaced Resized224 to Resize330'
arch = [arch1, arch2, arch3]


# **List of learning rates**

# In[ ]:


lrs1 = [1e-4, 1e-4]
lrs2 = [1e-3, 1e-4]
lrs = [lrs1, lrs2]


# **No of epoch used while training**

# In[ ]:


epoch1 = [5, 5]
epoch2 = [5, 5]
epochs = [epoch1, epoch2]


# **Final validation accuracy and loss**

# In[ ]:


valid_acc = [64, 71]
valid_loss = [1.76, 1.68]


# In[ ]:


torch.save(model.state_dict(), 'dance-resnet50.pth')


# In[ ]:


# Clear previously recorded hyperparams & metrics
jovian.reset()


# In[ ]:


jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)


# In[ ]:


jovian.log_metrics(valid_loss=valid_loss, valid_acc=valid_acc)


# In[ ]:


jovian.commit(project=project_name, outputs=['dance-resnet50.pth'], environment=None)


# In[ ]:




