#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/snrazavi/Deep_Learning_in_Python_2018/master/Week01/train.py')
get_ipython().system('wget https://raw.githubusercontent.com/snrazavi/Deep_Learning_in_Python_2018/master/Week01/utils.py')
get_ipython().system('wget https://raw.githubusercontent.com/snrazavi/Deep_Learning_in_Python_2018/master/Week01/data_utils.py')
get_ipython().system('wget https://raw.githubusercontent.com/snrazavi/Deep_Learning_in_Python_2018/master/Week01/model_utils.py')
get_ipython().system('wget https://raw.githubusercontent.com/snrazavi/Deep_Learning_in_Python_2018/master/Week01/predict_utils.py')
get_ipython().system('wget https://raw.githubusercontent.com/snrazavi/Deep_Learning_in_Python_2018/master/Week01/vis_utils.py')


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import math
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# Our libraries
from train import train_model
from model_utils import *
from predict_utils import *
from vis_utils import *

# some initial setup
np.set_printoptions(precision=2)
use_gpu = torch.cuda.is_available()
np.random.seed(1234)


# In[ ]:


use_gpu


# In[ ]:


DATA_DIR = '../input/dogs-vs-cats-train-validadion-and-evaluation/data'
sz = 224
batch_size = 16


# In[ ]:


os.listdir(DATA_DIR)


# In[ ]:


trn_dir = f'{DATA_DIR}/train'
val_dir = f'{DATA_DIR}/validation'


# In[ ]:


os.listdir(trn_dir)


# In[ ]:


trn_fnames = glob.glob(f'{trn_dir}/*/*.jpg')
trn_fnames[:5]


# In[ ]:


img = plt.imread(trn_fnames[3])
plt.imshow(img);


# In[ ]:


train_ds = datasets.ImageFolder(trn_dir)


# In[ ]:


train_ds.classes


# In[ ]:


train_ds.class_to_idx


# In[ ]:


train_ds.root


# In[ ]:


train_ds.imgs


# In[ ]:


type(train_ds.transform)


# 
# ## Transformations
# 
# Dataloader object uses these tranformations when loading data.
# 

# In[ ]:


tfms = transforms.Compose([
    transforms.Resize((sz, sz)),  # PIL Image
    transforms.ToTensor(),        # Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(trn_dir, transform=tfms)
valid_ds = datasets.ImageFolder(val_dir, transform=tfms)


# In[ ]:


len(train_ds), len(valid_ds)


# ###  Dataloaders

# In[ ]:


train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 
                                       shuffle=True, num_workers=8)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, 
                                       shuffle=True, num_workers=8)


# In[ ]:


inputs, targets = next(iter(train_dl))
out = torchvision.utils.make_grid(inputs, padding=3)
plt.figure(figsize=(16, 12))
imshow(out, title='Random images from training data')


# In[ ]:


class SimpleCNN(nn.Module):
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Linear(56 * 56 * 32, 2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)            # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W)
        out = self.fc(out)
        return out


# In[ ]:


model = SimpleCNN()

# transfer model to GPU
if use_gpu:
    model = model.cuda()


# In[ ]:


model


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)


# In[ ]:


num_epochs = 10
losses = []
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_dl):
        inputs = to_var(inputs)
        targets = to_var(targets)
        
        # forwad pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # loss
        loss = criterion(outputs, targets)
        losses += [loss.data]
        # backward pass
        loss.backward()
        
        # update parameters
        optimizer.step()
        
        # report
        if (i + 1) % 50 == 0:
            print('Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_ds) // batch_size, loss.data))


# In[ ]:


plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Cross Entropy Loss');


# In[ ]:


def evaluate_model(model, dataloader):
    model.eval()  # for batch normalization layers
    corrects = 0
    for inputs, targets in dataloader:
        inputs, targets = to_var(inputs, True), to_var(targets, True)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        corrects += (preds == targets.data).sum()
    
    print('accuracy: {:.2f}'.format(100. * corrects / len(dataloader.dataset)))


# In[ ]:


evaluate_model(model, valid_dl)


# In[ ]:


evaluate_model(model, train_dl)


# In[ ]:


visualize_model(model, train_dl)


# In[ ]:


visualize_model(model, valid_dl)


# In[ ]:


plot_errors(model, valid_dl)


# ### Confusion matrix

# In[ ]:


y_pred, y_true = predict_class(model, valid_dl)
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, train_ds.classes, normalize=True, figsize=(4, 4))

