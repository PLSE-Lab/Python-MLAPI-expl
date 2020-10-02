#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load data

# In[ ]:


get_ipython().system('wget https://github.com/SayedMaheen/sg_PlanetEarth/archive/master.zip')


# In[ ]:


get_ipython().system('unzip master.zip')


# In[ ]:


data_dir = 'sg_PlanetEarth-master/smoke_data'


# # Show some images

# In[ ]:


from PIL import Image
name = os.listdir(data_dir+'/smog')[10]
Image.open(data_dir+"/smog/"+name)


# # compare two image

# In[ ]:


from matplotlib import pyplot as plt
fig = plt.figure(figsize=(25,5))
ax = fig.add_subplot(1, 5, 1, xticks=[], yticks=[])
img = Image.open(data_dir+"/smog/"+name)
ax.imshow(img, cmap='gray')
ax.set_title('Smog')

# second image
ax = fig.add_subplot(1, 5, 2, xticks=[], yticks=[])
name2 = os.listdir(data_dir+'/clear')[10]
img2 = Image.open(data_dir+"/clear/"+name2)
ax.imshow(img2, cmap='gray')
ax.set_title('Clear')


# # Check image size

# In[ ]:


size = Image.open(data_dir+"/smog/"+name)
size.size


# # Load Helper class

# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/Iamsdt/60daysofudacity/master/day22/Helper.py')


# # Prepare Data

# In[ ]:


import Helper
import torch
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
                                transforms.RandomRotation(20),
                                transforms.Resize(255),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
test_transform = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

train_loader, valid_loader = Helper.prepare_loader(data_dir, data_dir,
                                                  train_transform, test_transform, batch_size=64)

len(train_loader)


# In[ ]:


classes =train_loader.dataset.classes
classes


# # See photos after transform

# In[ ]:


Helper.visualize(train_loader, classes, num_of_image=4)


# # Load pretrained Model

# In[ ]:


model = models.resnet50(pretrained=True)
model.fc


# Freeeze model

# In[ ]:


model = Helper.freeze_parameters(model)


# ### replace Classifier

# In[ ]:


import torch.nn as nn
from collections import OrderedDict

classifier = nn.Sequential(
  nn.Linear(in_features=2048, out_features=1536),
  nn.ReLU(),
  nn.Dropout(p=0.4),
  nn.Linear(in_features=1536, out_features=1024),
  nn.ReLU(),
  nn.Dropout(p=0.3),
  nn.Linear(in_features=1024, out_features=2),
  nn.LogSoftmax(dim=1) 
)
    
model.fc = classifier
model.fc


# # loss and optimizer

# In[ ]:


import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)


# # Train

# In[ ]:


epoch = 10


# In[ ]:


model, train_loss, test_loss = Helper.train(model, train_loader, valid_loader, epoch, optimizer, criterion)


# # load best model

# In[ ]:


model = Helper.load_latest_model(model)


# In[ ]:


Helper.check_overfitted(train_loss, test_loss)


# # Test

# In[ ]:


Helper.test(model, valid_loader, criterion)


# # test with new transform

# In[ ]:


test_transform = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

test_data = datasets.ImageFolder(data_dir, transform=test_transform)
print(len(test_data))

test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=128)
print(len(test_loader))


# In[ ]:


Helper.test(model, test_loader, criterion)


# # test single image

# In[ ]:


def test(file):
  ids = train_loader.dataset.class_to_idx

  with Image.open(file) as f:
      img = test_transform(f).unsqueeze(0)
      with torch.no_grad():
          out = model(img.to(device)).cpu().numpy()
          for key, value in ids.items():
              if value == np.argmax(out):
                    print(f"Predicted Label: {key}")
          plt.imshow(np.array(f))
          plt.show()


# ### check somg

# In[ ]:


name = os.listdir(data_dir+'/smog')[10]
file = data_dir+'/smog/'+name
print(file)
test(file)


# ## check clear

# In[ ]:


name = os.listdir(data_dir+'/clear')[10]
file = data_dir+'/clear/'+name
print(file)
test(file)

