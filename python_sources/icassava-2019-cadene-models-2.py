#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pretrainedmodels')


# In[ ]:


import pretrainedmodels


# In[ ]:


import os


# In[ ]:


os.listdir("../input/train/train")


# In[ ]:


os.listdir("../input/train/train/cgm")[:10]


# In[ ]:


os.listdir("../input/test/test/0")[:10]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import time
import numpy as np
import pandas as pd
import os
import cv2
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models

from sklearn.model_selection import train_test_split


# In[ ]:


train_path = "../input/train/train"
test_path = "../input/test/test/0"


# In[ ]:


def get_labels(file_path): 
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_levels = len(split_dir_name)
    label  = split_dir_name[dir_levels - 1]
    return(label)


# In[ ]:


get_labels("../input/train/train/cgm/train-cgm-528.jpg")


# In[ ]:


from glob import glob
imagePatches = glob("../input/train/train/*/*.*", recursive=True)
imagePatches[0:10]


# In[ ]:


images_df = pd.DataFrame()
images_df["images"] = imagePatches

y = []
for img in imagePatches:
    y.append(get_labels(img))   

images_df["labels"] = y


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
images_df["labels"] = labelencoder.fit_transform(images_df["labels"])


# In[ ]:


images_df.head()


# In[ ]:


class MyDataset(Dataset):
    def __init__(self, df_data,transform=None):
        super().__init__()
        self.df = df_data.values
        
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path,label = self.df[index]
        
        image = cv2.imread(img_path)
        #image = cv2.resize(image, (500,500))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# In[ ]:


batch_size = 16
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


#Splitting data into train and val
train, val = train_test_split(images_df, stratify=images_df.labels, test_size=0.2)
len(train), len(val)


# In[ ]:


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


# In[ ]:


imagePatches_test = glob("../input/test/test/0/*.*", recursive=True)
imagePatches_test[0:10]


# In[ ]:


images_df_test = pd.DataFrame()
images_df_test["images"] = imagePatches_test
images_df_test["label"] = -1


# In[ ]:


import random
import math
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


# In[ ]:


from torchvision import transforms as T
size = 500
trans = transforms.Compose([transforms.ToPILImage(),
                                  T.RandomApply([T.RandomAffine(45, shear=15)], 0.8),
                                T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(3/5, 5/3)),
                                T.RandomHorizontalFlip(),
                                T.RandomVerticalFlip(),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                RandomErasing(probability=0.3, sh=0.3)])

dataset_train = MyDataset(df_data=train, transform=trans)
dataset_valid = MyDataset(df_data=val,transform=trans)

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)


# In[ ]:


dataset_test = MyDataset(df_data=images_df_test,transform=trans)
loader_test = DataLoader(dataset = dataset_test, batch_size=batch_size//2, shuffle=False, num_workers=0)


# In[ ]:


def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    axis.imshow(inp)


# In[ ]:


img, label = next(iter(loader_train))


# In[ ]:


fig = plt.figure(1, figsize=(8,8))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.05)    
for i in range(4):
    ax = grid[i]
    imshow(ax, img[i])


# In[ ]:


device


# In[ ]:


use_gpu = torch.cuda.is_available()
inputs, labels = next(iter(loader_train))
if use_gpu:
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())   
else:
    inputs, labels = Variable(inputs), Variable(labels)


# In[ ]:


def se_resnext50_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return model


# In[ ]:


model = se_resnext50_32x4d(pretrained=False)


# In[ ]:


use_gpu = torch.cuda.is_available()


# In[ ]:


for param in model.parameters():
    param.requires_grad = False

# new final layer with 5 classes
model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
num_ftrs = model.last_linear.in_features
model.last_linear = torch.nn.Linear(num_ftrs, 5)
if use_gpu:
    model = model.cuda()


# In[ ]:


criterion = torch.nn.CrossEntropyLoss()


# In[ ]:


num_epochs = 15


# In[ ]:


total_step = len(loader_train)


# In[ ]:


for epoch in range(num_epochs):
    if (epoch < 2):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00008)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
    for i, (images, labels) in enumerate(loader_train):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# In[ ]:


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loader_valid:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
          
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

