#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# In this kernel I'll build a CNN in Pytorch to identify Whales. The first attempt was done by using a simple CNN from scratch, but it didn't work well, so I'll use pre-trained nets.
# 
# ![](https://s23444.pcdn.co/wp-content/uploads/2014/04/Gold-Coast-whale.jpg.optimal.jpg)

# ### Versions
# Here I'll list significant changes in the notebook.
# * v10 - added augmentations and scheduler
# * v18 - changed augmentations to albumentations

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
import tqdm
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

from collections import OrderedDict
import cv2


# In[ ]:


get_ipython().system('pip install albumentations > /dev/null 2>&1')


# In[ ]:


get_ipython().system('pip install pretrainedmodels > /dev/null 2>&1')


# In[ ]:


import albumentations
from albumentations import torch as AT
import pretrainedmodels


# ### Data overview

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df.head()


# For train data we have a DataFrame with image names and ids. And of course for train and test we have images in separate folders.

# In[ ]:


print(f"There are {len(os.listdir('../input/train'))} images in train dataset with {train_df.Id.nunique()} unique classes.")
print(f"There are {len(os.listdir('../input/test'))} images in test dataset.")


# 25k images in train and 5k different whales!
# Let's have a look at them

# In[ ]:


fig = plt.figure(figsize=(25, 4))
train_imgs = os.listdir("../input/train")
for idx, img in enumerate(np.random.choice(train_imgs, 20)):
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    im = Image.open("../input/train/" + img)
    plt.imshow(im)
    lab = train_df.loc[train_df.Image == img, 'Id'].values[0]
    ax.set_title(f'Label: {lab}')


# In[ ]:


train_df.Id.value_counts().head()


# In[ ]:


for i in range(1, 4):
    print(f'There are {train_df.Id.value_counts()[train_df.Id.value_counts().values==i].shape[0]} classes with {i} samples in train data.')


# In[ ]:


plt.title('Distribution of classes excluding new_whale');
train_df.Id.value_counts()[1:].plot(kind='hist');


# We can see that there is a huge disbalance in the data. There are many classes with only one or several samples , some classes have 50+ samples and "default" class has almost 10k samples.

# In[ ]:


np.array(im).shape


# At least some images are quite big

# ### Preparing data for Pytorch
# Data for Pytorch needs to be prepared:
# * we need to define transformations;
# * then we need to initialize a dataset class;
# * then we need to create dataloaders which will be used by the model;

# #### Transformations
# 
# Basic transformations include only resizing the image to the necessary size, converting to Pytorch tensor and normalizing

# In[ ]:


data_transforms = transforms.Compose([
                                      transforms.Resize((100, 100)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
    ])
data_transforms_test = transforms.Compose([
                                           transforms.Resize((100, 100)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
])


# #### Encoding labels
# Labels need to be one-hot encoded.

# In[ ]:


def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder


# In[ ]:


y, le = prepare_labels(train_df['Id'])


# #### Dataset
# 
# Now we need to create a dataset. Sadly, default version won't work, as images for each class are supposed to be in separate folders. So I write a custom WhaleDataset.

# In[ ]:


class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None, transform = transforms.Compose([transforms.ToTensor()]), y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train':
            self.df = df.values
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_files_list)
    
    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]
            
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            label = np.zeros((5005,))

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]


# In[ ]:


train_dataset = WhaleDataset(datafolder='../input/train/', datatype='train', df=train_df, transform=data_transforms, y=y)
test_set = WhaleDataset(datafolder='../input/test/', datatype='test', transform=data_transforms_test)


# #### Loaders
# 
# Now we create loaders. Here we define which images will be used, batch size and other things.

# In[ ]:


train_sampler = SubsetRandomSampler(list(range(len(os.listdir('../input/train')))))
valid_sampler = SubsetRandomSampler(list(range(len(os.listdir('../input/test')))))
batch_size = 512
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
# less size for test loader.
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=num_workers)


# ### Basic CNN
# 
# Now we can define the model. For now I'll use a simple architecture with two convolutional layers.

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)        
        self.pool2 = nn.AvgPool2d(3, 3)
        
        self.fc1 = nn.Linear(64 * 4 * 4 * 16, 1024)
        self.fc2 = nn.Linear(1024, 5005)

        self.dropout = nn.Dropout(0.5)        

    def forward(self, x):
        x = self.pool(F.relu(self.conv2_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# #### Initializing model
# 
# We need to define model, loss, oprimizer and possibly a scheduler.

# In[ ]:


model_conv = Net()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model_conv.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# But this was an example. The basic model isn't really good. You can try it, but it is much better to use pre-trained models.

# In[ ]:


# model_conv.cuda()
# n_epochs = 10
# for epoch in range(1, n_epochs+1):
#     print(time.ctime(), 'Epoch:', epoch)

#     train_loss = []
#     exp_lr_scheduler.step()

#     for batch_i, (data, target) in enumerate(train_loader):
#         #print(batch_i)
#         data, target = data.cuda(), target.cuda()

#         optimizer.zero_grad()
#         output = model_conv(data)
#         loss = criterion(output, target.float())
#         train_loss.append(loss.item())

#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}')

# sub = pd.read_csv('../input/sample_submission.csv')

# model_conv.eval()
# for (data, target, name) in test_loader:
#     data = data.cuda()
#     output = model_conv(data)
#     output = output.cpu().detach().numpy()
#     for i, (e, n) in enumerate(list(zip(output, name))):
#         sub.loc[sub['Image'] == n, 'Id'] = ' '.join(le.inverse_transform(e.argsort()[-5:][::-1]))
        
# sub.to_csv('basic_model.csv', index=False)


# ### Pretrained model
# 
# Now I'll use a pretrained model. First this is changing transformations: I use a bigger image size and add augmentations.

# In[ ]:


class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None, transform = transforms.Compose([transforms.ToTensor()]), y=None
                ):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train':
            self.df = df.values
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_files_list)
    
    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]
            
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            label = np.zeros((5005,))

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(image=img)
        image = image['image']
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]


# In[ ]:


data_transforms = albumentations.Compose([
    albumentations.Resize(160, 320),
    albumentations.HorizontalFlip(),
    albumentations.RandomBrightness(),
    albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
    albumentations.JpegCompression(80),
    albumentations.HueSaturationValue(),
    albumentations.Normalize(),
    AT.ToTensor()
    ])
data_transforms_test = albumentations.Compose([
    albumentations.Resize(160, 320),
    albumentations.Normalize(),
    AT.ToTensor()
    ])

train_dataset = WhaleDataset(datafolder='../input/train/', datatype='train', df=train_df, transform=data_transforms, y=y)
test_set = WhaleDataset(datafolder='../input/test/', datatype='test', transform=data_transforms_test)

train_sampler = SubsetRandomSampler(list(range(len(os.listdir('../input/train')))))
valid_sampler = SubsetRandomSampler(list(range(len(os.listdir('../input/test')))))
batch_size = 10
num_workers = 2
# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
#valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, num_workers=num_workers)


# In[ ]:


# model_conv = torchvision.models.resnet101(pretrained=True)
# for i, param in model_conv.named_parameters():
#     param.requires_grad = False


# In[ ]:


# final_layer = nn.Sequential(OrderedDict([
#                           ('fc1', nn.Linear(2048, 1024)),
#                           ('relu', nn.ReLU()),
#                           ('dropout', nn.Dropout(0.1)),
#                           ('fc2', nn.Linear(1024, 5005)),
#                           ]))
# model_conv.fc = final_layer
#model_conv.fc = nn.Linear(2048, 5005)


# In[ ]:


model_conv = pretrainedmodels.resnext101_64x4d()
model_conv.avg_pool = nn.AvgPool2d((5,10))
model_conv.last_linear = nn.Linear(model_conv.last_linear.in_features, 5005)


# In[ ]:


model_conv.cuda()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model_conv.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# In[ ]:


n_epochs = 4
for epoch in range(1, n_epochs+1):
    print(time.ctime(), 'Epoch:', epoch)

    train_loss = []
    

    for batch_i, (data, target) in enumerate(train_loader):
        # print(f'Batch {batch_i} of 50')
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model_conv(data)
        loss = criterion(output, target.float())
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    exp_lr_scheduler.step()

    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}')


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')

model_conv.eval()
for (data, target, name) in test_loader:
    data = data.cuda()
    output = model_conv(data)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        sub.loc[sub['Image'] == n, 'Id'] = ' '.join(le.inverse_transform(e.argsort()[-5:][::-1]))
        
sub.to_csv('submission.csv', index=False)
