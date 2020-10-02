#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('cd', '../input/dog-breed-identification/')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import os
import cv2
from PIL import Image
import time
import copy
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90,Transpose,RandomBrightnessContrast,RandomCrop)
from albumentations.pytorch import ToTensor
import albumentations as albu
import matplotlib.image as mpi
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# In[ ]:


sub = pd.read_csv('sample_submission.csv')


# In[ ]:


sub.keys()


# In[ ]:


df = pd.read_csv("labels.csv")
df1 = df['breed']
df2 = df["id"]
df1 = pd.get_dummies(df1)
df = pd.concat([df2,df1], axis=1)
df.head()


# In[ ]:


df_train,df_val = train_test_split(df,test_size=0.2,random_state=42)


# In[ ]:


class DogDataset(Dataset):

  def __init__(self,df,root,phase):
    self.df = df
    self.length = df.shape[0]
    self.root = root
    if phase=="train":
        self.transforms = albu.Compose([
            albu.SmallestMaxSize(256),
            albu.RandomCrop(256,256),
            albu.HorizontalFlip(p=0.5),
            albu.Cutout(),
            albu.RGBShift(),
            albu.Rotate(limit=(-90,90)),
            albu.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
    elif phase=="val":
        self.transforms = albu.Compose([
            albu.Resize(256,256),
            albu.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

  def __getitem__(self,index):
    label = self.df.iloc[index,1:]
    label = label.to_numpy()
    image_id = self.df.iloc[index,0]
    path = os.path.join(self.root,str(image_id) + ".jpg")
    img = plt.imread(path)
    img = self.transforms(image=np.array(img))
    img = img['image']
    img = np.transpose(img,(2,0,1)).astype(np.float32)
    img = torch.tensor(img, dtype = torch.float)
    label = np.argmax(label)
    return img,label
  
  def __len__(self):
    return self.length 
  
  def label_name(self,label):
    breeds = self.df.columns.values
    breeds = breeds[1:]
    idx = np.argmax(label)
    return breeds[idx]


# In[ ]:


traindata = DogDataset(df_train,root = "train", phase="train")
valdata = DogDataset(df_val,root = "train", phase="val")
trainloader = DataLoader(traindata,batch_size = 24,num_workers=0)
valloader = DataLoader(valdata,batch_size = 24,num_workers=0)


# In[ ]:


dataiter = iter(trainloader)
image,label = dataiter.next()


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


def show_img(img):
    plt.figure(figsize=(18,15))
    img = img / 2 + 0.5  
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

show_img(torchvision.utils.make_grid(image))


# In[ ]:


print(image.shape)


# In[ ]:


from torchvision import models
resnet = models.resnet152(pretrained=True).to(device)


# In[ ]:


for param in resnet.parameters():
    param.requires_grad=False
fc_inputs = resnet.fc.in_features
resnet.fc = nn.Linear(fc_inputs,120)


# In[ ]:


from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=0.001)
scheduler = ReduceLROnPlateau(optimizer,factor=0.33, mode="min", patience=2)


# In[ ]:


def train_model(dataloaders,model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    dataset_sizes = {'train': len(dataloaders['train'].dataset), 
                     'val': len(dataloaders['val'].dataset)}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    number_of_iter = 0
    acc_train = []
    acc_val = []
    loss_train = []
    loss_val = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            current_loss = 0.0
            current_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_corrects.double() / dataset_sizes[phase]
            if phase=="train":
                acc_train.append(epoch_acc)
                loss_train.append(epoch_loss)
            else:
                acc_val.append(epoch_acc)
                loss_val.append(epoch_loss)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    
    
    return model,acc_val,acc_train,loss_train,loss_val


# In[ ]:


resnet = resnet.to(device)
dataloaders = {"train":trainloader,"val":valloader}
num_epochs=25
start_time = time.time()
model,acc_val,acc_train,loss_train,loss_val = train_model(dataloaders, resnet, criterion, optimizer, scheduler, num_epochs=num_epochs)


# In[ ]:


epoch = []
for x in range(num_epochs):
    epoch.append(x)
plt.plot(epoch,loss_train,label = 'TrainLoss')
plt.plot(epoch,loss_val,label = 'ValLoss')
plt.legend()
plt.show()


# In[ ]:


get_ipython().run_line_magic('cd', "'/kaggle/working'")


# In[ ]:


torch.save(model.state_dict(),'res152.pth')


# In[ ]:


output = pd.DataFrame(index=sub.index,columns = sub.keys())
output['id'] = sub['id']


# In[ ]:


testdata = DogDataset(sub,root="test",phase='val')
testloader = DataLoader(testdata,batch_size=24)


# In[ ]:


def test_submission(model):
    since = time.time()
    sub_output = []
    model.train(False)
    for data in testloader:
        inputs,labels = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        sub_output.append(outputs.data.cpu().numpy())
    sub_output = np.concatenate(sub_output)
    for idx,row in enumerate(sub_output.astype('float')):
        sub_output[idx] = np.exp(row)/np.sum(np.exp(row))
    output.loc[:,1:] = sub_output
    print()
    time_elapsed = time.time() - since
    print('Run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        


# In[ ]:


get_ipython().run_line_magic('cd', "'/kaggle/input/dog-breed-identification'")


# In[ ]:


model = model.to(device)
test_submission(model)


# In[ ]:


output.head()


# In[ ]:


get_ipython().run_line_magic('cd', "'/kaggle/working'")


# In[ ]:


output.to_csv("dogs_idres152.csv", index=False)


# In[ ]:




