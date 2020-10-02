#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from IPython.display import display

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms,models
from torchvision.utils import make_grid
from PIL import Image


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path= '../input/flowers-recognition/flowers/'


# In[ ]:





# In[ ]:


img_names = []
for folder,subfolders,filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder+'/'+img)
print("images len:",len(img_names))


# In[ ]:


img_sizes = []
rejected = []
for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)
print(f'images:{len(img_sizes)}')
print(f'rejected:{len(rejected)}')


# In[ ]:


df = pd.DataFrame(img_sizes)
df.describe()


# In[ ]:


df.head()


# In[ ]:


daisy_flw = Image.open("../input/flowers-recognition/flowers/daisy/10466558316_a7198b87e2.jpg")
display(daisy_flw)


# In[ ]:





# ### Use train test split

# In[ ]:


train_transform =transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean =[0.485,0.456,0.406], std = [0.229,0.224,0.225])
])


# In[ ]:


test_transform =transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean =[0.485,0.456,0.406], std = [0.229,0.224,0.225])
])


# In[ ]:


## The full_dataset is splitted into train and test datsets

root= '../input/flowers-recognition/'
# we are defining the root where all train-test datas will be loaded

full_dataset = datasets.ImageFolder(os.path.join(root,'flowers'),transform=train_transform)
print('full dataset:',full_dataset)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

print('train size:',train_size)
print('test size:',test_size)
train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])


# In[ ]:


#train_data = datasets.ImageFolder(os.path.join(root,'flowers'),transform=train_transform)
#test_data = datasets.ImageFolder(os.path.join(root,'flowers'),transform=train_transform)

# we are loading datesets as a train loader and test loader with train batch size =20 and test batch size =10

train_btch_size =20
test_btch_size = 10

train_loader = DataLoader(train_data,batch_size=train_btch_size, shuffle = True,num_workers = 8)
test_loader =  DataLoader(test_data,batch_size =test_btch_size ,shuffle = False,num_workers = 8)

class_names = full_dataset.classes
print(class_names)


# In[ ]:


print("Train size:",len(train_data))
print("Test size:",len(test_data))


# In[ ]:


for images,labels in train_loader:
    break

print('labels:',labels.numpy())
print('images:',*np.array([class_names[i] for i in labels]))

im = make_grid(images,nrow=train_btch_size)

im_inv = transforms.Normalize( 
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])
im = im_inv(im)
plt.figure(figsize =(20,4))
plt.imshow(np.transpose(im.numpy(),(1,2,0)))


# In[ ]:



print("images.shape:",images.shape)
print("labels.shape:",labels.shape)
print("len(labels)",len(labels))


# In[ ]:


class ConvolutionalNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.conv3 = nn.Conv2d(64,128,3,1)
        self.fc1   = nn.Linear(128*26*26,16)
        self.fc2   = nn.Linear(16,8)
        self.fc3   = nn.Linear(8,6)
        
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X,2,2)
        
        X = X.view(-1,128*26*26)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X,dim =1)
    


# In[ ]:


# We are going to use pytorch GPU therefore we need to convert datatypes into .cuda() format
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()
use_gpu


# In[ ]:


CNN_model=ConvolutionalNetworks()
if torch.cuda.is_available():
    CNN_model.cuda() 
criterion= nn.CrossEntropyLoss()
if use_gpu:
     criterion = criterion.cuda()
optimizer = torch.optim.Adam(CNN_model.parameters(),lr=0.001)
CNN_model


# In[ ]:


total_param = []
for param in CNN_model.parameters():
    total_param.append(param.numel())

print(total_param)
print("Number of parameters:",sum(total_param))


# In[ ]:


epochs = 3
train_losses =[]
test_losses = []
train_correct =[]
test_correct=[]
for epoch in range(epochs):
    
    trn_corr =0
    tst_corr =0
    for batch,(X_train,y_train) in enumerate(train_loader):
        batch+=1
        if torch.cuda.is_available():
            X_train = Variable(X_train).cuda()
            y_train = Variable(y_train).cuda()
        
        y_pred = CNN_model(X_train)
        loss   = criterion(y_pred,y_train)
        predicted =torch.max(y_pred.data,1)[1]
        trn_corr+=(predicted == y_train).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%100 ==0:
            print(f'epoch:{epoch} batch:{batch} loss:{loss.item()} accuracy:{trn_corr.item()*100/(batch*train_btch_size)}')
            
        
    train_losses.append(loss)
    train_correct.append(trn_corr.item()*100/(batch*train_btch_size))
    
    with torch.no_grad():
        for X_test,y_test in test_loader:
            
            if torch.cuda.is_available():
                X_test = Variable(X_test).cuda()
                y_test = Variable(y_test).cuda()

            y_val= CNN_model(X_test)
            predicted_test = torch.max(y_val.data,1)[1]
            tst_corr +=(predicted_test==y_test).sum()

        loss =criterion(y_val,y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr.item()*100/(batch*test_btch_size))  
    
   
    


# In[ ]:


#  Plot train loss and test loss 
plt.figure(figsize=(12,8))
plt.plot(train_losses,label = 'train loss')
plt.plot(test_losses,label  = 'test loss')
plt.legend()

#  Plot train and test accuracy
plt.figure(figsize=(12,8))
plt.plot(train_correct,label = 'train correct')
plt.plot(test_correct, label = 'test correct')
plt.legend()


# In[ ]:


train_correct[-1]


# > ## Using Pretrained Model

# In this section we are implementing trasferlearning in order to tune parameters in the fully connected layer 

# # 1) AlexNet Model

# In[ ]:


AlexNetModel = models.alexnet(pretrained = True)
AlexNetModel


# In[ ]:


for param in AlexNetModel.parameters():
    param.requires_grad = True # it will train all weights .it will take long time to train it
    #param.requies_grad = False # it will only train last couple of layers. therefor it will take short time to update the parameters


# In[ ]:


# Modifying AlexNet architecture
torch.manual_seed(42)
AlexNetModel.classifier = nn.Sequential(nn.Linear(9216,1024),
                                             nn.ReLU(),
                                             nn.Dropout(0.4),
                                             nn.Linear(1024,6),
                                             nn.LogSoftmax(dim=1),)
AlexNetModel.eval()


# In[ ]:


if torch.cuda.is_available():
    AlexNetModel.cuda()


# In[ ]:


epochs = 3
train_losses =[]
test_losses = []
train_correct =[]
test_correct=[]
for epoch in range(epochs):
    
    trn_corr =0
    tst_corr =0
    for batch,(X_train,y_train) in enumerate(train_loader):
        batch+=1
        if torch.cuda.is_available():
            X_train = Variable(X_train).cuda()
            y_train = Variable(y_train).cuda()
        
        y_pred = AlexNetModel(X_train)
        loss   = criterion(y_pred,y_train)
        predicted =torch.max(y_pred.data,1)[1]
        trn_corr+=(predicted == y_train).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%100 ==0:
            print(f'epoch:{epoch} batch:{batch} loss:{loss.item()} accuracy:{trn_corr.item()*100/(batch*20)}')
            
        
    train_losses.append(loss)
    train_correct.append(trn_corr)


# # 2) Resnet18 Model

# In[ ]:


ResnetModel = models.resnet18(pretrained= True)
ResnetModel


# In[ ]:


for param in ResnetModel.parameters():
    param.requires_grad = True


# ## Modifying Resnet Acrhitecture

# In[ ]:


ResnetModel.fc = nn.Linear(512,6)
ResnetModel


# In[ ]:


if torch.cuda.is_available():
    ResnetModel.cuda()


# In[ ]:


epochs = 3
train_losses =[]
test_losses = []
train_correct =[]
test_correct=[]
for epoch in range(epochs):
    
    trn_corr =0
    tst_corr =0
    for batch,(X_train,y_train) in enumerate(train_loader):
        batch+=1
        if torch.cuda.is_available():
            X_train = Variable(X_train).cuda()
            y_train = Variable(y_train).cuda()
        
        y_pred = ResnetModel(X_train)
        loss   = criterion(y_pred,y_train)
        predicted =torch.max(y_pred.data,1)[1]
        trn_corr+=(predicted == y_train).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%100 ==0:
            print(f'epoch:{epoch} batch:{batch} loss:{loss.item()} accuracy:{trn_corr.item()*100/(batch*10)}')
            
        
    train_losses.append(loss)
    train_correct.append(trn_corr)


# In[ ]:


plt.plot(train_losses,label = 'train loss')
plt.legend()


# ### Test your Model

# In[ ]:


## my CNN test
CNN_model.eval()
CNN_model.cuda()
with torch.no_grad():
    y_result =CNN_model(Variable(train_data[25][0].view(1,3,224,224)).cuda()).argmax()
print('predicted:{}'.format(y_result.item()))
print('label:',train_data[25][1])


# In[ ]:





# In[ ]:


## Testing Alexnet model
AlexNetModel.eval()
AlexNetModel.cuda()
with torch.no_grad():
    y_result =AlexNetModel(Variable(train_data[25][0].view(1,3,224,224)).cuda()).argmax()
print('predicted:{}'.format(y_result.item()))
print('label:',train_data[25][1])


# In[ ]:


plt.imshow(np.transpose(train_data[25][0],(1,2,0)))
print('label:',train_data[25][1])

