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
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm_notebook as tqdm
from imgaug import augmenters as iaa
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import cv2
import gc


# ## Make Train_Images/Test_Images Directory and Unzip

# In[ ]:


get_ipython().system('mkdir train_images')
get_ipython().system('mkdir test_images')
get_ipython().system('unzip /kaggle/input/2019-3rd-ml-month-with-kakr/train.zip -d train_images')
get_ipython().system('unzip /kaggle/input/2019-3rd-ml-month-with-kakr/test.zip -d test_images')


# ## Read Train.csv / Class.csv

# In[ ]:


train = pd.read_csv('/kaggle/input/2019-3rd-ml-month-with-kakr/train.csv')
classes = pd.read_csv('/kaggle/input/2019-3rd-ml-month-with-kakr/class.csv')


# In[ ]:


class_dict = classes.set_index('id').to_dict()['name']


# ### Dict List

# In[ ]:


class_dict


# ## One-Hot Encoding

# In[ ]:


train = pd.concat([train,pd.get_dummies(train['class'])],axis=1)


# ## Check The Number of Classes and Balance

# In[ ]:


train['class'].nunique()


# In[ ]:


train.groupby('class')['class'].count().plot(kind='bar',title='Classes Counting')


# ## Crop and Resize Images

# In[ ]:


class crop_resize(object):
    def __init__(self,df):
        self.df = df
        
            
    def __call__(self):
        if 'class' in self.df.columns:
            path = os.path.join(os.getcwd(),'train_images')
            save_path = os.path.join(os.getcwd(),'resized_train_images')
            if os.path.isdir('resized_train_images'):
                get_ipython().system("rm -r 'resized_train_images'")
            get_ipython().system('mkdir resized_train_images')
        else:
            path = os.path.join(os.getcwd(),'test_images')
            save_path = os.path.join(os.getcwd(),'resized_test_images')
            if os.path.isdir('resized_test_images'):
                get_ipython().system("rm -r 'resized_test_images'")
            get_ipython().system('mkdir resized_test_images')
        
        for fname in self.df.img_file:
            Image = cv2.imread(os.path.join(path,fname))
            x1,x2,y1,y2 = tuple(self.df.set_index('img_file').loc[fname, ['bbox_x1','bbox_x2','bbox_y1','bbox_y2']])
            h,w,_ = Image.shape #height x width x channels
            b1,b2 = x1-x2 , y1-y2
            padd_x1,padd_x2 = max(int(x1 - b1*0.01),0), min(int(x2 - b1*0.01),w-1)
            padd_y1,padd_y2 = max(int(y1 - b2*0.01),0), min(int(y2 - b2*0.01),h-1)
            #get crop it
            Image = Image[padd_y1:padd_y2,padd_x1:padd_x2]
            Image = cv2.resize(Image,(244,244))
            status = cv2.imwrite(os.path.join(save_path,fname),Image)
            print('save {} at {}'.format(fname,save_path))
        
            
            


# In[ ]:


preprocess = crop_resize(train)
preprocess()


# In[ ]:


gc.collect()


# ## Image Augmentation

# In[ ]:


class transforms(object):
    def __init__(self):
        #Scheme(Tentative)
        self.transforms = iaa.Sequential(
            [iaa.Fliplr(0.5),
             iaa.Sometimes(0.5,iaa.AdditiveGaussianNoise(loc=0, scale =(0.0, 0.05*255),per_channel=0.5)),
             iaa.Sometimes(0.5,iaa.ContrastNormalization((0.75,1.5))),
             iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0,0.5))),
             iaa.Sometimes(0.5,iaa.Affine(shear=(-5, 5))),
             iaa.Sometimes(0.5,iaa.Grayscale(alpha=(0.0, 1.0)))
            ],random_order=True)
        
    def __call__(self,Image,fname,df,preprocessed):
        if preprocessed:
            Image = self.transforms.augment_image(Image)
        else:
            #get bounding_box
            x1,x2,y1,y2 = tuple(df.set_index('img_file').loc[fname, ['bbox_x1','bbox_x2','bbox_y1','bbox_y2']])
            h,w,_ = Image.shape #height x width x channels
            b1,b2 = x1-x2 , y1-y2
            padd_x1,padd_x2 = max(int(x1 - b1*0.01),0), min(int(x2 - b1*0.01),w-1)
            padd_y1,padd_y2 = max(int(y1 - b2*0.01),0), min(int(y2 - b2*0.01),h-1)
            #get crop it
            Image = Image[padd_y1:padd_y2,padd_x1:padd_x2]
            #get transforms
            Image = self.transforms.augment_image(Image)
            #resize
            Image = cv2.resize(Image,(244,244))
        return Image


# ## Custom DataLoader

# In[ ]:


class CarDataset(Dataset):
    
    def __init__(self,df,transforms=None,preprocessed=True,root='/kaggle/input/'):
        self.transforms = transforms
        self.df = df
        self.preprocessed = preprocessed
        if 'class' in self.df:
            self.df = df
            self.classes = df.set_index('img_file')['class'].to_dict
            if self.preprocessed:
                self.path = os.path.join(os.getcwd(),'resized_train_images')
            else:
                self.path = os.path.join(os.getcwd(),'train_images')
        else:
            if self.preprocessed:
                self.path = os.path.join(os.getcwd(),'resized_test_images')
            else:
                self.path = os.path.join(os.getcwd(),'test_images')
            
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        if 'class' in self.df:
            #label = self.df.iloc[:,-196:].values[idx]
            label = self.df['class'].values[idx]-1
            fname = self.df.img_file.values[idx]
            Image = cv2.imread(os.path.join(self.path,fname))
            if self.transforms is not None:
                transform = self.transforms()
                Image = transform(Image,fname,self.df,self.preprocessed)
            else:
                Image = cv2.resize(Image,(244,244))
            return Image.astype(np.float)/255.0,label
        else:
            fname = self.df.img_file.values[idx]
            Image = cv2.imread(os.path.join(self.path,fname))
            #Image = cv2.resize(Image,(244,244))
            return np.transpose(Image.astype(np.float)/255.0,(2,1,0))
            


# ## Before Crop and Augmentation

# In[ ]:


train_images = CarDataset(train, preprocessed=False)
train_loader = torch.utils.data.DataLoader(train_images,batch_size=9,shuffle=True)


# In[ ]:


import matplotlib.pyplot as plt
a = next(iter(train_loader))
fig,ax = plt.subplots(3,3, figsize=(25,25))
for i in range(9):
    j = i//3
    k = i%3
    ax[j,k].imshow(a[0][i])
    ax[j,k].set_title(class_dict[int(torch.max(a[1][i],0)[1].detach().numpy())+1],fontsize= 15)
plt.show()


# ## After Crop and Augmentation

# In[ ]:


train_images = CarDataset(train,transforms=transforms)
train_loader = torch.utils.data.DataLoader(train_images,batch_size=9,shuffle=True)


# In[ ]:


a = next(iter(train_loader))
fig,ax = plt.subplots(3,3, figsize=(25,25))
for i in range(9):
    j = i//3
    k = i%3
    ax[j,k].imshow(a[0][i])
    ax[j,k].set_title(class_dict[int(torch.max(a[1][i],0)[1].detach().numpy())+1],fontsize= 15)
plt.show()


# ## Build Custom Model

# In[ ]:


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self,size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        size = size or (1,1)
        self.avgpool = nn.AdaptiveAvgPool2d(size)
        self.maxpool = nn.AdaptiveMaxPool2d(size)
        
    def forward(self,x):
        return torch.cat([self.maxpool(x),self.avgpool(x)],1)


# In[ ]:


class ResNeXt50(nn.Module):
    def __init__(self,pretrained=True):
        super(ResNeXt50,self).__init__()
        encoder = models.resnext50_32x4d(pretrained=pretrained)#,progress=False)
        encoder = nn.Sequential(*list(encoder.children()))
        
        # cut tail
        self.cnn = nn.Sequential(
            encoder[0],
            encoder[1],
            encoder[2],
            encoder[3],
            encoder[4],
            encoder[5],
            encoder[6],
            encoder[7],
        )
        '''
        # freeze weight
        print('Freeze Pretrained model')
        for param in self.cnn.parameters():
            print(f'before {param.requires_grad}')
            param.requires_grad = False
            print(f'after {param.requires_grad}')
        '''
            
        
        # add layers
        self.clf = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(4096,512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512,196)
        )
        
    def forward(self,x):
        x = self.cnn(x)
        x = self.clf(x)
        return x


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNeXt50().to(device)


# In[ ]:


gc.collect()


# ## Check Model Summary

# In[ ]:


get_ipython().system('pip install torchsummary')
from torchsummary import summary
summary(model, input_size=(3,224,224))


# ## Custom Learner

# In[ ]:


gc.collect()


# ### Strafified split(6:4) X
# ### Strafified split(9:1) O

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


criterion = nn.CrossEntropyLoss(reduction = 'mean')


# In[ ]:


class Learner(object):
    
    def __init__(self):
        
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []
        
    def fit(self,epochs=5,batch_size=64,shuffle=True):
        model.to(device)
        self.train(model=model,epochs=epochs,batch_size=batch_size,shuffle=shuffle)
        
        return self.train_losses,self.valid_losses,self.train_accs,self.valid_accs
    
    def train(self,model,epochs,batch_size=32,shuffle=True):
        optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.05)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_f1 = 0.0
            running_acc = 0.0
            print(f'epoch {epoch+1}/{epochs}')
            x_train, x_valid, _, _ = train_test_split(train, train['class'],
                                                    stratify=train['class'], 
                                                    test_size=0.1)
            train_set = CarDataset(x_train,transforms=transforms)
            train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=shuffle)
            for idx,(inputs,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs.permute(0,3,2,1).float())
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss
                running_acc += (outputs.argmax(dim=1) == labels).float().mean()
            self.train_losses.append(running_loss/len(train_loader))
            self.train_accs.append(running_acc/len(train_loader))
            if epoch%5 == 3:
            #if epoch == epoch: #for monitoring
                gc.collect()
                val_loss , val_acc = self.valid(model=model,shuffle=shuffle,x_valid=x_valid)
                print('train_loss : {:.2f} | train_acc : {:.2f} | valid_loss : {:.2f} | valid_acc : {:.2f}'.format(running_loss/len(train_loader),running_acc/len(train_loader)
                                                                                                                         ,val_loss,val_acc))
    def valid(self,model,batch_size=200,shuffle=True,x_valid=None):
        valid_set = CarDataset(x_valid,transforms=transforms)
        valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=batch_size,shuffle=shuffle)
        model.eval()
        running_loss = 0.0
        running_f1 = 0.0
        running_acc = 0.0
        with torch.no_grad():
            for idx,(inputs,labels) in tqdm(enumerate(valid_loader),total=len(valid_loader)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs.permute(0,3,2,1).float())
                loss = criterion(outputs,labels)
                running_loss += loss
                running_acc += (outputs.argmax(dim=1) == labels).float().mean()
        self.valid_losses.append(running_loss/len(valid_loader))
        self.valid_accs.append(running_acc/len(valid_loader))
        return running_loss/len(valid_loader), running_acc/len(valid_loader)
            


# In[ ]:


learner = Learner()
train_losses, valid_losses, train_accs, valid_accs = learner.fit(epochs=10)


# In[ ]:


fig, axs = plt.subplots(2, 2, figsize=(15, 5))
axs[0,0].plot(train_losses)
axs[0,0].set_title('Train Loss')
axs[0,1].plot(train_accs)
axs[0,1].set_title('Train acc')
axs[1,0].plot(valid_losses)
axs[1,0].set_title('Valid Loss')
axs[1,1].plot(valid_accs)
axs[1,1].set_title('Valid acc')
fig.tight_layout()


# In[ ]:


test = pd.read_csv('/kaggle/input/2019-3rd-ml-month-with-kakr/test.csv')


# In[ ]:


preprocess = crop_resize(test)
preprocess()


# In[ ]:


test_images = CarDataset(test)
test_loader = torch.utils.data.DataLoader(test_images,batch_size=9,shuffle=False)
prediction = []
model.eval()
with torch.no_grad():
    
    for idx, (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):
        inputs.to(device)
        
        outputs = model(inputs.float().cuda())
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        prediction.append(preds)
    prediction = np.hstack(prediction)


# In[ ]:


prediction = np.hstack(prediction)


# In[ ]:


submission = pd.read_csv('/kaggle/input/2019-3rd-ml-month-with-kakr/sample_submission.csv')


# In[ ]:


submission['class'] =prediction + 1 # correction
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission


# In[ ]:


test_images = CarDataset(test)
test_loader = torch.utils.data.DataLoader(test_images,batch_size=9,shuffle=True)


# In[ ]:


a = next(iter(train_loader))
fig,ax = plt.subplots(3,3, figsize=(25,25))
for i in range(9):
    j = i//3
    k = i%3
    ax[j,k].imshow(a[0][i])
    ax[j,k].set_title(class_dict[int(torch.max(a[1][i],0)[1].detach().numpy())+1],fontsize= 15)
plt.show()


# ### Incomplete ver.
# To do.
# * Fine-Tuning
# * Evaluation //
# * Submision //
# 
# - unfreeze
# - apply different learning rate for each layers.
# 
# Any advices will be appreciated.

# In[ ]:


get_ipython().system('rm -r train_images')
get_ipython().system('rm -r resized_train_images')
get_ipython().system('rm -r test_images')
get_ipython().system('rm -r resized_test_images')

