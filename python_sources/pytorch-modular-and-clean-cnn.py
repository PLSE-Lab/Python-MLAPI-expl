#!/usr/bin/env python
# coding: utf-8

# # PyTorch GPU based CNN using BCELoss and torchvision import transforms
# 
# This is my fifth generation PyTorch script which includes many improvements over the previous versions. 
# The previous versions are here:
# 
# - https://www.kaggle.com/solomonk/pytorch-gpu-cnn-bceloss-0-2198-lb
# 
# - https://www.kaggle.com/solomonk/pytorch-gpu-based-cnn-bceloss-with-predictions
#  
# Improvements include:
# 1. Automatic calculation of the FC layer size
# 2. A nice fit() methood which also does validation (but albeit is slower) 
# 3. Saving and loading the CNN model 
# 
# Todo:
# 1. Add image transforms, see:  https://discuss.pytorch.org/t/applying-an-image-transform-data-augumentations-to-a-2d-floattensor-pil/9359 
# 
# Comments are welocmed, 
# 
# ## Updates 
# - Update 2/11/2017:
# Added IcebergCustomDataSet and torchvision  transforms using https://www.kaggle.com/supersp1234/tools-for-pytorch-transform
# 
# 

# In[1]:


get_ipython().run_line_magic('reset', '-f')
import torch
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)

import numpy
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

import pandas
import pandas as pd

import logging
handler=logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
get_ipython().run_line_magic('matplotlib', 'inline')

# !pip install psutil
import psutil
import os
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

cpuStats()


# # Concatenate and Reshape
# Here we load the data and then combine the two bands and recombine them into a single image/tensor for training

# In[2]:


# Data params
TARGET_VAR= 'target'
BASE_FOLDER = '../input/'
data = pd.read_json(BASE_FOLDER + '/train.json')

data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

print (type(data))

# Suffle
import random
from datetime import datetime
random.seed(datetime.now())
from sklearn.utils import shuffle
# data = shuffle(data) # otherwise same validation set each time!
# data= data.reindex(np.random.permutation(data.index))

# data= data.reindex(np.random.permutation(data.index))
# data = shuffle(data) # otherwise same validation set each time!

band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
full_img = np.stack([band_1, band_2], axis=1)


from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import cv2 

batch_size=64


# # Custom PyTorch Dataset to enable applying image Transforms
# - Since we have a non regular image type, a custom Dataset has to be written (adapted from:https://www.kaggle.com/supersp1234/tools-for-pytorch-transform and https://www.kaggle.com/heyt0ny/pytorch-custom-dataload-with-augmentaion)
# - This is required for enrichment 
# 

# In[3]:


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets, models
import random
import PIL
from PIL import Image, ImageOps
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class IcebergCustomDataSet(Dataset):
    """total dataset."""

    def __init__(self, data, labels,transform=None):
        self.data= data
        self.labels = labels
        self.transform = transform        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx,:,:,:], 'labels': np.asarray([self.labels[idx]])}
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        image = image.astype(float)/255
        return {'image': torch.from_numpy(image.copy()).float(),
                'labels': torch.from_numpy(labels).float()
               }
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels = sample['image'], sample['labels']
        
        if random.random() < 0.5:
            image=np.flip(image,1)
        
        return {'image': image, 'labels': labels}
    
class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):     
        image, labels = sample['image'], sample['labels']
        if random.random() < 0.5:
            image=np.flip(image,0)
        return {'image': image, 'labels': labels} 

class RandomTranspose(object):
    def __call__(self, sample):     
        image, labels = sample['image'], sample['labels']
        if random.random() < 0.7: 
            image=np.transpose(image,0)
        return {'image': image, 'labels': labels} 

class Normalize(object):   
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):      
        # TODO: make efficient
        img=tensor['image'].float()
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': img, 'labels': tensor['labels']}  

from random import randrange    
random.seed(datetime.now()) # re seed 

X_train,X_val,y_train,y_val=train_test_split(full_img,data['is_iceberg'].values,
                                                   test_size=0.33, 
                                                   random_state=randrange(50000))


# val_dataset = IcebergCustomDataSet(X_val, y_val, transform=transforms.Compose([
#                                                               ToTensor(), 
#                                                               ])) 
# train_dataset = IcebergCustomDataSet(X_train, y_train, transform=transforms.Compose([
#                                                               ToTensor(), 
#                                                               ])) 


train_ds = IcebergCustomDataSet(X_train, y_train, transform=transforms.Compose([
                                                              RandomHorizontalFlip(), 
                                                              RandomVerticallFlip(),
                                                              
                                                              ToTensor(), 
                                                              ])) 

val_dataset = IcebergCustomDataSet(X_val, y_val, 
                                transform=transforms.Compose([
                                                              RandomHorizontalFlip(), 
                                                              RandomVerticallFlip(), 
                                                               
                                                              ToTensor(), 
                                                              ])) 

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=1)

print (train_loader)
print (val_loader)    


# # Train/Validation split 
# (Not currently in use old version that did not involve image transforms)

# In[4]:


# # Data params
# TARGET_VAR= 'target'
# BASE_FOLDER = '../input/'


# data = pd.read_json(BASE_FOLDER + '/train.json')

# data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
# data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
# data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

# # Suffle
# import random
# from datetime import datetime
# random.seed(datetime.now())
# # np.random.seed(datetime.now())
# from sklearn.utils import shuffle
# data = shuffle(data) # otherwise same validation set each time!
# data= data.reindex(np.random.permutation(data.index))

# band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
# band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
# full_img = np.stack([band_1, band_2], axis=1)

# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def XnumpyToTensor(x_data_np):
    x_data_np = np.array(x_data_np, dtype=np.float32)        
    print(x_data_np.shape)
    print(type(x_data_np))

    if use_cuda:
        lgr.info ("Using the GPU")    
        X_tensor = (torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    
    else:
        lgr.info ("Using the CPU")
        X_tensor = (torch.from_numpy(x_data_np)) # Note the conversion for pytorch
        
    print((X_tensor.shape)) # torch.Size([108405, 29])
    return X_tensor


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def YnumpyToTensor(y_data_np):    
    y_data_np=y_data_np.reshape((y_data_np.shape[0],1)) # Must be reshaped for PyTorch!
    print(y_data_np.shape)
    print(type(y_data_np))

    if use_cuda:
        lgr.info ("Using the GPU")            
    #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float        
    else:
        lgr.info ("Using the CPU")        
    #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #         
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float        

    print(type(Y_tensor)) # should be 'torch.cuda.FloatTensor'
    print(y_data_np.shape)
    print(type(y_data_np))    
    return Y_tensor


# #  Custom data loader

# In[17]:

# class FullTrainningDataset(torch.utils.data.Dataset):
#     def __init__(self, full_ds, offset, length):
#         self.full_ds = full_ds
#         self.offset = offset
#         self.length = length
#         assert len(full_ds)>=offset+length, Exception("Parent Dataset not long enough")
#         super(FullTrainningDataset, self).__init__()
        
#     def __len__(self):        
#         return self.length
    
#     def __getitem__(self, i):
#         return self.full_ds[i+self.offset]
    
# validationRatio=0.22    

# def trainTestSplit(dataset, val_share=validationRatio):
#     val_offset = int(len(dataset)*(1-val_share))
#     print ("Offest:" + str(val_offset))
#     return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, 
#                                                                               val_offset, len(dataset)-val_offset)
# batch_size=32

# from torch.utils.data import TensorDataset, DataLoader

# # train_imgs = torch.from_numpy(full_img_tr).float()
# train_imgs=XnumpyToTensor (full_img)
# train_targets = YnumpyToTensor(data['is_iceberg'].values)
# dset_train = TensorDataset(train_imgs, train_targets)


# train_ds, val_ds = trainTestSplit(dset_train)

# train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False,
#                                             num_workers=1)
# val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)

# print (train_loader)
# print (val_loader)


# # CNN

# In[ ]:


import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# See https://github.com/kimhc6028/forward-thinking-pytorch/blob/master/forward_thinking.py for a great example    
# loss_func=torch.nn.BCELoss() # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
# dropout = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# See https://github.com/kimhc6028/forward-thinking-pytorch/blob/master/forward_thinking.py
def cnnBlock(in_planes, out_planes,kernel_size=7, padding=2,pool_size=2):
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,padding=padding)
        bn   = torch.nn.BatchNorm2d(out_planes)
        relu = torch.nn.LeakyReLU()        
        pl   = torch.nn.MaxPool2d(pool_size,pool_size)
        av   = torch.nn.AvgPool2d(pool_size,pool_size)
#         dr   = torch.nn.Dropout(d_rate)
        return nn.Sequential(conv, bn, relu,pl,av)
                
dropout = [0.65, 0.55, 0.30, 0.20, 0.10, 0.05]

class CNNClassifier(torch.nn.Module):        
    def __init__(self, img_size, img_ch, kernel_size, pool_size, n_out, padding):
        super(CNNClassifier, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        
        self.n_out = n_out
        self.sig=torch.nn.Sigmoid()
        self.all_losses = []
        self.val_losses = []  
        self.cnn_features = []
        self.layers = []
        self.build_model()        
#         print (self)
    # end constructor
        
    def build_model(self):           
        self.conv1=cnnBlock(self.img_ch, 16, kernel_size=self.kernel_size,padding=self.padding)        
        self.conv2=cnnBlock(16, 32, kernel_size=5,padding=self.padding)
        self.conv3=cnnBlock(32, 64, kernel_size=3,padding=self.padding)
        
        self.cnn_features = [self.conv1, 
                             self.conv2,
                             self.conv3,
                            ]                
        self.fc = nn.Sequential(
            nn.Linear(64, self.n_out),
        )
        
        self.criterion = torch.nn.BCELoss()          
        LR = 0.0005        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR,weight_decay=5e-5) #  L2 regularization
    # end method build_model

    def forward(self, x):
        for c in self.cnn_features:
            x = (c(x))
        x= self.shrink(x)
        x= self.fc(x)
        return self.sig(x)
    # end method forward

    def shrink(self, X):
        return X.view(X.size(0), -1)
    # end method flatten

    def fit(self,loader, num_epochs, batch_size):               
        self.train()
        for epoch in range(num_epochs):
            self.train()
            print('Epoch {}'.format(epoch + 1))
            print('*' * 5 + ':')
            running_loss = 0.0
            running_acc = 0.0            
    
            for i, dict_ in enumerate(loader):
                images  = dict_['image']
                target  = dict_['labels']
#                 images, target=dict_
#                 self.train()
                inputs = torch.autograd.Variable(images)
                labels = torch.autograd.Variable(target)                
        
                preds = self.forward(inputs)            # cnn output
                loss = self.criterion(preds, labels)    # cross entropy loss
                running_loss += loss.data[0] * labels.size(0)
                self.optimizer.zero_grad()              # clear gradients for this training step
                loss.backward()                         # backpropagation, compute gradients
                self.optimizer.step()                   # apply gradients
                preds = torch.max(preds, 1)[1].data.numpy().squeeze()
                acc = (preds == target.numpy()).mean()
                if (i+1) % 10 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f'
                           %(epoch+1, num_epochs, i+1, 
                             int(len(train_ds)/batch_size), loss.data[0], acc)) 
                    
            #save model
            torch.save(self.state_dict(), './cnn.pth')
            #Cross validation
            self.LeavOneOutValidation(val_loader)            
        torch.save(self.state_dict(), './cnn.pth')
    # end method fit
    
    def LeavOneOutValidation(self, val_loader): 
        print ('Leave one out VALIDATION ...')
        model = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, 
                            pool_size=pool_size, n_out=n_out, padding=padding)
        # .. to load your previously training model:
        model.load_state_dict(torch.load('./cnn.pth'))
        val_losses = []
        model.eval()
        print (val_loader)
        eval_loss = 0
        eval_acc = 0
        for data in val_loader:        
            img  = data['image']
            label  = data['labels']
#             img, label=data
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)

            out = model(img)
            loss = model.criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)

        print('Leave one out VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_dataset))))
        val_losses.append(eval_loss / (len(val_dataset)))
        print()    
        
    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    # end method gen_batch
    
   
    # end class CNNClassifier


# # Train

# In[ ]:


img_size = (75,75)
img_ch = 2
kernel_size = 7
pool_size = 2
padding=2
n_out = 1
n_epoch = 35

if __name__ == '__main__':
    cnn = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, 
                        pool_size=pool_size, n_out=n_out, padding=padding)
    cnn.fit(train_loader,n_epoch, batch_size)
#     cnn.evaluate(val_loader, batch_size=8)


# 

# In[ ]:


from sklearn.cross_validation import train_test_split

# def kFoldValidation(folds): 
#     print ('K FOLD VALIDATION ...')
#     cnn = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, 
#                         pool_size=pool_size, n_out=n_out, padding=padding)
#     # .. to load your previously training model:
#     model.load_state_dict(torch.load('./cnn.pth'))
#     val_losses = []
#     model.eval()
    
#     for e in range(folds):
#         print ('Fold:' + str(e))        
#         data = pd.read_json('../input/train.json')        
#         data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
#         data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
#         data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
#         band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
#         band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
#         full_img = np.stack([band_1, band_2], axis=1)
                                
#         X_train,X_val,y_train,y_val=train_test_split(full_img,data['is_iceberg'].values,
#                                                    test_size=0.11, 
#                                                    random_state=randrange(3000))
#         # Only need val set
#         val_dataset_kfold = IcebergCustomDataSet(X_val, y_val, 
#                                 transform=transforms.Compose([RandomHorizontalFlip(), 
#                                                               RandomVerticallFlip(), 
#                                                               ToTensor(), 
#                                                               ])) 
#         val_loader_kfold = DataLoader(dataset=val_dataset, batch_size=batch_size,
#                           shuffle=True, num_workers=1)
        
#         print (val_loader_kfold)

#         eval_loss = 0
#         eval_acc = 0
#         for data in val_loader:
#             img  = data['image']
#             label  = data['labels']

#             img = Variable(img, volatile=True)
#             label = Variable(label, volatile=True)

#             out = model(img)
#             loss = model.criterion(out, label)
#             eval_loss += loss.data[0] * label.size(0)

#         print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_dataset_kfold))))
#         val_losses.append(eval_loss / (len(val_dataset_kfold)))
#         print()
    
def LeavOneOutValidation(val_loader): 
    print ('Leave one out VALIDATION ...')
    model = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, 
                        pool_size=pool_size, n_out=n_out, padding=padding)
    # .. to load your previously training model:
    model.load_state_dict(torch.load('./cnn.pth'))
    val_losses = []
    model.eval()        
    print (val_loader)
    eval_loss = 0
    eval_acc = 0
    for data in val_loader:        
        img  = data['image']
        label  = data['labels']
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
        out = model(img)
        loss = model.criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
    print('Leave one out VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_dataset))))
    val_losses.append(eval_loss / (len(val_dataset)))
    print()
    print()        
    
LeavOneOutValidation(val_loader)    


# In[ ]:


# kFoldValidation(10)


# # Make Predictions
# Here we make predictions on the output and export the CSV so we can submit

# In[ ]:


# load the model
# model=torch.load('./cnn.pth')
model = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, 
                        pool_size=pool_size, n_out=n_out, padding=padding)
# .. to load your previously training model:
model.load_state_dict(torch.load('./cnn.pth'))
print (model)

df_test_set = pd.read_json('../input/test.json')

df_test_set['band_1'] = df_test_set['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
df_test_set['band_2'] = df_test_set['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
df_test_set['inc_angle'] = pd.to_numeric(df_test_set['inc_angle'], errors='coerce')

df_test_set.head(3)


print (df_test_set.shape)
columns = ['id', 'is_iceberg']
df_pred=pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
# df_pred.id.astype(int)

for index, row in df_test_set.iterrows():
    rwo_no_id=row.drop('id')    
    band_1_test = (rwo_no_id['band_1']).reshape(-1, 75, 75)
    band_2_test = (rwo_no_id['band_2']).reshape(-1, 75, 75)
    full_img_test = np.stack([band_1_test, band_2_test], axis=1)

    x_data_np = np.array(full_img_test, dtype=np.float32)        
    if use_cuda:
        X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    
    else:
        X_tensor_test = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch
                    
#     X_tensor_test=X_tensor_test.view(1, trainX.shape[1]) # does not work with 1d tensors            
    predicted_val = (model(X_tensor_test).data).float() # probabilities     
    p_test =   predicted_val.cpu().numpy().item() # otherwise we get an array, we need a single float
    
    df_pred = df_pred.append({'id':row['id'], 'is_iceberg':p_test},ignore_index=True)
#     df_pred = df_pred.append({'id':row['id'].astype(int), 'probability':p_test},ignore_index=True)

df_pred.head(5)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(model.all_losses)
plt.show()


# In[ ]:


# df_pred.id=df_pred.id.astype(int)

def savePred(df_pred):
#     csv_path = 'pred/p_{}_{}_{}.csv'.format(loss, name, (str(time.time())))
#     csv_path = 'pred_{}_{}.csv'.format(loss, (str(time.time())))
    csv_path='sample_submission.csv'
    df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
    print (csv_path)
    
savePred (df_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




