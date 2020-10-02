#!/usr/bin/env python
# coding: utf-8

# **Tried using tta with single run epoch model of efficient-net b5**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn,optim
import random
import os
get_ipython().system('pip install tqdm')
from tqdm import tqdm
get_ipython().system('pip install albumentations')
import albumentations as aug
from albumentations.pytorch.transforms import ToTensor
get_ipython().system('pip install timm')
import timm
get_ipython().system('pip install wtfml')
from wtfml.utils import EarlyStopping
get_ipython().system('pip install pytorch_ranger')
from pytorch_ranger import Ranger
import warnings
warnings.simplefilter('ignore')


# In[ ]:


get_ipython().system('pip install efficientnet_pytorch')


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b5',num_classes=1)
model.to(device)


# In[ ]:


model = torch.load('/kaggle/input/effnetb5/modeleffnet-b5.pth')


# In[ ]:


test_df = pd.read_csv('/kaggle/input/dataset/test_dum.csv')


# In[ ]:


test_pred=test_df[['image_name']]
test_pred['target']=pd.DataFrame(np.zeros((test_df.shape[0],1)))
test_pred[['sex_female', 'sex_male', 'age_approx_Middle',
       'age_approx_Old', 'age_approx_Young',
       'anatom_site_general_challenge_head/neck',
       'anatom_site_general_challenge_lower extremity',
       'anatom_site_general_challenge_oral/genital',
       'anatom_site_general_challenge_palms/soles',
       'anatom_site_general_challenge_torso',
       'anatom_site_general_challenge_upper extremity']]=test_df[['sex_female', 'sex_male', 'age_approx_Middle',
       'age_approx_Old', 'age_approx_Young',
       'anatom_site_general_challenge_head/neck',
       'anatom_site_general_challenge_lower extremity',
       'anatom_site_general_challenge_oral/genital',
       'anatom_site_general_challenge_palms/soles',
       'anatom_site_general_challenge_torso',
       'anatom_site_general_challenge_upper extremity']]


# In[ ]:


get_ipython().system('pip install ttach')
import ttach as tta


# In[ ]:


transforms = tta.Compose(
    [
        
        tta.Rotate90(angles=[0, 180])
        
    ]
)

tta_model = tta.ClassificationTTAWrapper(model, transforms,merge_mode='mean')


# In[ ]:


import cv2


class Classify(Dataset):
    
    def __init__(self,df,phase):
        self.phase = phase
        self.df = df
        if phase == 'train':
            self.transforms = aug.Compose([
            aug.Resize(380,380,p=1),
            aug.Flip(),
            aug.GridDistortion(),
            #aug.HorizontalFlip(p=0.5),
            #aug.RandomBrightness(limit=10,p=0.5),
            aug.GaussNoise(),
            #aug.Cutout(num_holes=4, max_h_size=64, max_w_size=64,p=0.5),
            aug.RandomGamma(p=0.3),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ToTensor()
            ])
        elif phase == 'val':
            self.transforms = aug.Compose([
                aug.Resize(380,380),
                aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        elif phase == 'test':
            self.transforms = aug.Compose([
                aug.Resize(380,380),
                aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ToTensor()
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        name = self.df.iloc[index,0]
        target = self.df.target[index]
        if self.phase=='train':
          path = '/kaggle/input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/' + str(name) +'.jpg'
        elif self.phase=='val':
          path = '/kaggle/input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/' + str(name) +'.jpg'
        elif self.phase=='test':
          path = '/kaggle/input/melanoma-merged-external-data-512x512-jpeg/512x512-test/512x512-test/'+ str(name) +'.jpg'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        img = self.transforms(image = img)
        img = img['image']
        return (img,target)
        


# In[ ]:


dataset_test=Classify(test_pred,'test')
test_dataloader=torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,num_workers=4)
tta_model.train(False)
valid_preds2=[]
tk = tqdm(test_dataloader, total=len(test_dataloader), position=0, leave=True)
for i,(inputs,labels)in enumerate(tk):
  labels=labels.type(torch.float32)
  inputs,labels=inputs.to(device),labels.to(device)
  outputs=tta_model(inputs)
  pred = torch.sigmoid(outputs.view(labels.shape[0],))
  valid_preds2.extend(pred.detach().cpu().numpy())


# In[ ]:


df=pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")
df['target']=valid_preds2
df.to_csv('submission.csv',index=False)

