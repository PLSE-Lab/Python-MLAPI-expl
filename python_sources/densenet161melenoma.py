#!/usr/bin/env python
# coding: utf-8

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


#Seed everything at ones for reproducibility
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


test_df = pd.read_csv('/kaggle/input/traintest/test_dum.csv')
train_df = pd.read_csv('/kaggle/input/traintest/train_dum.csv')


# In[ ]:


Train_dir='/kaggle/input/siic-isic-224x224-images/train/'
Test_dir='/kaggle/input/siic-isic-224x224-images/test/'


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


class Classify(Dataset):
    
    def __init__(self,df,phase):
        self.phase = phase
        self.df = df
        if phase == 'train':
            self.transforms = aug.Compose([
            aug.Flip(),
            aug.GridDistortion(),
            aug.Cutout(),
            aug.RandomGamma(p=0.4),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ToTensor()
            ])
        elif phase == 'val':
            self.transforms = aug.Compose([
                aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ToTensor()
            ])
        elif phase == 'test':
            self.transforms = aug.Compose([
                aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ToTensor()
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        name = self.df.iloc[index,0]
        target = self.df.target[index]
        if self.phase=='train':
          path = Train_dir + str(name) +'.png'
        elif self.phase=='val':
          path = Train_dir + str(name) +'.png'
        elif self.phase=='test':
          path = Test_dir+ str(name) +'.png'
        img = plt.imread(path)*255.0
        img = self.transforms(image = np.array(img))
        img = img['image']
        return (img,target)


# In[ ]:


train_df["kfold"] = -1    
train_df = train_df.sample(frac=1).reset_index(drop=True)
y = train_df.target.values
kf = model_selection.StratifiedKFold(n_splits=4)

for f, (t_, v_) in enumerate(kf.split(X=train_df, y=y)):
    train_df.loc[v_, 'kfold'] = f

train_df.to_csv("train_folds.csv", index=False)


# In[ ]:


dfs={}
for fol in range(4):
  dfs['train'+str(fol)]=train_df[train_df.kfold != fol].reset_index(drop=True).drop(['kfold'],axis=1)
  dfs['val'+str(fol)]=train_df[train_df.kfold == fol].reset_index(drop=True).drop(['kfold'],axis=1)


# In[ ]:


dats={}
for fol in range(4):
  dats['train'+str(fol)]=Classify(dfs['train'+str(fol)],'train')
  dats['val'+str(fol)]=Classify(dfs['val'+str(fol)],'val')


# In[ ]:


dloader={}
for fol in range(4):
  dloader['d'+str(fol)]={'train':torch.utils.data.DataLoader(dats['train'+str(fol)],batch_size=32,shuffle=True,num_workers=4),
  'val':torch.utils.data.DataLoader(dats['val'+str(fol)],batch_size=16,shuffle=False,num_workers=4)}


# In[ ]:


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# In[ ]:


def train_model(part,dataloader, model, criterion, optimizer, scheduler, num_epochs=15):
    best_model_wts = model.state_dict()
    best_auc=0
    dataset_sizes = {'train': len(dataloader['train'].dataset), 
                     'val': len(dataloader['val'].dataset)}
          
    for epoch in range(num_epochs):
        valid_preds, valid_targets = [], []
        for phase in ['train','val']:
            if phase=='train':
                model.train(True)
            else:
                model.train(False)
            running_loss=0.0
            running_correct=0
            tk = tqdm(dataloader[phase], total=len(dataloader[phase]), position=0, leave=True)
            for i,(inputs,labels) in enumerate(tk):
                labels=labels.type(torch.float32)
                inputs,labels=inputs.to(device),labels.to(device)
                optimizer.zero_grad()
                if phase=='train':
                    outputs=model(inputs)
                    pred = torch.sigmoid(outputs.view(labels.shape[0],))
                    loss=criterion(outputs.view(labels.shape[0],), labels)
                    loss.backward()
                    optimizer.step()                    
                else:
                    with torch.no_grad():
                        outputs=model(inputs)
                        pred = torch.sigmoid(outputs.view(labels.shape[0],))
                        loss=criterion(outputs.view(labels.shape[0],), labels)
                    valid_preds.extend(pred.detach().cpu().numpy())
                    valid_targets.extend(labels.detach().cpu().numpy())
                running_loss+=(loss.data).item()
                running_correct+=torch.sum(torch.round(pred)==labels)
            if phase=='train':
                train_epoch_loss=float(running_loss)/dataset_sizes[phase]
                train_epoch_acc=float(running_correct)/dataset_sizes[phase]
            else:
                valid_epoch_loss=float(running_loss)/dataset_sizes[phase]
                valid_epoch_acc=float(running_correct)/dataset_sizes[phase]
                auc =  roc_auc_score(valid_targets, valid_preds) 
                scheduler.step(auc)
                es(auc, model, f'/kaggle/working/{part}epoch.pth')
        if es.early_stop:
          print("Early stopping")
          break
        print('Current LR: {:.6f}'.format(get_lr(optimizer)))
        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} valid loss: {:.4f} acc: {:.4f}'.format(
          epoch, num_epochs - 1,
          train_epoch_loss, train_epoch_acc, 
          valid_epoch_loss, valid_epoch_acc))
        print()
    return model


# In[ ]:


model1=timm.create_model('densenet161',pretrained=True,num_classes=1)
model1.to(device)
es = EarlyStopping(patience=8, mode="max")
optimizer_b0 = Ranger(model1.parameters(),lr=1e-3)
criterion_b0 = nn.BCEWithLogitsLoss()
scheduler_b0 = ReduceLROnPlateau(optimizer_b0,factor=0.33, mode="max", patience=4)
model1=train_model(0,dloader['d0'],model1,criterion_b0,optimizer_b0,scheduler_b0,num_epochs=50)


# In[ ]:




