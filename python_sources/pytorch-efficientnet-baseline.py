#!/usr/bin/env python
# coding: utf-8

# # 1. Install

# In[ ]:


get_ipython().system('pip install efficientnet_pytorch')


# # 2. Import Library

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn
import gc
import cv2
import random
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch import optim
import torchvision
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss

from efficientnet_pytorch import EfficientNet

import sys
sys.path.append('../input/autoaug')
from auto_augment import AutoAugment
from tqdm import tqdm


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
seed_everything(2020)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


# # 3. Config

# In[ ]:


class Config:
    train_batch_size = 64
    test_batch_size = 32
    epochs = 15
    lr = 1e-4
    
config = Config()


# # 4. Data

# ## 4.1 Path and csv data

# In[ ]:


train_path = '../input/siic-isic-224x224-images/train'
test_path = '../input/siic-isic-224x224-images/test'
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


train_df['image_path'] = train_path +'/'+train_df['image_name']+'.png'
test_df['image_path'] = test_path +'/'+test_df['image_name']+'.png'
train_df.head()


# In[ ]:


anatom = pd.get_dummies(train_df['anatom_site_general_challenge'], prefix='anotom')
anatom_cols = list(anatom.columns)
sex = pd.get_dummies(train_df['sex'], prefix='sex')
sex_cols = list(sex.columns)
meta_cols = anatom_cols + sex_cols + ['age_approx']
pd.concat([train_df,sex,anatom],axis=1)


# ## 4.2 Data Transform

# In[ ]:


train_transform = transforms.Compose([
    #transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
    #transforms.RandomAffine(90, translate=(0.1,0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    AutoAugment(),
    #transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
    #transforms.Cutout(scale=(0.05, 0.1), value=(0, 0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


# ## 4.3 Data Helper

# In[ ]:


class Data(Dataset):
    def __init__(self,df=train_df,is_train=True,transform=None):
        super(Data,self).__init__()
        self.df = df
        self.is_train = is_train
        self.transform = transform
        pass
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,i):
        path = self.df.image_path.iloc[i]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToPILImage()(img)
        img = self.transform(img)
        
        if self.is_train:
            y = self.df.target.iloc[i]
            return img,y
        else:
            return img
    


# In[ ]:


kf = StratifiedKFold(5,shuffle=True,random_state=0)

test_data = Data(test_df,is_train=False,transform=test_transform)
test_loader = DataLoader(test_data,batch_size=config.test_batch_size,shuffle=False,num_workers=6)


# # 5. Model

# In[ ]:


class MyModel(nn.Module):
    def __init__(self,):
        super(MyModel,self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone._fc = nn.Linear(in_features=1280,out_features=1,bias=True)
    
    def forward(self,x):
        x = self.backbone(x)
        x = torch.sigmoid(x)
        return x


# # 6. Function for train

# In[ ]:


class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.sum = 0
        self.n = 0
        
    def update(self,val,n=1):
        self.sum+=val*n
        self.n+=n
        self.avg = self.sum/self.n


# In[ ]:


def eval_model(val_loader,model):
    model.eval()
    metrics = {}
    preds = []
    targets = []
    avg_loss = AverageMeter()
    with torch.no_grad():
        for it,(imgs,target) in enumerate(val_loader):
            imgs,target = imgs.cuda(),target.cuda().float().unsqueeze(1)
            pred = model(imgs)
            avg_loss.update(nn.BCELoss()(pred,target),imgs.size(0))
            preds.extend(pred.detach().cpu())
            targets.extend(target.detach().cpu())
    
    acc = accuracy_score(targets,np.round(preds))
    auc = roc_auc_score(targets,preds)
    return {'loss':avg_loss.avg,'acc':acc,'auc':auc}
    
def train_model(trn_loader,val_loader,model,optimizer,criterion,fold):
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        threshold=0.0001,
        mode="min"
    )
    optimizer.zero_grad()
    
    metrics = defaultdict(list)
    best_model = None
    best_auc = 0
    
    es = 0
    for epoch in range(config.epochs):
        model.train()
        avg_loss,avg_auc = AverageMeter(),AverageMeter()
        
        print('Epoch : {}'.format(epoch + 1))
        for it,(imgs,target) in enumerate(trn_loader):
            imgs,target = imgs.cuda(),target.cuda().float().unsqueeze(1)
            
            pred = model(imgs)
            loss = criterion(pred,target)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            avg_loss.update(loss,imgs.size(0))
            
            if it % 40==0:
                print('It: {}/{} | Loss: {}'.format(it,len(trn_loader),avg_loss.avg))
        
        val_score = eval_model(val_loader,model)
        scheduler.step(avg_loss.avg)
        metrics['train_loss'].append(avg_loss.avg)
        for k,v in val_score.items():
            metrics[k].append(v)
            
        
        if val_score['auc'] > best_auc:
            best_auc = val_score['auc']
            best_model = model
            es = 0
        else:
            es += 1
            if es >= 3:
                break
            
        print('It: {}/{} | Loss: {}'.format(it,len(trn_loader),avg_loss.avg))
        
        print('| Val loss: {} | Val auc: {} | Best auc: {}'.format(val_score['loss'],val_score['auc'],best_auc))
        
    return best_model,best_auc,metrics
            
def train_kfolds(create_version):
    fold = 0
    cv_aucs = []
    for trn_ind,val_ind in kf.split(train_df.image_name,train_df.target):
        fold += 1
        model,optimizer,criterion = create_version()
        print('---- Fold {}'.format(fold))
        trn_df = train_df.iloc[trn_ind]
        val_df = train_df.iloc[val_ind]
        trn_data = Data(trn_df,is_train=True,transform=train_transform)
        val_data = Data(val_df,is_train=True,transform=test_transform)
        trn_loader = DataLoader(trn_data,batch_size=config.train_batch_size,shuffle=True,num_workers=6)
        val_loader = DataLoader(val_data,batch_size=config.train_batch_size,shuffle=False,num_workers=6)
        model,auc,metrics = train_model(trn_loader,val_loader,model,optimizer,criterion,fold)
        torch.save(model.state_dict(),'weight_{}.pt'.format(fold))
        cv_aucs.append(auc)
    return cv_aucs


# # 7. Training

# In[ ]:




def create_baseline():
    model = MyModel().cuda()
    model.train()
    params = list(model.named_parameters())
    def is_fc(n):
        return 'fc' in n
    optimizer_grouped_parameters = [
        {"params": [p for n, p in params if not is_fc(n)], "lr": 1e-4},
        {"params": [p for n, p in params if is_fc(n)], "lr": 1e-4 * 200},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters,lr=config.lr,weight_decay=0)
    #optimizer = optim.AdamW(model.parameters(),lr=config.lr,weight_decay=0)
    criterion = nn.BCELoss()
    return model,optimizer,criterion

cvs = train_kfolds(create_baseline)
print(cvs,np.mean(cvs))


# In[ ]:



def model_pred(model,test_loader):
    model.eval()
    prediction = []
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.cuda()
            pred = list(model(imgs).squeeze(-1).cpu().detach().numpy())
            prediction.extend(pred)
    return prediction


# In[ ]:


preds = []
for i in range(5):
    model = MyModel().cuda()
    model.load_state_dict(torch.load('weight_{}.pt'.format(i+1)))
    pred = model_pred(model,test_loader)
    preds.append(pred)
    
preds = np.array(preds)


# In[ ]:


test_preds = np.mean(preds,axis=0)


# # 8. Submission

# In[ ]:


sample_df = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
sample_df.target = test_preds
sample_df.to_csv('submission.csv',index=False)

