#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install timm')


# In[ ]:


import numpy as np 
import pandas as pd 
import os
import cv2
import torch.nn.init as init
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
import seaborn as sns
import random
import timm
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('../input/autoaug')
from auto_augment import AutoAugment, Cutout
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


# In[ ]:


seed_everything(2020)
num_classes = 2
bs = 80
lr = 1e-3
IMG_SIZE = 224


# In[ ]:


train_path = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/'
test_path = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-test/512x512-test/'
train_csv = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/folds.csv')
test_csv = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
sample = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')


# In[ ]:


train_csv.head()


# In[ ]:


class MyDataset(Dataset):
    
    def __init__(self, dataframe, transform=None, test=False):
        self.df = dataframe
        self.transform = transform
        self.test = test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.target.values[idx]
        p = self.df.image_id.values[idx]
        
        if self.test == False:
            p_path = train_path + p + '.jpg'
        else:
            p_path = test_path + p + '.jpg'
            
#         image = cv2.imread(p_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.imread(p_path, cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0
#         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#         image = transforms.ToPILImage()(image)
        
        if self.transform:
            sample = {'image': image}
            sample = self.transform(**sample)
            image = sample['image']
        
        return image, label


# In[ ]:


train_transform = A.Compose([
            A.RandomSizedCrop(min_max_height=(200, 200), height=224, width=224, p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=224, width=224, p=1),
            A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),                  
            ], p=1.0)

test_transform = A.Compose([
            A.Resize(height=224, width=224, p=1.0),
            ToTensorV2(p=1.0),
            ], p=1.0)


testset      = MyDataset(sample, transform=test_transform, test=True)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)


# In[ ]:


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[ ]:


def train_model(model, epoch):
    model.train() 
    
    losses = AverageMeter()
    avg_loss = 0.

    optimizer.zero_grad()
    
    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        imgs_train, labels_train = imgs.cuda(), labels.cuda().long()
        output_train = model(imgs_train)

        loss = criterion(output_train, labels_train)
        loss.backward()

        optimizer.step() 
        optimizer.zero_grad() 
        
        avg_loss += loss.item() / len(train_loader)
        
        losses.update(loss.item(), imgs_train.size(0))

        tk.set_postfix(loss=losses.avg)
        
    return avg_loss


def test_model(model):    
    model.eval()
    
    losses = AverageMeter()
    avg_val_loss = 0.
    
    valid_preds, valid_targets = [], []
    
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid, labels_valid = imgs.cuda(), labels.cuda().long()
            output_valid = model(imgs_valid)
            
            loss = criterion(output_valid, labels_valid)
            
            avg_val_loss += loss.item() / len(val_loader)

            losses.update(loss.item(), imgs_valid.size(0))
            
            tk.set_postfix(loss=losses.avg)
            
            valid_preds.append(torch.softmax(output_valid,1)[:,1].detach().cpu().numpy())
            valid_targets.append(labels_valid.detach().cpu().numpy())
            
        valid_preds = np.concatenate(valid_preds)
        valid_targets = np.concatenate(valid_targets)
        auc =  roc_auc_score(valid_targets, valid_preds) 
            
    return avg_val_loss, auc


# In[ ]:


kf = StratifiedKFold(5, shuffle=True, random_state=0)

cv = []


# In[ ]:


for i in range(5):
    fold = i+1
    print('fold:', fold)

    train_df = train_csv[train_csv['fold'] != i]
    val_df = train_csv[train_csv['fold'] == i]
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    trainset = MyDataset(train_df, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)
   
    valset = MyDataset(val_df, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=False, num_workers=4)

    model = timm.create_model('tf_efficientnet_b2_ns', pretrained=True, num_classes=num_classes)
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         patience=1,
#         threshold=1e-4,
#         mode="max"
#     )
    
    best_auc = 0
    n_epochs = 10
    es = 0

    for epoch in range(n_epochs):
        avg_loss = train_model(model, epoch)
        avg_val_loss, auc = test_model(model)

        if auc > best_auc:
            es = 0
            best_auc = auc
            torch.save(model.state_dict(), str(fold) + 'weight.pt')
        else:
            es += 1
            if es > 2:
                break
        print('current_val_auc:', auc, 'best_val_auc:', best_auc)
        
        scheduler.step(auc)

    cv.append(best_auc)


# In[ ]:


print(cv)


# In[ ]:


model1 = timm.create_model('tf_efficientnet_b2_ns', pretrained=True, num_classes=num_classes)
model1.cuda()
model1.load_state_dict(torch.load("./1weight.pt"))

model2 = timm.create_model('tf_efficientnet_b2_ns', pretrained=True, num_classes=num_classes)
model2.cuda()
model2.load_state_dict(torch.load("./2weight.pt"))

model3 = timm.create_model('tf_efficientnet_b2_ns', pretrained=True, num_classes=num_classes)
model3.cuda()
model3.load_state_dict(torch.load("./3weight.pt"))

model4 = timm.create_model('tf_efficientnet_b2_ns', pretrained=True, num_classes=num_classes)
model4.cuda()
model4.load_state_dict(torch.load("./4weight.pt"))

model5 = timm.create_model('tf_efficientnet_b2_ns', pretrained=True, num_classes=num_classes)
model5.cuda()
model5.load_state_dict(torch.load("./5weight.pt"))


# In[ ]:


model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()


# In[ ]:


test_pred = np.zeros((len(sample),))

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, position=0, leave=True)):
        images, _ = data
        images = images.cuda()
        
        pred = (model1(images) + model2(images) + model3(images) + model4(images) + model5(images)) 
        
        pred = torch.softmax(pred,1).cpu().detach().numpy()[:,1]
    
        test_pred[i*bs: (i+1)*bs] = pred


# In[ ]:


print(test_pred[:10])


# In[ ]:


sample.target = test_pred


# In[ ]:


sample.to_csv('submission.csv',index=False)

