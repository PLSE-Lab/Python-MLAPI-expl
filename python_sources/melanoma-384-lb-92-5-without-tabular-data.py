#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Hi, This is my first step into Image classification. Hope this notebook helps for any beginner like me. Please use the dataset from here. 
# https://www.kaggle.com/cdeotte/jpeg-melanoma-384x384?select=train  
# 
# I have managed to reach LB : 0.925. I have few more ideas that will push LB to atleast 0.935 / 0.94 without tabular data.
# 
# ToDos :
# 1. Use 512x512 data for ensembling
# 2. Use EfficientNet 
# 3. Hair removal from images : There is already an existing implementation for the same
# 4. As many have suggested that external data reduces the LB score, however that can be used as an ensembling technique to improve score. Or better way analyse the malignant images and take the ones with which you feel confident (costs time). 
# 4. Think more ....
# 

# In[ ]:


from tqdm.autonotebook import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import sys
import os
from torch.optim.lr_scheduler import  StepLR
from sklearn.model_selection import train_test_split
import torchvision.models as model;from sklearn.metrics import roc_auc_score,accuracy_score
import matplotlib.pyplot as plt
import pretrainedmodels
import efficientnet_pytorch
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


DIR_INPUT = '../input/siim-isic-melanoma-classification/jpeg'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
FILE_CSV = '../input/siim-isic-melanoma-classification/train.csv'
TEST_CSV = '../input/siim-isic-melanoma-classification/test.csv'
SUBMISSION_CSV = '../input/siim-isic-melanoma-classification/sample_submission.csv'


# In[ ]:


def color_constancy(img, power=6, gamma=None):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255*pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return img.astype(img_dtype)


# In[ ]:


def split_data_v1(path,fold,random_state=42):
    df = pd.read_csv(path) 
    df = df.sort_values('target') 
    df = df.reset_index(drop=True)
    batch_ls = np.array([[0,2000],[2000,4000],[4000,6000],[6000,8000],[8000,10000],[10000,12000],[12000,14000],
                        [14000,16000],[16000,18000],[18000,20000],[20000,22000],[22000,24000],[24000,26000],
                        [26000,28000],[28000,30000],[30000,32540]])
    v1 = 583
    df_ls = []
    for i in range(batch_ls.shape[0]):
        temp = pd.concat([df[batch_ls[i][0]:batch_ls[i][1]].reset_index(drop=True),df[-v1:].reset_index(drop=True)], ignore_index=True)
        df_ls.append(temp)
    X = df_ls[fold].pop('image_name')
    Y = df_ls[fold].pop('target')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test


# In[ ]:


class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    '../input/se-resnext-weight/se_resnext50_32x4d-a260b3a4.pth'
            )
        #self.l0 = nn.Linear(2048, 1)
        self.l0 = nn.Sequential(nn.Linear(2048, 32),nn.Dropout(0.05),nn.ReLU(),nn.Linear(32,1))
    def forward(self, image):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        out = self.l0(x)
        return out


# In[ ]:


class MelanomaDataSet(torch.utils.data.Dataset):
    def __init__(self,image_path,targets,mode):
        self.image_path = image_path
        self.targets = targets
        self.mode = mode
        
        self.aug = A.Compose({
        A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=0.5),
        
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-20, 20)),
        A.VerticalFlip(p=0.5),
        A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),              
        })
        self.aug1 = A.Compose({
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-20, 20)),
        A.VerticalFlip(p=0.5),    
        A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),             
        })
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, item):
        image = self.image_path[item]
        targets = self.targets[item]
        
        if self.mode == 'test':
            image = DIR_TEST+'/'+image+'.jpg'
        else:
            image = DIR_TRAIN+'/'+image+'.jpg'
        
        img = plt.imread(image)
        img = Image.fromarray(img).convert('RGB')
        img = color_constancy(img)
        if(self.mode == 'train' or self.mode=='validation'):
            img = self.aug(image=np.array(img))['image']
            img = np.transpose(img, (2,0,1))
        else:
            img = self.aug1(image=np.array(img))['image']
            img = np.transpose(img, (2,0,1))
        return torch.tensor(img,dtype=torch.float32), torch.tensor(targets,dtype=torch.float32)


# In[ ]:


def run_fold(fold,LR,EPOCHS=10,bs=256):
    train_images, test_images, train_targets, test_targets = split_data_v1(FILE_CSV,fold)
    class_sample_count = np.array([len(np.where(train_targets==t)[0]) for t in np.unique(train_targets)])
    print(class_sample_count)
    weight = 1. / class_sample_count
    print(weight)
    samples_weight = np.array([weight[t] for t in train_targets])

    samples_weight = torch.from_numpy(samples_weight)

    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    
    train_dataset = MelanomaDataSet(train_images.values, train_targets.values, 'train')
    valid_dataset = MelanomaDataSet(test_images.values, test_targets.values, 'validation')
    
    train_data_loader = DataLoader(
    train_dataset,
    sampler=sampler,
    batch_size=bs,
    num_workers=0
    )

    valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=bs,
    shuffle=False,
    num_workers=0
    )
    
    device = torch.device('cuda')
    model = SEResnext50_32x4d()
    fold_val = fold
    if(os.path.isfile(f'melanoma_best___{fold_val}_384.pth')):
        model.load_state_dict(torch.load(f'melanoma_best___{fold_val}_384.pth'))
        print('Loaded ',f'melanoma_best___{fold_val}_384.pth')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_score = 0
    for epoch in range(EPOCHS):
        print('Epoch:', epoch)
        loss_arr = []
        model.train()
        for images, labels in tqdm(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())
            del images, labels
        print('-'*50)
        print("epoch = {},   loss = {}".format(epoch, sum(loss_arr)/len(loss_arr)))
        print('-'*50)
        model.eval()
        final_predictions = []
        for val_images, val_labels in tqdm(valid_data_loader):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            with torch.no_grad():
                val_output = model(val_images)
                val_output = torch.sigmoid(val_output)
                pred = val_output.cpu()
                final_predictions.append(pred)
                del val_images, val_labels
            
        predictions = np.vstack(final_predictions).ravel()
        k = roc_auc_score(test_targets, predictions)
        print('CM : ',confusion_matrix(test_targets, np.round(predictions)))
        
        print('-'*50)
        print('AUC Score = {}'.format(k))
        print('-'*50)
        if(k > best_score):
            best_score = k
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold,epoch+1))
            torch.save(model.state_dict(), f'melanoma_best___{fold}_384.pth')
    return model
def eval_test(test_path,test_csv,submission_csv,fname,bs=128,TTA_COUNT=2,fold=0):
    model = SEResnext50_32x4d()
    device = torch.device('cuda')
    if(os.path.isfile(f'melanoma_best___{fold}_384.pth')):
        model.load_state_dict(torch.load(f'melanoma_best___{fold}_384.pth'))
        model = model.to(device)
        print('Loaded ',f'melanoma_best___{fold}_384.pth')
    else:
        print('Error!! Model not found')
    
    ls_sample = []
    for i in range(TTA_COUNT):
        
        df = pd.read_csv(test_csv)
        targets = np.zeros(len(df))
        print(df.image_name.values)
        test_dataset = MelanomaDataSet(df.image_name.values, targets, 'test')
        test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=0
        )
        model.eval()
        final_predictions = []
        for test_data in test_loader:
            test_images,test_labels = test_data
            test_images = test_images.to(device)
            with torch.no_grad():
                test_output = model(test_images)
                test_output = torch.sigmoid(test_output)
                pred = test_output.cpu()
                final_predictions.append(pred)
                del test_images,test_data
        predictions = np.vstack(final_predictions).ravel()
        sample = pd.read_csv(submission_csv)
        sample.loc[:, "target"] = predictions
        ls_sample.append(sample)
        #sample.to_csv(fname, index=False)
    return ls_sample


# In[ ]:


splits=16
lr = 0.000002
bs = 2
ep = 3
ls_model = []
ls_sample = []
for i in range(splits):
    print(f'Modeling split : {i}')
    ls_model.append(run_fold(i,lr,ep,bs))

for i in range(splits):
    print(f'Evaluating split : {i}')
    ls_sample.append(eval_test(DIR_TEST,TEST_CSV,SUBMISSION_CSV,f'sample_{i}.csv',bs=bs,fold=i))


# Average the predictions from different fold and submit the result.
# Hope it helps!!

# In[ ]:


pred = []
for i in range(splits):
    pred.append((ls_sample[i][0].target.values + ls_sample[i][1].target.values) / 2.0) 
#####
#Average out all pred and submit the csv

