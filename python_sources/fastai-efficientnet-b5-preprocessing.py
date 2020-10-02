#!/usr/bin/env python
# coding: utf-8

# This kernel is based on DrHabib's starter kernel: - Great starter and lots of thanks!
# https://www.kaggle.com/drhabib/starter-kernel-for-0-79
# Also codes from Mendonca's EfficientNetB4 -
# https://www.kaggle.com/hmendonca/efficientnetb4-fastai-blindness-detection
# Preprocessing methods (circle_crop/ben's cropping) from -
# https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping

# In[ ]:


'''
# directly from pretrained:
k = 0
preds = ['','','','','']
for image_size in [224,232,240,248,256]:
    test = ImageList.from_df(test_df,
                             '../input/aptos2019-blindness-detection',
                             folder='test_images',
                             suffix='.png')
    data = (ImageList.from_df(df=train_df,path='./',cols='path')
            .split_by_rand_pct(0.2) 
            .label_from_df(cols='diagnosis',label_cls=FloatList)
            .add_test(test)
            .transform(transforms,size=image_size,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
            .databunch(bs=batch_size,num_workers=4) 
            .normalize(imagenet_stats)  
           )
    learn = Learner(data, 
                    model_b5, 
                    metrics = [quadratic_kappa], 
                    model_dir="models").to_fp16()
    learn.load('abcdef');
    opt = OptimizedRounder()
    preds[k],y = learn.get_preds(DatasetType.Test)
    k = k + 1
preds_1 = preds[0] * 0.2 + preds[1] * 0.20 + preds[2] * 0.2 + preds[3] * 0.2 + preds[4] * 0.2
'''


# In[ ]:


'''
# directly from pretrained:
k = 0
preds = ['','','','','']
for image_size in [224,232,240,248,256]:
    test = ImageList.from_df(test_df,
                             '../input/aptos2019-blindness-detection',
                             folder='test_images',
                             suffix='.png')
    data = (ImageList.from_df(df=train_df,path='./',cols='path')
            .split_by_rand_pct(0.2) 
            .label_from_df(cols='diagnosis',label_cls=FloatList)
            .add_test(test)
            .transform(transforms,size=image_size,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
            .databunch(bs=batch_size,num_workers=4) 
            .normalize(imagenet_stats)  
           )
    learn = Learner(data, 
                    model_b5, 
                    metrics = [quadratic_kappa], 
                    model_dir="models").to_fp16()
    learn.load('best_kappa');
    opt = OptimizedRounder()
    preds[k],y = learn.get_preds(DatasetType.Test)
    k = k + 1
preds_2 = preds[0] * 0.2 + preds[1] * 0.20 + preds[2] * 0.2 + preds[3] * 0.2 + preds[4] * 0.2
'''


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
# data visualisation and manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#
from joblib import load, dump
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import torch
from torchvision import models as md
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
import re
import math
import json
import os
import sys
import cv2
import collections
from functools import partial
from collections import Counter

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# loading fastai
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.basic_train import *
from fastai.vision.learner import *

# set directory
dir_19_name = os.path.join('..', 'input/aptos2019-blindness-detection/')
dir_15_name = os.path.join('..', 'input/diabetic-retinopathy-resized/')
# loading EfficientNet
# Repository source: https://github.com/qubvel/efficientnet
sys.path.append(os.path.abspath('../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master'))
from efficientnet_pytorch import EfficientNet


# In[ ]:


#making model
md_ef = EfficientNet.from_name('efficientnet-b5',override_params={'num_classes':1})
#copying weighst to the local directory 
get_ipython().system('mkdir models')
get_ipython().system("cp '../input/kaggle-public/abcdef.pth' 'models'")


# In[ ]:


def get_df(dir_15_name,dir_19_name):
    valid_dir = os.path.join(dir_19_name,'train_images/')
    valid_df = pd.read_csv(os.path.join(dir_19_name,'train.csv'))
    valid_df['path'] = valid_df['id_code'].map(lambda x: os.path.join(valid_dir,'{}.png'.format(x)))
    #valid_df = valid_df.drop(columns=['id_code'])
    valid_df['is_valid'] = [True] * len(valid_df)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
    test_df = pd.read_csv(os.path.join(dir_19_name,'sample_submission.csv'))
    train_dir = os.path.join(dir_15_name,'resized_train_cropped/resized_train_cropped/')
    train_df = pd.read_csv(os.path.join(dir_15_name,'trainLabels_cropped.csv'))
    train_df['path'] = train_df['image'].map(lambda x: os.path.join(train_dir,'{}.jpeg'.format(x)))
    train_df['diagnosis'] = train_df['level']
    train_df['id_code'] = train_df['image']
    train_df['is_valid'] = [False] * len(train_df)
    train_df1 = train_df[train_df['diagnosis'] == 0]
    train_df2 = train_df[train_df['diagnosis'] != 0]
    train_df1 = train_df1.sample(frac=1).reset_index(drop=True) #shuffle dataframe
    #train_df1 = train_df1[:5000]
    train_df = pd.concat([train_df1,train_df2],axis=0,ignore_index=True)
    train_df = train_df.drop(columns = ['Unnamed: 0.1','Unnamed: 0','level','image'])
    train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
    return train_df, valid_df, test_df
train_df, valid_df, test_df = get_df(dir_15_name,dir_19_name)


# In[ ]:


#%% train_test_split using Stratified K-folds:
SEED = 42
def skfold_split(x, y, n_folds=5, random_seed = SEED):  
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    split_index = [(train,valid) for train, valid in skf.split(x, y)]
    return split_index
df = pd.concat([train_df,valid_df],axis = 0,ignore_index = True)
split_idx = skfold_split(df['id_code'],df['diagnosis'])
train_df0 = df[df.index.isin(split_idx[0][0])]
valid_df0 = df[df.index.isin(split_idx[0][1])]
train_df1 = df[df.index.isin(split_idx[1][0])]
valid_df1 = df[df.index.isin(split_idx[1][1])]
train_df2 = df[df.index.isin(split_idx[2][0])]
valid_df2 = df[df.index.isin(split_idx[2][1])]
train_df3 = df[df.index.isin(split_idx[3][0])]
valid_df3 = df[df.index.isin(split_idx[3][1])]
train_df4 = df[df.index.isin(split_idx[4][0])]
valid_df4 = df[df.index.isin(split_idx[4][1])]
print('train_df.shape:', train_df3.shape)
print('valid_df.shape:', valid_df3.shape)


# train_df1 = train_df[:4000]
# train_df2 = train_df[4000:8000]
# train_df3 = train_df[8000:12000]
# train_df4 = train_df[12000:]
# valid_df1 = valid_df[:1000]
# valid_df2 = valid_df[1000:2000]
# valid_df3 = valid_df[2000:3000]
# valid_df4 = valid_df[3000:]
# res1 = pd.concat([train_df1,valid_df1],axis=0,ignore_index=True)
# res2 = pd.concat([train_df2,valid_df2],axis=0,ignore_index=True)
# res3 = pd.concat([train_df3,valid_df3],axis=0,ignore_index=True)
# res4 = pd.concat([train_df4,valid_df4],axis=0,ignore_index=True)

# In[ ]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    
    #img = cv2.imread(img)
    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    
    return img 

def circle_crop_v2(img):
    """
    Create circular crop around image centre
    """
    #img = cv2.imread(img)
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

def qk(y_pred, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_pred), y, weights='quadratic'), device='cuda:0')
#https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']
def _load_format(path, convert_mode, after_open)->Image:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = circle_crop(image)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.addWeighted(image,4,cv2.GaussianBlur(image,(0,0),10),-4,128)
    return Image(pil2tensor(image, np.float32).div_(255)) #return fastai Image format


# In[ ]:


batch_size = 64
image_size = 256
transforms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360, max_zoom=1.35, max_lighting=0.1,p_affine = 0.5)


# In[ ]:


preds = ['','','','','','','','','','']
i = 0
#for sz in [264,232,240,248,256,264,232,240,248,256]:
test = (ImageList.from_df(test_df,
                          '../input/aptos2019-blindness-detection',
                          folder='test_images',
                          suffix='.png'))
data = (ImageList.from_df(df=valid_df,path='./',cols='path') 
        .split_by_rand_pct(0.2) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
        .add_test(test)
        .transform(transforms,size=image_size,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=batch_size,num_workers=4) 
        .normalize(imagenet_stats)  
        )
learn = Learner(data, 
                md_ef, 
                metrics = [qk], 
                model_dir="models").to_fp16()
learn.load('abcdef');
preds_A,y = learn.get_preds(DatasetType.Test)


# In[ ]:


batch_size = 64
image_size = 224
get_ipython().system("cp '../input/aptosdataset/19model2.pth' 'models'")
vision.data.open_image = _load_format


# In[ ]:


data = (ImageList.from_df(df=df,path='./',cols='path') 
        .split_by_idxs(split_idx[0][0],split_idx[0][1]) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
        .transform(transforms,size=image_size,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=batch_size,num_workers=6) 
        .normalize(imagenet_stats)  
        )
learn = Learner(data, 
                md_ef, 
                metrics = [qk], 
                callback_fns=[BnFreeze,
                              partial(SaveModelCallback, monitor='valid_loss', name='model0')],
                model_dir="models").to_fp16()

learn.data.add_test(ImageList.from_df(test_df,
                                      '../input/aptos2019-blindness-detection',
                                      folder='test_images',
                                      suffix='.png'))
learn.load('19model2')
learn.fit_one_cycle(20,1e-3,wd = 1e-2)
learn.load('model0')
preds0,y = learn.get_preds(DatasetType.Test)


# In[ ]:


data = (ImageList.from_df(df=df,path='./',cols='path') 
        .split_by_idxs(split_idx[1][0],split_idx[1][1]) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
        .transform(transforms,size=image_size,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=batch_size,num_workers=6) 
        .normalize(imagenet_stats)  
        )
learn = Learner(data, 
                md_ef, 
                metrics = [qk], 
                callback_fns=[BnFreeze,
                              partial(SaveModelCallback, monitor='valid_loss', name='model1')],
                model_dir="models").to_fp16()

learn.data.add_test(ImageList.from_df(test_df,
                                      '../input/aptos2019-blindness-detection',
                                      folder='test_images',
                                      suffix='.png'))
learn.load('19model2')
learn.fit_one_cycle(20,1e-3,wd = 1e-2)
learn.load('model1')
preds1,y = learn.get_preds(DatasetType.Test)


# In[ ]:


data = (ImageList.from_df(df=df,path='./',cols='path') 
        .split_by_idxs(split_idx[2][0],split_idx[2][1]) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
        .transform(transforms,size=image_size,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=batch_size,num_workers=6) 
        .normalize(imagenet_stats)  
        )
learn = Learner(data, 
                md_ef, 
                metrics = [qk], 
                callback_fns=[BnFreeze,
                              partial(SaveModelCallback, monitor='valid_loss', name='model2')],
                model_dir="models").to_fp16()

learn.data.add_test(ImageList.from_df(test_df,
                                      '../input/aptos2019-blindness-detection',
                                      folder='test_images',
                                      suffix='.png'))
learn.load('19model2')
learn.fit_one_cycle(20,1e-3,wd = 1e-2)
learn.load('model2')
preds2,y = learn.get_preds(DatasetType.Test)


# In[ ]:


data = (ImageList.from_df(df=df,path='./',cols='path') 
        .split_by_idxs(split_idx[3][0],split_idx[3][1]) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
        .transform(transforms,size=image_size,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=batch_size,num_workers=6) 
        .normalize(imagenet_stats)  
        )
learn = Learner(data, 
                md_ef, 
                metrics = [qk], 
                callback_fns=[BnFreeze,
                              partial(SaveModelCallback, monitor='valid_loss', name='model3')],
                model_dir="models").to_fp16()

learn.data.add_test(ImageList.from_df(test_df,
                                      '../input/aptos2019-blindness-detection',
                                      folder='test_images',
                                      suffix='.png'))
learn.load('19model2')
learn.fit_one_cycle(20,1e-3,wd = 1e-2)
learn.load('model3')
preds3,y = learn.get_preds(DatasetType.Test)


# In[ ]:


data = (ImageList.from_df(df=df,path='./',cols='path') 
        .split_by_idxs(split_idx[4][0],split_idx[4][1]) 
        .label_from_df(cols='diagnosis',label_cls=FloatList) 
        .transform(transforms,size=image_size,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
        .databunch(bs=batch_size,num_workers=4) 
        .normalize(imagenet_stats)  
        )
learn = Learner(data, 
                md_ef, 
                metrics = [qk], 
                callback_fns=[BnFreeze,
                              partial(SaveModelCallback, monitor='valid_loss', name='model4')],
                model_dir="models").to_fp16()

learn.data.add_test(ImageList.from_df(test_df,
                                      '../input/aptos2019-blindness-detection',
                                      folder='test_images',
                                      suffix='.png'))
learn.load('19model2')
learn.fit_one_cycle(20,1e-3,wd = 1e-2)
learn.load('model4')
preds4,y = learn.get_preds(DatasetType.Test)


# In[ ]:


'''
preds = ['','','','','','','','','','']
i = 0
opt = OptimizedRounder()
for sz in [264,232,240,248,256,264,232,240,248,256]:
    test = (ImageList.from_df(test_df,
                              '../input/aptos2019-blindness-detection',
                              folder='test_images',
                              suffix='.png'))
    data = (ImageList.from_df(df=valid_df,path='./',cols='path') 
            .split_by_rand_pct(0.2) 
            .label_from_df(cols='diagnosis',label_cls=FloatList) 
            .add_test(test)
            .transform(transforms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 
            .databunch(bs=batch_size,num_workers=4) 
            .normalize(imagenet_stats)  
           )
    learn = Learner(data, 
                    md_ef, 
                    metrics = [quadratic_kappa], 
                    model_dir="models").to_fp16()
    learn.load('19_kappa');
    preds[i],y = learn.get_preds(DatasetType.Test)
    i = i + 1
preds_B = (preds[0] + preds[1] + preds[2] + preds[3] + preds[4] + preds[5] + preds[6] + preds[7] + preds[8] + preds[9])/10
'''
pred_final = preds_A * 0.3 + preds0 * 0.14 + preds1 * 0.14 + preds2 * 0.14 + preds3 * 0.14 + preds5 * 0.14


# In[ ]:


#learn.load('final_kappa')
opt = OptimizedRounder()
tst_pred = opt.predict(pred_final,coef=[0.5, 1.5, 2.5, 3.5])
test_df.diagnosis = tst_pred.astype(int)
test_df.to_csv('submission.csv',index=False)
print ('done')

