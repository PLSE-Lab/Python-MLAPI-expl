#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('cp ../input/*.csv ./')

import zipfile

Dataset = "training"
Dataset1="test"
# Will unzip the files so that you can see them..
with zipfile.ZipFile("../input/"+Dataset+".zip","r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/"+Dataset1+".zip","r") as z:
    z.extractall(".")


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from fastai.vision import *
from fastai import *

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import shutil
import torch

rn_seed=1
np.random.seed(rn_seed)
# Any results you write to the current directory are saved as output.


# # Load Data

# Control Variables

# In[ ]:


train_path = Path('/tmp/train')
test_path = Path('/tmp/test')


# ### Read CSV data

# In[ ]:


root = Path('../working')


# In[ ]:


id_lookup = pd.read_csv(root/'IdLookupTable.csv')
# try fillna median
# train_csv = pd.read_csv(root/'training/training.csv').dropna(axis=0)
train_csv = pd.read_csv(root/'training.csv')
test_csv = pd.read_csv(root/'test.csv')


# In[ ]:


id_lookup.head()


# In[ ]:


for c in train_csv.columns:
    if(train_csv[c].dtype!='object'):
        train_csv[c]=train_csv[c].fillna(train_csv[c].median())


# In[ ]:


train_csv.head()


# In[ ]:


train_csv.describe()


# In[ ]:


test_csv.head()


# ### Image array to images 

# Save train images

# In[ ]:


def save_str_img(strimg,w,h,flpath):
    px=255-np.array(strimg.split(),dtype=int)
    if(len(px)==w*h and len(px)%w==0 and len(px)%h==0):
        cpx = list(px.reshape(w,h))
        img = Image(Tensor([cpx,cpx,cpx]))
        img.save(flpath)
        return img
    else:
        raise Exception("Invalid height and width")


# In[ ]:


# make image folders
# shutil.rmtree(train_path)
train_path.mkdir(exist_ok=True)
test_path.mkdir(exist_ok=True)


# In[ ]:


# for each row
for index, train_row in train_csv.iterrows():
    save_str_img(train_row.Image,96,96,train_path/(str(index)+'.jpg'))


# Save test images

# In[ ]:


# for each row
for index, test_row in test_csv.iterrows():
    save_str_img(test_row.Image,96,96,test_path/(str(index)+'.jpg'))


# ### Make Data bunch

# In[ ]:


def get_locs(flname):
    index = int(flname.name[:-4])
    plist=[]
    coords=list(train_csv.loc[index])
    for i in range(len(coords)//2):
        plist.append([coords[i*2+1],coords[i*2]])
    return tensor(plist)
#     return tensor([coords[1],coords[0]])


# In[ ]:


# make points image data bunch
# TODO remove transforms
data = (PointsItemList.from_folder(train_path)
        .split_by_rand_pct(0.05,seed=rn_seed)
        .label_from_func(get_locs)
        .transform([],size=(96,96))
        .databunch(num_workers=0)
       )


# In[ ]:


data.show_batch(3,figsize=(6,6))


# # Train Model

# In[ ]:


# calculates distance between true and predictions
def mloss(y_true, y_pred):
    y_true=y_true.view(-1,15,2)
    
    y_true[:,:,0]=y_true[:,:,0].clone()-y_pred[:,:,0]
    y_true[:,:,1]=y_true[:,:,1].clone()-y_pred[:,:,1]
    
    y_true[:,:,0]=y_true[:,:,0].clone()**2
    y_true[:,:,1]=y_true[:,:,1].clone()**2
    
    return y_true.sum(dim=2).sum(dim=1).sum()


# In[ ]:


learn = cnn_learner(data,models.resnet152,loss_func=mloss)


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.show_results(rows=3,figsize=(6,6))


# In[ ]:


learn.save('s1')


# # Fine tune model

# In[ ]:


learn.load('s1');


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(30,5e-5)


# In[ ]:


learn.show_results(rows=3,figsize=(6,6))


# # Make predictions

# Predict and display in one image

# In[ ]:


def flp(npa):
    for i in range(npa.shape[0]):
        if(i%2==1):
            tmp=npa[i]
            npa[i]=npa[i-1]
            npa[i-1]=tmp
    return npa


# In[ ]:


def get_coords(imgpnts):
    return ((imgpnts+1)*48).numpy()


# In[ ]:


test_img = open_image(test_path/'1600.jpg')
pred = learn.predict(test_img)
test_img.show(y=pred[0])


# In[ ]:


test_img = open_image(test_path/'1600.jpg')
pred = learn.predict(test_img)
test_img.show(y=ImagePoints(FlowField(test_img.size,torch.from_numpy(get_coords(pred[1])))))


# Make predictions and save dataframe

# In[ ]:


a=list(train_csv.columns.values)
a.remove('Image')
a.append('ImageId')


# In[ ]:


test_preds = pd.DataFrame(columns=a)


# In[ ]:


from ipywidgets import IntProgress
from IPython.display import display

f = IntProgress(min=0, max=test_csv.count()[0]) # instantiate the bar
display(f)
for test_index in range(test_csv.count()[0]):
    timg = open_image(test_path/(str(test_index)+'.jpg'))
    pred = learn.predict(timg)
    a=np.abs(flp(get_coords(pred[1]).reshape(1,-1)[0]))
    a=np.append(a,test_csv.loc[test_index].ImageId)
    test_preds.loc[test_index]=a
    f.value+=1


# In[ ]:


test_preds.describe()


# In[ ]:


test_preds.ImageId=test_preds.ImageId.astype('int')
test_preds.head()


# In[ ]:


sub = pd.DataFrame(columns=['RowId','Location'])
for index,row in id_lookup.iterrows():
    fname = row.FeatureName
    trow=test_preds.loc[test_preds['ImageId']==row.ImageId]
    sub.loc[index]=[row.RowId,trow.iloc[0][fname]]


# In[ ]:


sub.RowId=sub.RowId.astype('int')
sub.head()


# In[ ]:


sub.to_csv("sub.csv",index=False)

