#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import torch
import torchvision
torch.cuda.empty_cache()
from torchvision import models
from torch import nn
path = '../input'
device = torch.device("cuda:0")

model1 = torch.load("../input/a-simple-fastai-ensemble-training-kernel-0-60/model1.pth")
model2 = torch.load("../input/a-simple-fastai-ensemble-training-kernel-0-60/model2.pth")
model3 = torch.load("../input/a-simple-fastai-ensemble-training-kernel-0-60/model3.pth")
model4 = torch.load("../input/a-simple-fastai-ensemble-training-kernel-0-60/model4.pth")


# In[ ]:


from fastai.vision import *
dff = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
src = (ImageList.from_df(dff, path='../input/aptos2019-blindness-detection', folder='test_images', suffix='.png')
               .split_none()
               .label_empty())
iB = ImageDataBunch.create_from_ll(src,size=299,bs=32,
                                  ds_tfms=get_transforms(do_flip=True,
                                      max_warp=0,
                                      max_rotate=0,
                                      max_lighting=0,
                                      p_affine=0.2,
                                      xtra_tfms=[crop_pad()]))
labels1,labels2,labels3,labels4 = [],[],[],[]
predictor1 = Learner(data=iB,model=model1,model_dir='/tmp/models')
preds1 = predictor1.get_preds(ds_type=DatasetType.Fix)
predictor2 = Learner(data=iB,model=model2,model_dir='/tmp/models')
preds2 = predictor2.get_preds(ds_type=DatasetType.Fix)
predictor4 = Learner(data=iB,model=model4,model_dir='/tmp/models')
preds4 = predictor4.get_preds(ds_type=DatasetType.Fix)
labels1,labels2,labels3,labels4 = [],[],[],[]
print("Predicting from model1....")
for pr in preds1[0]:
    p = pr.tolist()
    labels1.append(np.argmax(p))
print("Predicting from model2....")
for pr in preds2[0]:
    p = pr.tolist()
    labels2.append(np.argmax(p))
print("Predicting from model4....")
for pr in preds4[0]:
    p = pr.tolist()
    labels4.append(np.argmax(p))


# In[ ]:


finalPreds=[]
for i in range(len(labels1)):
    pp = (0.3*labels1[i]+0.5*labels2[i]+0.2*labels4[i])/1
    pp = np.floor(pp)
    finalPreds.append(int(pp))


# In[ ]:


dff = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
ids=  list(dff["id_code"])
submit = pd.DataFrame(data={'id_code':ids,'diagnosis':finalPreds})
submit.to_csv('./submission.csv',index=False)

