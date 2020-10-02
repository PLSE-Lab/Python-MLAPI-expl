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


PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=224


# In[ ]:


fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])
labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])


# In[ ]:


print(fnames[-2],labels[-2])


# In[ ]:


from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


arch=resnet50


# **Model 1**

# data = ImageClassifierData.from_names_and_array(
#     path=PATH, 
#     fnames=fnames, 
#     y=labels, 
#     classes=['dogs', 'cats'], 
#     test_name='test', 
#     tfms= tfms_from_model(arch, sz)   
# )
# learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)

# %%time
# learn.fit(0.01, 2)

# In[ ]:


#lrf= learn.lr_find()


# In[ ]:


#learn.sched.plot_lr()


# In[ ]:


#learn.sched.plot()


# In[ ]:


#learn.save('model1')


# This model got 0.06218 binary loss error on test set on Kaggle with resnet 34
# and 0.05806 with resnet 50 and 0.05876 with resnet101   and 0.06055 with TTA resnet50

# **Model 2**

# In[ ]:


data = ImageClassifierData.from_names_and_array(
    path=PATH, 
    fnames=fnames, 
    y=labels, 
    classes=['dogs', 'cats'], 
    test_name='test', 
    tfms= tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)   #data augmentation
)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)


# In[ ]:


learn.fit(0.01,4)


# In[ ]:


learn.precompute=False
learn.fit(1e-2, 3, cycle_len=2)


# In[ ]:


#learn.sched.plot_lr()


# In[ ]:


#learn.save('model2')


# In[ ]:


#learn.load('model2')


# This model got  0.06917 log loss error on test set on Kaggle using resnet50  and 0.05605 with TTA resent50

# **Model 3**

# In[ ]:


learn.unfreeze()


# In[ ]:


lr=np.array([1e-4,1e-3,1e-2])


# In[ ]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# 0.07477 log losss model 3

# In[ ]:


#learn.sched.plot_lr()


# In[ ]:


#learn.save('model3')


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'learn.TTA')


# In[ ]:


log_predictions,y = learn.TTA(is_test=True)
prob_predictions = np.mean(np.exp(log_predictions),0)
probs = prob_predictions[:,1]


# In[ ]:


valid_preds= np.argmax(prob_predictions, axis=1)


# log_preds = learn.predict(is_test=True)
# preds = np.argmax(log_preds, axis=1) 
# probs = np.exp(log_preds[:,1])
# #probs= probs[:,1]
# #probs = [max(np.exp(i)[0],np.exp(i)[1]) for i in log_preds]

# In[ ]:


#tmpk= log_preds
#tmpk= log_preds[:,:,0]
#tmpk=tmpk.reshape(tmpk.shape[1],tmpk.shape[0])


# In[ ]:


#tmpk= [np.mean(i) for i in tmpk]


# In[ ]:


#tmpk= [ np.exp(i) for i in tmpk]


# In[ ]:


log_predictions.shape


# In[ ]:


ids= fnames = np.array([f'{f}' for f in os.listdir(f'{PATH}test')])


# In[ ]:


ids= [i.replace(".jpg","") for i in ids]
ids[0]


# In[ ]:


ans= pd.DataFrame({"id":ids,"label":probs})
ans= ans.sort_values('id')
ans.head()


# In[ ]:


ans.describe()


# In[ ]:


cm = confusion_matrix(y, valid_preds)
plot_confusion_matrix(cm, data.classes)


# In[ ]:


# model 2 mean 5.006981e-01   std  4.964828e-01    min  1.536870e-09   [0.025943222, 0.9912974683544303]


# In[ ]:


#model 1 mean 5.014179e-01   std 4.955358e-01     min 2.701442e-08     [0.026336912, 0.991495253164557]


# In[ ]:


ans.to_csv('submission.csv', index=False)


# In[ ]:




