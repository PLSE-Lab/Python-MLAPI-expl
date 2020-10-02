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


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())


# In[ ]:


get_ipython().system('pip install fastai --upgrade')
import fastai
from fastai.tabular import * 


# In[ ]:


train_df = pd.read_csv('../input/train/train.csv')
test_df = pd.read_csv('../input/test/test.csv')
train_df.head()


# In[ ]:


train_df['Description_Length'] = train_df.Description.apply(lambda x: len(str(x)))
test_df['Description_Length'] = test_df.Description.apply(lambda x: len(str(x)))


# In[ ]:


train_df = train_df.drop(labels=['PetID','Description','RescuerID','Name'],axis=1)
test_df = test_df.drop(labels=['PetID','Description','RescuerID','Name'],axis=1)


# In[ ]:


valid_idx = range(len(train_df)-2000, len(train_df))

dep_var = 'AdoptionSpeed'
cat_names = ['Type','Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health','State']

data = TabularDataBunch.from_df('./', train_df, dep_var, valid_idx=valid_idx,
                                procs=[FillMissing, Categorify, Normalize],
                                cat_names=cat_names, test_df=test_df)


# In[ ]:


#learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04,metrics=accuracy)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, 1e-3, wd=0.2)


# In[ ]:


learn.recorder.plot_losses(last=-1)


# In[ ]:


learn.fit_one_cycle(1, 3e-4)


# In[ ]:


from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer

def metric(y1,y2):
    return cohen_kappa_score(y1,y2, weights='quadratic')


# In[ ]:


train_preds = np.argmax(learn.get_preds(data.train_ds)[0],axis=1)
train_preds


# In[ ]:


# training set qk
metric(train_df['AdoptionSpeed'][0:12993],train_preds)


# In[ ]:


# training set accuracy
sum(np.array(train_df['AdoptionSpeed'][0:12993])==np.array(train_preds))/12993


# In[ ]:


val_preds = np.argmax(learn.get_preds()[0],axis=1)
val_preds


# In[ ]:


# validation set qk
metric(train_df['AdoptionSpeed'][12993:],val_preds)


# In[ ]:


# validation set accuracy
sum(np.array(train_df['AdoptionSpeed'][12993:])==np.array(val_preds))/2000


# In[ ]:


test_preds = np.argmax(learn.get_preds(DatasetType.Test)[0],axis=1)


# In[ ]:


# Store predictions for Kaggle Submission
submission_df = pd.DataFrame(data={'PetID' : pd.read_csv('../input/test/test.csv')['PetID'], 
                                   'AdoptionSpeed' : test_preds})
submission_df.to_csv('submission.csv', index=False)


# In[ ]:




