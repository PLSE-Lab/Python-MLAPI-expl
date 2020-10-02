#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will build a simple LGBM model where we will use 6 simple image stats features which are inspired from [this](https://www.kaggle.com/iafoss/panda-16x128x128-tiles) great kernel. The purpose of this kernel is just to provide others a pipeline/structure how to properly submit in this competition. As we see a lot of discussions around submission problem that kagglers are facing in this competition.

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")
test = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/test.csv")
train.head()


# In[ ]:


sz = 128
N=16


# In[ ]:


import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

from sklearn.metrics import cohen_kappa_score , confusion_matrix
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# Let's create the feature engineering function which would create those 6 image stats features and the indicator features from data provider

# In[ ]:


def feature_engineering(data = train , dir_name = "train_images"):
    r_mean = []
    g_mean=[]
    b_mean = []
    r_sd = []
    g_sd = []
    b_sd = []
    
    for i in data['image_id'].values:
        img = skimage.io.MultiImage(os.path.join(f"/kaggle/input/prostate-cancer-grade-assessment/{dir_name}"+"/"+str(i)+".tiff"))[2]
    #print(img.shape)
        shape = img.shape
        pad0 = (sz-shape[0]%sz)%sz  #### horizontal padding
        pad1 = (sz-shape[1]%sz)%sz  #### vartical padding
        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)
        img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
        img = img.transpose(0,2,1,3,4)
        img = img.reshape(-1,sz,sz,3)
        if len(img) < N:
            img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
 
        idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
        img = img[idxs]
        img = (img/255.0).reshape(-1,3)
        #print(img.mean(0)[0])
        r_mean.append(img.mean(0)[0])
        g_mean.append(img.mean(0)[1])
        b_mean.append(img.mean(0)[2])
    
        r_sd.append(img.std(0)[0])
        g_sd.append(img.std(0)[1])
        b_sd.append(img.std(0)[2])
        
        del img
        gc.collect()
    
    data['r_mean'] = r_mean
    data['g_mean'] = g_mean
    data['b_mean'] = b_mean

    data['r_sd'] = r_sd
    data['g_sd'] = g_sd
    data['b_sd'] = b_sd
    
    data['data_prov_ind'] = np.where(data['data_provider'] == "radboud" , 1 , 0)
    
    return data


# In[ ]:


train = feature_engineering(data = train , dir_name = "train_images")


# In[ ]:


train.head()


# Let's quickly check the object sizes of the local files that we have generated in the process as we need to be careful about memory optimization throughout this competition.

# In[ ]:


#https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
from __future__ import print_function  
import sys

local_vars = list(locals().items())
for var, obj in local_vars:
    if not var.startswith('_'):
        print(var, sys.getsizeof(obj))


# Our LGBM features are the seven following ones

# In[ ]:


features = ["data_prov_ind" , 'r_mean', 'g_mean', 'b_mean', 'r_sd', 'g_sd', 'b_sd']


# QWK: Our competition metric 

# In[ ]:


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


# In[ ]:


def QWK(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.rint(preds)
    score = quadratic_weighted_kappa(preds, labels)
    return ("QWK", score, True)


# In[ ]:


y = train["isup_grade"]
train = train[features]
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)


# Now Let's quickly build the LGBM model.

# In[ ]:


train_dataset = lgb.Dataset(X_train, y_train)
valid_dataset = lgb.Dataset(X_test, y_test)

params = {
            "objective": 'regression',
            "metric": 'rmse',
            "seed": 0,
            "learning_rate": 0.01,
            "boosting": "gbdt",
            }
        
model = lgb.train(
            params=params,
            num_boost_round=10000,
            early_stopping_rounds=100,
            train_set=train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            verbose_eval=100,
            feval=QWK)


# In[ ]:


preds = model.predict(X_test, num_iteration=model.best_iteration)
preds = np.rint(preds)
preds = np.clip(preds, 0 , 5)


# In[ ]:


model.best_iteration


# In[ ]:


print("our validation score is" , quadratic_weighted_kappa(preds, y_test))


# Let's also look at the confusion matrix for the validation set

# In[ ]:


print(confusion_matrix(preds,y_test))


# This does not look good, but fair enough as we are mostly interested in what is coming next, that is submitting using this model

# In[ ]:


from __future__ import print_function  
import sys

local_vars = list(locals().items())
for var, obj in local_vars:
    if not var.startswith('_'):
        print(var, sys.getsizeof(obj))


# In[ ]:


del X_train ,X_test ,y_train ,y_test 
gc.collect()


# In[ ]:


del train_dataset, valid_dataset , train , filenames
gc.collect()


# Now, let's create our inference function 

# In[ ]:


def inference (da = test , dir_path = "test_images"):
    if os.path.exists(f'../input/prostate-cancer-grade-assessment/{dir_path}'):
        print('run inference')
        
        preds = model.predict(da[features], num_iteration=500)
        preds = np.rint(preds)
        preds = np.clip(preds, 0 ,5)
        da['isup_grade'] = preds.astype(int)
        cols = ["image_id" , "isup_grade"]
        da = da[cols]
        
    return da


# In the following section, we will check if the inference function is working on the training set or not, keep this part just to mnake sure that the pipeline is ready enough to handle the unseen test data as well, since, we have already deleted the train data, let's start afresh by reading the train data again.

# In[ ]:


train = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")


# In[ ]:


sub = inference(da = feature_engineering(data = train.head(10) , dir_name = "train_images") , dir_path = "train_images")
sub['isup_grade'] = sub['isup_grade'].astype(int)
sub.to_csv('submission.csv', index=False)
sub.head()


# Now, in this final section, we will try to submit in this competition.
# When, we are committing our kernel, note that, the code would not be able to access the test data, so your model submission would not be prepared. Probably, because hosts did not want us to receive any information from the test data. However, when we submit, then your kernel would access the test data. Now, that is why we need to prepare this *if-else statement*, so that when we are just committing, sample submission file would be generated in the output and when we are submitting, the intended file would be submitted. Finally, keep the internet off in your inference kernels.

# In[ ]:


if os.path.exists(f'../input/prostate-cancer-grade-assessment/test_images'):
    print("still can not access the test file ?")
    sub = inference(da = feature_engineering(data = test , dir_name = "test_images") , dir_path = "test_images")
    sub['isup_grade'] = sub['isup_grade'].astype(int)
    sub.to_csv('submission.csv', index=False)
    
else:
    sub = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/sample_submission.csv")
    sub.to_csv('submission.csv', index=False)


# Final note, I know the model is substandard, but the objective was to keep everyone on the same page regarding how to submit in this competition. 
# Happy Kaggling 

# In[ ]:




