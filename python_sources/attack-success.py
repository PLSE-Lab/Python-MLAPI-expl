#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')


# In[ ]:


plt.figure(figsize=(12, 12))
sns.heatmap(train_csv.corr().loc[:'alcohol', 'quality':])


# In[ ]:


import copy
train_D = train_csv.drop(columns = ['index','quality','density'], axis = 1)
train_L = copy.deepcopy(train_csv.quality)


# In[ ]:


test_D = test_csv.drop(columns =  ['index','density'], axis = 1)


# In[ ]:


train_D.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
mscaler = MinMaxScaler()
scaler = StandardScaler()
train_D = scaler.fit_transform(np.array(train_D))
test_D = scaler.transform(np.array(test_D))


# In[ ]:


train_D = torch.FloatTensor(train_D)
train_L = torch.LongTensor(train_L)
test_D = torch.FloatTensor(test_D)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


RF = RandomForestClassifier()


# In[ ]:


RF


# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'max_depth':[None, 3,4,5,6], 'min_samples_leaf':[1,2,3],'n_estimators' :[100,200,500,1000]}
cv= GridSearchCV(RF, params)
cv.fit(train_D, train_L)


# In[ ]:


cv.best_params_


# In[ ]:


RF.fit(train_D, train_L)


# In[ ]:


pred = cv.predict(test_D)


# In[ ]:


pred


# In[ ]:


sample = pd.read_csv('sample_submission.csv')
sample['quality'] = pred
sample


# In[ ]:


sample.to_csv('happyRF.csv', index = False)

