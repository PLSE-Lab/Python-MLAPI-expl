#!/usr/bin/env python
# coding: utf-8

# In[1]:


#First public kernel, Here goes!


# In[2]:


import os
import numpy as np 
import pandas as pd 


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


#Load Data
input_dir = '../input/'
train = pd.read_csv(input_dir + 'train.csv')
test = pd.read_csv(input_dir + 'test.csv')
resources = pd.read_csv(input_dir + 'resources.csv')
submission = pd.read_csv(input_dir + 'sample_submission.csv')


# In[5]:


#Get the total price of each resource
resources['total_price'] = resources.quantity * resources.price
resources.head()


# In[6]:


#For every project, get its mean price
mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean()) 
mean_total_price.head()


# In[7]:


#Add id as column for merging
mean_total_price['id'] = mean_total_price.index
train = pd.merge(train, mean_total_price, on='id')


# In[8]:


X_train = train.total_price.values.reshape(-1, 1) #reshape because of one column
y_train = train.project_is_approved.values


# In[9]:


#CV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
n_folds = 5 
kf = KFold(n_splits=n_folds)
Average_AUC = 0
for (train_index, test_index) in kf.split(X_train):
    X, X_val = X_train[train_index], X_train[test_index]
    y, y_val = y_train[train_index], y_train[test_index]
    clf = LogisticRegression(random_state=333)
    clf.fit(X, y)
    pred = clf.predict_proba(X_val)[:,1]
    AUC = roc_auc_score(y_val, pred)
    Average_AUC += AUC
    print("AUC: {}".format(AUC))
Average_AUC = Average_AUC / n_folds
print("Average_AUC: {}".format(Average_AUC))


# In[10]:


test = pd.merge(test, mean_total_price, on='id')


# In[11]:


X_test = test.total_price.values.reshape(-1, 1)
pred_test = clf.predict_proba(X_test)[:,1]


# In[12]:


submission.project_is_approved = pred_test
submission.head(5)


# In[13]:


submission.to_csv('submission.csv', index=False)

