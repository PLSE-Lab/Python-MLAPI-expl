#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


ntrain=pd.read_csv('../input/train.csv')
ntest=pd.read_csv('../input/test.csv')    


# In[8]:


ntrain.shape


# In[9]:


ntest.shape


# In[10]:


ntrain.head()


# In[11]:


# Create target vector
y = ntrain.label


# In[12]:


### Create features vector
x = ntrain.drop('label', axis=1)


# In[13]:


x.shape


# In[14]:


# Load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets


# In[15]:


# Create random forest classifer object that uses entropy
n_estimators=100
clf = RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1, n_estimators=n_estimators)


# In[16]:


# Train model
model = clf.fit(x, y)


# In[17]:


model.score(x,y)


# In[18]:


# Predict observation's class    
y_pred=model.predict(x)


# In[19]:


y_pred


# In[20]:


from sklearn.metrics import confusion_matrix


# In[21]:


confusion_matrix(y, y_pred)


# In[22]:


y_rf=model.predict(ntest)


# In[23]:


y_rf


# In[24]:


solution = pd.DataFrame({"ImageId": ntest.index+1, "Label":y_rf})


# In[25]:


solution.to_csv("new_digit_rf_submission.csv", index = False)


# In[26]:


pd.value_counts(y).plot(kind='bar', legend =True, table= True, figsize=(10,10))


# In[27]:


pd.value_counts(y).plot(kind='pie', legend =True, table= True, figsize=(10,10))


# In[28]:


pd.value_counts(y).plot(kind='density', legend =True, figsize=(10,10))


# In[29]:


#### With Standardisation
# Load libraries
from sklearn import preprocessing


# In[30]:


# Create scaler
scaler = preprocessing.StandardScaler()


# In[31]:


# Transform the feature
x_new = scaler.fit_transform(x)


# In[32]:


# Print mean and standard deviation
print('Mean:', round(x_new[:,0].mean()))
print('Standard deviation:', x_new[:,0].std())


# In[33]:


# Train model
model = clf.fit(x_new, y)


# In[34]:


clf.score(x_new, y)


# In[35]:


test_new=scaler.fit_transform(ntest)


# In[36]:


y_rf1=clf.predict(test_new)


# In[37]:


y_rf1


# In[38]:


confusion_matrix(y_rf, y_rf1)


# In[39]:


solution = pd.DataFrame({"ImageId": ntest.index+1, "Label":y_rf1})
solution.to_csv("new_rfscaled_submission.csv", index = False)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


n_estimators=100
clf = GradientBoostingClassifier(n_estimators=n_estimator)


# In[ ]:


clf.fit(x,y)


# In[ ]:


clf.score(x,y)


# In[ ]:


y_gbm=clf.predict(ntest)


# In[ ]:


pd.crosstab(y_rf,y_gbm)


# In[ ]:


solution = pd.DataFrame({"ImageId": ntest.index+1, "Label":y_gbm})
solution.to_csv("new_gbm_submission.csv", index = False)

