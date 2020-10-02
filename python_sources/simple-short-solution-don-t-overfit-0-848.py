#!/usr/bin/env python
# coding: utf-8

# ##Importing data

# 

# In[1]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler , RobustScaler

from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score

from sklearn.linear_model import LogisticRegression
import os
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv' , index_col = 'id')
test = pd.read_csv('../input/test.csv'  , index_col = 'id')

print(os.listdir("../input/"))


# In[2]:


y = train['target']
train = train.drop(['target'] , axis = 1)
train.info()


# In[3]:


test.info()


# In[4]:


train.head()


# In[5]:


scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)


# In[6]:


clf = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)
clf.fit(train, y)
scores = cross_val_score(clf, train, y, cv=5)
scores


# In[7]:


clf.fit(train, y)
ans = clf.predict_proba(test)
ans


# In[8]:


submit = pd.read_csv('../input/sample_submission.csv')
submit.head()


# In[9]:


ans


# Required result is stored in second column so, let's extract it .

# In[10]:


submit['target'] = ans[:,1]


# Create submition file

# In[11]:


submit.to_csv('submit.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




