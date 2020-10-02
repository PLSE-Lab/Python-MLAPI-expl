#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Reading the dataset

# In[3]:


data=pd.read_csv('../input/heart.csv')
data.head()


# Let's analyse what factors are responsible for cause of heart disease

# Creating a training and testing dataset

# In[5]:


msk=np.random.rand(len(data))<0.8
train_df=data[msk]
test_df=data[~msk]


# In[7]:


len(train_df)


# In[8]:


len(test_df)


# In[9]:


from sklearn import ensemble


# In[11]:


train_df_x=train_df.drop(columns=['target'])
train_df_y=train_df.target

test_df_x=test_df.drop(columns=['target'])
test_df_y=test_df.target


# In[ ]:


regr1=ensemble.RandomForestClassifier()


# In[13]:


regr1.fit(train_df_x,train_df_y)


# In[14]:


from sklearn.metrics import accuracy_score


# In[16]:


train_ac=accuracy_score(regr1.predict(train_df_x),train_df_y)
train_ac


# In[17]:


test_ac=accuracy_score(regr1.predict(test_df_x),test_df_y)
test_ac


# Let's look at the feature importance

# In[42]:


pd.DataFrame({'Features':train_df_x.columns,'Importance':regr1.feature_importances_}).plot.bar(x='Features')


# Let's see what happens when we use Gradient Boosted Classifier

# In[36]:


gbt=ensemble.GradientBoostingClassifier()


# In[37]:


gbt.fit(train_df_x,train_df_y)


# In[38]:


train_ac=accuracy_score(gbt.predict(train_df_x),train_df_y)
train_ac


# In[40]:


test_ac=accuracy_score(gbt.predict(test_df_x),test_df_y)
test_ac


# This shows GBT had overfitting problem.
# 
# It is learning well on train model, but can't apply the same on test model.

# Let's look at Feature Importance

# In[64]:


pd.DataFrame({'Features':train_df_x.columns,'Importance':gbt.feature_importances_}).plot.bar(x='Features',color='orange')


# As you can see, both the models give different degree of feature importances.

# In[62]:


pd.DataFrame({'Features':train_df_x.columns,
              'Importance_RF':regr1.feature_importances_,
              'Importance_GBT':gbt.feature_importances_}).plot.bar(x='Features')

