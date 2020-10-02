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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


dataset = pd.read_csv("../input/train.csv")
print(dataset.head(20))


# In[ ]:


import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.columns


# In[5]:


import seaborn as sns
sns.distplot(dataset['SalePrice'])


# In[11]:


dataset.corr()


# In[10]:


corr = dataset.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)


# In[26]:


X = dataset[['LotArea']]
y = dataset['SalePrice']


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=101)


# In[28]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[29]:


predictions = lm.predict(X_test)


# In[30]:


import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)

