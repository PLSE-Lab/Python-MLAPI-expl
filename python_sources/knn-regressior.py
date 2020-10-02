#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


# In[23]:


import sklearn
from sklearn import cluster
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score , accuracy_score , confusion_matrix , classification_report
import matplotlib.pyplot as plt


# In[29]:


data = pd.read_csv('../input/xclara.csv')
data = data.dropna(how = 'any')
validation , train = train_test_split(data , test_size = 0.3)
x_train = train.iloc[: , 0:1].values
y_train = train.iloc[: , 1:].values
x_validation = validation.iloc[: , 0:1].values
y_validation = validation.iloc[: , 1:]
plt.scatter(x_train , y_train)
plt.show()


# In[30]:


model = cluster.KMeans(n_clusters = 3)
model.fit(x_train)


# In[31]:


y_predict = model.predict(x_validation)


# In[32]:


regressior = KNeighborsRegressor(n_neighbors=3)
regressior.fit(x_train , y_train)


# In[33]:


y_reg_predict = regressior.predict(x_validation)
print(regressior.score(x_validation , y_validation))


# In[ ]:





# In[ ]:




