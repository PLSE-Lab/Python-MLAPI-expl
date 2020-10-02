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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Loading the libraries.

# In[ ]:


dataset=pd.read_csv('E:/forestfire/forestfires.csv')


# loaded the forest fire data from the local directory.

# In[1]:


dataset.head()


# command shows the heads of the variables present in the dataset. As it can be seen that the data contain verbal form also.

# dataset.describe(include='all')

# In[2]:


dataset.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)


# converting the months into numerical form.

# In[3]:


dataset.day.replace(('mon','tue','wed','thr','fri','sat','sun'),(1,2,3,4,5,6,7),inplace=True)


# same for the weekdays

# In[4]:


dataset.head()


# Checking the entire data to see whether the data is converted into numeral form or not.

# In[5]:


print("correlation:",dataset.corr(method='pearson'))


# finding the correlation between different variables using pearson method.

# In[6]:


corr = dataset.corr(method='pearson')
f,ax = plt.subplots(figsize=(16, 16))
print("correlation:")
sns.heatmap(corr(),annot= True, linewidths=.5, fmt='.1f', ax= ax)
plt.show()


# Plotting the corelation between different variables.
# Thus it helps for further process of machine learning to find the various answers like the variables involved and responsible for the forest fire.
