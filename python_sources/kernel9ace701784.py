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


data = pd.read_csv('../input/ARCH4450_Final_BatuhanDolgun_PnarEngr_MerveKoyiit_MehmetTakran.csv')
data.head(5)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.pairplot(data)


# In[ ]:


X= data[['FLOOR','WALL','COLOUR']]
X.head()


# In[ ]:


data.tail()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(5,5))
sns.heatmap(data.corr(), annot=True)


# In[ ]:


y= data['COLOUR']
y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=100 )


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression 
lr = LinearRegression()#Creating a LinearRegression object
lr.fit(X_train, y_train)

