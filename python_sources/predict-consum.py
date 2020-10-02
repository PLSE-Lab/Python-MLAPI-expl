#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('pylab', 'inline')
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_excel("../input/Planon.xlsx")


# In[3]:


data.head(5)


# In[4]:


#transform from date to float
j = 0
for i in data["REGISTRATIONDATE"]:
    data.loc[j,"REGISTRATIONDATE"] = i.timestamp()
    j += 1


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


Target = data["CONSUMPTION"]
Data = data.drop(["CONSUMPTION"], axis=1).values


# In[8]:


std_scale_data = StandardScaler()


# In[9]:


std_scale_data.fit(Data)


# In[10]:


Data = std_scale_data.transform(Data)


# In[15]:


from sklearn import decomposition

pca = decomposition.PCA(n_components=3)
pca.fit(Data)
Data = pca.transform(Data)


# In[16]:


print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.sum())


# In[17]:


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Data,Target,test_size=0.2)


# In[18]:


X_train.shape


# In[ ]:




