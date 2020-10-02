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


data = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


data.head(10)


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:





# In[ ]:


data["Type"] = data["Type"].map({"Free":0, "Paid":1})


# In[ ]:


data.head()


# In[ ]:


data["Size"] = data["Size"].map(lambda x:x.rstrip("M"))


# In[ ]:


data.head()


# In[ ]:


data["Installs"] = data["Installs"].map(lambda x: x.rstrip("+"))


# In[ ]:


data["Price"] = data["Price"].map(lambda x:x.lstrip("$"))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set_style("darkgrid")


# In[ ]:


data.head()


# In[ ]:


X= data[["Reviews","Size", "Type", "Installs", "Price"]]


# In[ ]:


y = data["Rating"]


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train, y_train)


# In[ ]:




