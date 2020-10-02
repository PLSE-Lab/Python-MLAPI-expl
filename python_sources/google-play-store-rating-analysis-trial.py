#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/googleplaystore.csv")

data.info()

data.columns

# Any results you write to the current directory are saved as output.


# In[ ]:


data.head(10)


# In[ ]:


data.tail(10)


# In[ ]:


data.Installs = data.Installs.str.replace(",", "")
data.Installs = data.Installs.apply(lambda x: x.strip("+"))
data.Installs = data.Installs.replace("Free", "")
data.Installs = pd.to_numeric(data.Installs)


# In[ ]:


plt.scatter(data['Rating'], data['Installs'], alpha = 0.5, color = 'DarkBlue')
plt.xlabel('Installs')
plt.xlabel('Rating')
plt.grid()
plt.show()


# In[ ]:


install = data['Rating'] > data.Rating.mean()
data[install]
        


# In[ ]:


for index, value in enumerate(list(data['App'][:51])):
    print(index, value)


# In[ ]:


colmCat = data.Category.head(100)
colmRat = data.Rating.head(100)
newColmn = pd.concat([colmCat, colmRat], axis = 1)

data.Category.value_counts().plot(kind='barh',figsize= (12,8))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




