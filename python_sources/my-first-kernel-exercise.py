#!/usr/bin/env python
# coding: utf-8

# This is my exercise of something about datascience. I am a learner and this is my homework given by @DATAI 

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

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/tmdb_5000_movies.csv")


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.dtypes #budget = integer, popularity = float


# In[ ]:


print(data["budget"])


# In[ ]:



data["popularity"] = data.popularity.astype(int)
data["popularity"] = data.popularity*1000000
print(data["popularity"].dtypes)  
print (data["popularity"])


# In[ ]:


data.budget.plot(color = "k", label = "Budget", grid = True, linestyle = "--")
data.popularity.plot(color = "r", label = "Popularity", linewidth= 1 , grid = True, linestyle = "--")
plt.show()


# In[ ]:


plt.scatter(data.budget, data.popularity, color = "r")


# In[ ]:


data.budget.plot(kind = "hist", bins = 100)


# In[ ]:


data[np.logical_and(data["vote_average"]>8,data["popularity"]>300000000)]


# In[ ]:




