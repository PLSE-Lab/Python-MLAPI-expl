#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
import seaborn as sns  # visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_movies = pd.read_csv("../input/tmdb_5000_movies.csv")

data_movies.info()


# In[ ]:





# In[ ]:


#correlation map

f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data_movies.corr(), annot=True, linewidth=0.3, fmt=".1f", ax=ax)
plt.show


# In[ ]:


data_movies.columns


# 

# In[ ]:


data_movies.budget.plot(kind="line", color="b", label="Budgets", linewidth=0.85, alpha=0.7, grid=True, figsize=(10,10))
data_movies.revenue.plot(kind="line", color="r", label="Revenue", linewidth=0.75, alpha=0.5,)
plt.legend(loc="upper center")
plt.show()


# In[ ]:


data_movies.plot(kind="Scatter", x="budget", y="revenue", alpha=0.6, grid=True, color="green", figsize=(10,10))
plt.show()


# In[ ]:


data_movies.revenue.plot(kind="hist", bins=50, figsize=(10,10))
plt.show


# applying a filter for finding movies that owns higher votes ( higher than 8)
# 

# In[ ]:


filtre = data_movies["vote_average"]>8
data_movies[filtre]


# i have found the movies have got a vote average higher than 8 but i noticed that some movies has been voted by just 2 or 3 people. so i want to eliminate these. to do that i will improve my filter which will filter not just vote average but also vote count.

# In[ ]:


filtre2 = data_movies[(data_movies["vote_average"]>8) & (data_movies["vote_count"]>3000)]
filtre2


# now i have got a dataframe which could be assumed as movies should be watched :). maybe now i should do some analysis just on this dataframe. so ill call this new dataframe as best movies.

# In[ ]:


best_movies = filtre2

best_movies.info()
best_movies.columns


# lets have a look if there is another correlation, when compared to all movies, between variables in "best movies"

# In[ ]:


f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(best_movies.corr(), annot=True, linewidth=0.3, fmt=".1f", ax=ax)
plt.show


# In[ ]:


best_movies.plot(kind="Scatter", x="budget", y="revenue", alpha=0.6, grid=True, color="green", figsize=(10,10), s=150)
plt.show()


# In[ ]:


best_movies.budget.plot(kind="line", color="b", label="Budgets", linewidth=3, alpha=0.7, grid=True, figsize=(10,10))
best_movies.revenue.plot(kind="line", color="r", label="Revenue", linewidth=3, alpha=0.5,grid=True)
plt.legend(loc="upper center")
plt.show()


# In[ ]:




