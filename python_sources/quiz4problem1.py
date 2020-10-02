#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.loc[:, "Name"]


# In[ ]:


df.loc[df['Average User Rating'] > 4.95,:]


# In[ ]:


df[['Average User Rating']].median()


# In[ ]:


df.groupby('Genres')
df.first


# In[ ]:


df.groupby('Genres').size()


# In[ ]:


df.plot.scatter(x = 'Average User Rating', y = 'User Rating Count')


# In[ ]:


df.groupby('Genres').size().sort_values(ascending = False)
# See which games have the highest ratings, but also with the highest count of ratings to know which have the highest ratings by the most people


# In[ ]:


df.groupby('Genres').size().sort_values(ascending = False).max()
# See the most common genres available in the dataset

