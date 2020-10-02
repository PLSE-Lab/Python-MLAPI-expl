#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


amazon = pd.read_csv("/kaggle/input/amazon-prime-tv-shows/Prime TV Shows Data set.csv",encoding="iso-8859-1")


# In[ ]:


amazon.info()


# In[ ]:


#Lot of IMDB ratings are missing (283)
amazon.isnull().sum()


# In[ ]:


#Now lets try to see missing IMDB values by Genre
amazon.groupby('Genre')['S.no.'].count()
#amazon['Genre'].value_counts()


# In[ ]:


#Dropping columns where IMDB rating is missing
amazon1 = amazon.dropna(axis=0)
amazon1.info()


# In[ ]:


amazon1.head()


# In[ ]:


amazon_2020 = amazon1[amazon1['Year of release'] == 2020].sort_values(by='IMDb rating',ascending = False)
#Top 5 movies of 2020
amazon_2020.head(5)


# In[ ]:




