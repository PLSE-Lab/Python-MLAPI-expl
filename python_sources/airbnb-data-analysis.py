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


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df = pd.DataFrame(data=airbnb)
df.head()


# In[ ]:


df.columns


# In[ ]:


df.sort_values('neighbourhood').head()


# In[ ]:


df.describe()


# In[ ]:


# Average Prices for each neighbourhood
df.groupby('neighbourhood')['price'].mean()


# In[ ]:


# Maximum price listed for each neighbourhood
df.groupby('neighbourhood')['price'].max()


# In[ ]:


# There are 3 airbnbs with $10000 price and 11 airbnbs with $0 price
df.groupby('price')['neighbourhood'].count()


# In[ ]:


# Astoria, Greenpoint and Upper West Side
df.loc[df.price == 10000]


# In[ ]:


# 11 listings priced at 0
df.loc[df.price == 0]


# In[ ]:


# All the listings sorted by availability
df.sort_values('availability_365')


# In[ ]:


df['neighbourhood_group'].unique()


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', s=20, data=df)

