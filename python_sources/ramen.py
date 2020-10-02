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
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/ramen-ratings.csv')
df.head()


# In[ ]:


df['Top Ten'].sort_values().head(10)


# The values in top ten are a little messy. Lets split the year and the position the ramen came into seperate columns, change any other value to NaN, then drop the Top Ten column

# In[ ]:


df['Year'] = df['Top Ten'].map(lambda x: str(x)[:4])
df['Position'] = df['Top Ten'].map(lambda x: str(x).split('#')[-1])

cols = ['Position','Year','Stars']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', downcast='signed')

df = df.drop('Top Ten', axis=1)


# In[ ]:


df.sort_values('Position').head()


# Much better. Now we're going to create a column called Score which gives a higher rating for a ramen's position.

# In[ ]:


df['Score'] = df.Position.map(lambda x: 11-x)


# Now to see which countries have consistently high rated Ramen.

# In[ ]:


df.groupby('Country').sum().sort_values('Score', ascending=False)['Score'].head(10).plot.bar()


# Singapore, and Malaysia look to be the best place to get good quality tasting Ramen. These are the peak best though. What about aggregating for all the Rated ramens in the DataFrame?

# In[ ]:


df.groupby('Country').mean().sort_values('Stars', ascending=False)['Stars'].head(10).plot.bar()


# Brazil and Sarawak have the highest rated ramen on average. So if you were to pick a pack of ramen at ramdom, doing so from a brazilian selection are your best odds at finding a good one.

# In[ ]:


g = df[df['Style'].isin(['Bowl','Box','Cup','Pack','Tray'])].groupby(['Style','Country']).agg({'Stars': 'mean'})
g = g['Stars'].groupby(level=0, group_keys=False)
g = g.nlargest(5).reset_index()


# In[ ]:


g[g.Style == "Bowl"].plot.bar(x='Country')


# In[ ]:


g[g.Style == "Box"].plot.bar(x='Country')


# In[ ]:


g[g.Style == "Cup"].plot.bar(x='Country')


# In[ ]:


g[g.Style == "Pack"].plot.bar(x='Country')


# In[ ]:


g[g.Style == "Tray"].plot.bar(x='Country')

