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


import pandas as pd
df = pd.read_csv("../input/videogamesales/vgsales.csv")


# In[ ]:


df.head()


# In[ ]:


df['Genre'].unique()


# In[ ]:


import plotly.express as px


# In[ ]:


px.box(df, x='Genre', y='Global_Sales', hover_data=['Name', 'Rank'])


# In[ ]:


df[['Genre', 'Global_Sales']].groupby('Genre').mean().head(100)


# In[ ]:


px.box(df, y='Global_Sales')


# In[ ]:


px.scatter(df, x='Year', y='Global_Sales')


# In[ ]:


px.box(df, x='Publisher', y='Global_Sales', hover_data=['Name', 'Rank'])


# In[ ]:


px.box(df, x='Platform', y='Global_Sales', hover_data=['Name', 'Rank'])


# In[ ]:




