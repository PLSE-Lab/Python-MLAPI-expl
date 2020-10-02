#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Reading data

# In[ ]:


df = pd.read_csv('/kaggle/input/belarus-used-cars-prices/cars.csv')


# ## Quick view

# In[ ]:


df.head()


# ## Price distribution

# In[ ]:


df['priceUSD'].hist(bins=40);


# ## Price outliers

# In[ ]:


sns.boxplot(df['priceUSD']);  # yeah, that`s cool


# ## Categorical features values distribution

# In[ ]:


sns.countplot(df['segment']);


# In[ ]:


sns.countplot(df['fuel_type']);


# In[ ]:


sns.countplot(df['transmission']);


# In[ ]:


sns.countplot(df['condition']);


# In[ ]:


sns.countplot(df['drive_unit']);


# ## Make distribution

# In[ ]:


labels = df['make'].value_counts().index
values = df['make'].value_counts().values

fig = px.pie(df, values=values, names=labels, title='Make distribution')
fig.show()


# ## Segment distribution

# In[ ]:


labels = df['segment'].value_counts().index
values = df['segment'].value_counts().values

fig = px.pie(df, values=values, names=labels, title='Segment Distribution')
fig.show()


# ## Colors :)

# In[ ]:


labels = df['color'].value_counts().index
values = df['color'].value_counts().values

fig = px.pie(df, values=values, names=labels, title='Segment Distribution')
fig.show()


# ## Scatter plot price by year

# In[ ]:


fig = px.scatter(x=df['year'], y=df['priceUSD'])
fig.show()


# In[ ]:


fig = px.scatter(x=df['mileage(kilometers)'], y=df['priceUSD'])
fig.show()


# # So

# This website market allows people to type the mileage by hand, so sometimes we have a cars with 10kk mileage. 

# In[ ]:


sns.boxplot(df['mileage(kilometers)']);

