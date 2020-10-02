#!/usr/bin/env python
# coding: utf-8

# ***Insights :
# 1. The city with most number of Pizza outlets
# 2. Find out cities with most number of outlets serving unique Pizzas (using Scattergeo in plotly)*******
# 
# Please upvote if you like it :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.basemap import Basemap
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


        # Any results you write to the current directory are saved as output.


# In[ ]:


#Examining a sample ..

pizza_df = pd.read_csv("../input/pizza-restaurants-and-the-pizza-they-sell/8358_1.csv")
pizza_df.head()


# In[ ]:


#Check for nulls in main fields 
pizza_df.isnull().sum()


# In[ ]:


#No.of unique cities ..
pizza_df['city'].nunique()


# **1. The city with most number of Pizza outlets**

# In[ ]:


pizza_df['city'].value_counts().head(1)


# In[ ]:


pizza_df.head()


# In[ ]:


pizza_df['menus.name'].unique()


# In[ ]:


#Rare Pizza(with occurence = 1)
city_pizza=pizza_df[['id','city','latitude','longitude','name','menus.name']]
s1=city_pizza['menus.name'].value_counts()
s2=s1[s1 == 1].index.tolist()
subs="Pizza,"
s3=[x for x in s2 if not re.search(subs, x)]
#s4=np.random.choice(s3,20)
plt_data=city_pizza[city_pizza['menus.name'].isin(s3)]


# ** Using simple scatter plot to identify if there is any pattern wrt locations for  most number of special/unique Pizzas

# In[ ]:


#plt.scatter(x=plt_data['longitude'],y=plt_data['latitude'])
plt_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)
plt.show()


# **2. Find out cities with most number of outlets serving unique Pizzas (using Scattergeo in plotly)**
# 
# We could infer some patterns wrt locations as per above scatter plot(for city), now lets use plotly Scattergeo to find out corresponding locations 

# In[ ]:


fig = go.Figure(data=go.Scattergeo(
        lon = pizza_df['longitude'],
        lat = pizza_df['latitude'],
        text = pizza_df['city'],
        mode = 'markers',
        ))
fig.update_layout(
        title = 'Cities with outlets serving unique Pizzas<br>(Hover for city names)',
        geo_scope='usa',
    )
fig.show()

