#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from matplotlib import pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Loading the dataset

# In[ ]:


data = pd.read_csv(r'/kaggle/input/military-expenditure-of-countries-19602019/Military Expenditure.csv')
data.head()


# We can drop the columns that contain trivial or irrelevant information

# In[ ]:


data = data.drop(['Indicator Name'], axis=1)


# One of the instances in the dataset contained the Military Expenditure date for the entire planet combined. We can extract that as a seperate series and then plot a portion of it whixh starts from the year 1990

# In[ ]:


World = data[data['Name']=='World']
World = World.drop(['Code', 'Type'], axis=1)
World = World.set_index('Name')
World.index = World.index.rename('Year')
World = World.T
World = World[30:]
World.head()


# In[ ]:


WLD = World['World']/1e9
plt.figure(figsize=(8, 5.5), linewidth=1)
WLD.plot(linestyle='-', marker='*', color='b')
title_font = {'size':'22', 'weight' : 'bold'}
plt.title('Annual Global Military Spending', **title_font)
axis_font = {'size':'22', 'weight' : 'bold'}
plt.ylabel('Billion US $', **axis_font)
plt.xlabel('Years', **axis_font)
plt.grid(color='y', linestyle='-', linewidth=0.5)
font = {'weight' : 'bold', 'size' : 22}
matplotlib.rc('font', **font)


# In addition to data pertaining to countries, we have many other categories. Let's filter out the data only belonging to countries

# In[ ]:


Nations = data[data['Type']=='Country']
Nations = Nations.drop(['Code', 'Type'], axis=1)
Nations = Nations.set_index('Name')
Nations.index = Nations.index.rename('Year')
Nations = Nations.dropna(axis=0, how='all')
Nations = Nations.T
Nations = Nations[30:]
Nations.head()


# By sorting and selecting after filtering for the year 2018, we can see the top 10 spenders for that year.

# In[ ]:


Top_10 = pd.DataFrame((Nations.T['2018']).sort_values(ascending=False)[:10])
Top_10 = Top_10/1e9
Top_10.index = Top_10.index.rename('Country')
Top_10 = Top_10.rename(columns={'2018':'Billion $'})
Top_10


# In[ ]:


Top_10.plot(kind='bar')


# To make intuitive comparisons, instead of comparing actual values, we can first normalize all values with the total World values as base.

# In[ ]:


Nations_world_share = Nations.copy()
cols = Nations_world_share.columns
for col in cols:
    Nations_world_share[col] = (Nations_world_share[col]/ World['World'])*100
Nations_world_share.head()


# According to the Global Firepower Index, these four countries are the top 4 strongest firepowers. Let's see how they have been spending on their militaries in the last 3 decades.

# In[ ]:


USA = Nations_world_share['United States']
CHN = Nations_world_share['China']
RUS = Nations_world_share['Russian Federation']
IND = Nations_world_share['India']


# In[ ]:


plt.figure(figsize=(8, 8))
USA.plot(linestyle='-', color='blue')
CHN.plot(linestyle='-', color='red')
RUS.plot(linestyle='-', color='green')
IND.plot(linestyle='-', color='orange')
plt.title('Military Spending of 4 most Powerful Countries Today', **title_font)
plt.ylabel('% of world share', **axis_font)
plt.xlabel('Year', **axis_font)
plt.legend(['USA', 'China', 'Russia', 'India',])
plt.grid(color='y', linestyle='-', linewidth=0.5)


# So here we can see that US is stiil way ahead of the rest of the world when it comes to its military budget. China though is catching up really fast. In this kernel, we cleaned the data, and did some insightful visual inspection. Next, we will use this data to predict the possible future trends.
