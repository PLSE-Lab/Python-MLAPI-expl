#!/usr/bin/env python
# coding: utf-8

# # ** This notebook mainly deals with different visualization techniques and how we will use them to find out meaningful insights from the data. At the end please provide your suggestions. So lets get started. **
# 
# ## Importing the libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import warnings
warnings.filterwarnings('ignore')
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/commodity_trade_statistics_data.csv')


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# ## Which year saw the highest import export? 

# In[ ]:


df_ie2=df.groupby(['year'],as_index=True)['weight_kg','quantity'].agg('sum')
df_ie=df.groupby(['year'],as_index=False)['weight_kg','quantity'].agg('sum')


# In[ ]:


df_ie2.head(1)


# In[ ]:


df_ie.head(1)


# In[ ]:


df_ie2.plot(figsize=(12,6))


# So we see that there's a sudden jump in quantity in the year around 1996-97, but no weight_change. That's an outlier, I think, what do you say?

# In[ ]:


temp1 = df_ie[['year', 'weight_kg']] 
temp2 = df_ie[['year', 'quantity']] 
# temp1 = gun[['state', 'n_killed']].reset_index(drop=True).groupby('state').sum()
# temp2 = gun[['state', 'n_injured']].reset_index(drop=True).groupby('state').sum()
trace1 = go.Bar(
    x=temp1.year,
    y=temp1.weight_kg,
    name = 'Year with Import/Export in terms of Weight (Kg.)'
)
trace2 = go.Bar(
    x=temp2.year,
    y=temp2.quantity,
    name = 'Year with Import/Export in terms of no. items (Quantity)'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Import/Export in terms of Weight (Kg.)', 'Year with Import/Export in terms of no. items (Quantity)'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout']['xaxis1'].update(title='Year')
fig['layout']['xaxis2'].update(title='Year')

fig['layout']['yaxis1'].update(title='Year with Import/Export in terms of Weight (Kg.)')
fig['layout']['yaxis2'].update(title='Year with Import/Export in terms of no. items (Quantity)')
                          
fig['layout'].update(height=500, width=1500, title='Import/Export in terms of Weight(kg.) & No. of Items')
iplot(fig, filename='simple-subplot')


# In[ ]:


df.shape


# ## Visualizing some insights from data

# ## Ratio of Imports and Exports

# In[ ]:


cnt_srs = df['flow'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic'
    ),
)

layout = go.Layout(
    title='Imports/Exports Ratio'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# # Lets have a look at which countries have been dominating the global trade

# In[ ]:


df_country=df.groupby(['country_or_area'],as_index=False)['weight_kg','quantity'].agg('sum')


# In[ ]:


df_country=df_country.sort_values(['weight_kg'],ascending=False)


# In[ ]:


fig, ax = plt.subplots()

fig.set_size_inches(13.7, 8.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df_country["country_or_area"].head(20), y=df_country['weight_kg'].head(20), data=df_country)
f.set_xlabel("Name of Country",fontsize=15)
f.set_ylabel("Import/Export Amount",fontsize=15)
f.set_title('Top countries dominating Global Trade')
for item in f.get_xticklabels():
    item.set_rotation(90)


# ### So its China

# Since China is the most dominating, we'll come to China specifically at a later stage to see what are the products they have been trading. Before that lets visualize some more  data

# # Lets have a look at the commodity items which have been traded maximum!

# In[ ]:


df_commodity=pd.concat([df['commodity'].str.split(', ', expand=True)], axis=1)


# In[ ]:


df_commodity.head()


# In[ ]:


temp_series = df_commodity[0].value_counts().head(20)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Top 20 Commodity Items to be Traded',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Commodity")


# # Lets check out which animals are always in high demand for trading

# In[ ]:


df_animals=df.ix[df['category']=='01_live_animals']


# In[ ]:


df_animals.head(1)


# In[ ]:


df_animals['animal']=df_animals['commodity'].str.split(',').str[0]


# ## Top 20 animals traded

# In[ ]:


cnt_srs = df_animals['animal'].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1]
    ),
)

layout = dict(
    title='Animals Traded according to demand',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Animals")


# ## Lets now have look at China's trading, how are they most dominating country in Global Trade

# In[ ]:


df_china=df.ix[df['country_or_area']=='China']


# In[ ]:


df_china.head(1)


# In[ ]:


df_chinaie2=df_china.groupby(['year'],as_index=True)['weight_kg','quantity'].agg('sum')


# In[ ]:


df_chinaie2.plot(figsize=(10,6))


# ### This is how China has done for the past 20 years

# In[ ]:


cnt_srs = df_china['flow'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values
    ),
)

layout = go.Layout(
    title='Imports/Exports Ratio (China)'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# In[ ]:


df_china_export=df_china.ix[df_china['flow']=="Export"]


# In[ ]:


df_china_commodity=pd.concat([df_china_export['commodity'].str.split(', ', expand=True)], axis=1)


# In[ ]:


df_china_commodity.head(1)


# In[ ]:


temp_series = df_china_commodity[0].value_counts().head(20)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Top 20 Commodity Items Exported by China',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Commodity")


# ### Nice

# ### Ok so, let's have a quick look at what China imports maximum!

# In[ ]:


df_china_import=df_china.ix[df_china['flow']=="Import"]


# In[ ]:


cnt_srs = df_china_import['category'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1])
)

layout = dict(
    title='Top Imported Items by China',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Animals")


# In[ ]:


df_china_icommodity=pd.concat([df_china_import['commodity'].str.split(', ', expand=True)], axis=1)


# ## Now we'll have a look at my motherland's statistics, India's trading

# In[ ]:


df_india=df.ix[df['country_or_area']=='India']


# In[ ]:


df_indiaie2=df_india.groupby(['year'],as_index=True)['weight_kg','quantity'].agg('sum')


# In[ ]:


df_indiaie2.plot(figsize=(10,6))


# ### This is how India has done in the last 20yrs in the global trading market

# In[ ]:


cnt_srs = df_india['flow'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values
    ),
)

layout = go.Layout(
    title='Imports/Exports Ratio (India)'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# In[ ]:


df_india_export=df_india.ix[df_india['flow']=="Export"]


# In[ ]:


df_india_commodity=pd.concat([df_india_export['commodity'].str.split(', ', expand=True)], axis=1)


# In[ ]:


df_india_import=df_india.ix[df_india['flow']=="Import"]


# In[ ]:


df_india_commodity_import=pd.concat([df_india_import['commodity'].str.split(', ', expand=True)], axis=1)


# ## Lets have a look at India's imported and exported items

# In[ ]:


temp_series = df_india_commodity[0].value_counts().head(20)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Top 20 Commodity Items Exported by India',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Commodity")

temp_series = df_india_commodity_import[0].value_counts().head(20)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Top 20 Commodity Items Imported by India',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Commodity")


# # Do upvote if you've liked  it and don't forget to checkout my other kernels

# In[ ]:




