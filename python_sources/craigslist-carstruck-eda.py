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
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df1= pd.read_csv('../input/craigslistVehiclesFull.csv')


# First things first, I check the head to get a general idea of the dataframe

# In[ ]:


df1.head()


# In[ ]:


initshape = df1.shape


# I want to see how much luxury cars are driving up the average price, so I check the quartiles:
# 
# I see the standard deviation is large (in the millions) and that 75% of cars are below ~17 000$.
# 
# Still, the max price is too high, so I check for where to omit this by looking at further quantiles.

# In[ ]:


print("The max and min respectively")
print(df1['price'].max())
print(df1['price'].min())


# In[ ]:


print(df1.describe()['price'])
print(df1.quantile(q=[0.01,0.05,0.8,0.85,0.90,0.95,0.99])['price'])


# To me it seems like a good cutoff point for my cars would be at least 52000, but perhaps removing cars with a value over 75k would be better.
# Notice we only shave off 5255 listings out of ~547 000.
# 
# It's also a good idea at this point to remove any cars with low prices. I'd wager no car is worth less than 250 dollars given scrap value.

# In[ ]:


df1 = df1.drop(df1[df1.price > 75000].index)
secondshape = df1.shape
print("Listings removed >75000 :")
print(initshape[0] - df1.shape[0])


# In[ ]:


df1 = df1.drop(df1[df1.price < 250].index)
print("Listings removed == 0 :")
print(secondshape[0] - df1.shape[0])


# I begin by looking for any **NaN** values across the dataset. 
# 
# I spot the following categoricals in the heatmap to disregard:
# condition, cylinders, VIN, drive, size, paint_color

# In[ ]:


plt.figure(figsize=(9,9))
sns.heatmap(df1.isnull(),cmap="Pastel1");


# It's a good idea to remove any column that has too many null values, so it is done here:

# In[ ]:


del df1['condition']
del df1['odometer']
del df1['cylinders']
del df1['vin']
del df1['drive']
del df1['size']
del df1['type']
del df1['paint_color']


# A few things I want to begin looking at off the top of my head:
# 
# What are the most popular manufacturers?
# Whats the distribution of the sale prices?

# In[ ]:


plt.figure(figsize=(24,6))
ax = sns.countplot(x='manufacturer',data=df1,order=df1['manufacturer'].value_counts().index[:30])
locs, labels = plt.xticks();
plt.setp(labels, rotation=45);
ax.set_title("Listing Count per Manufacturer - top 30");



# In[ ]:


x = df1.price

f, (ax_hist, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.75, .25)},figsize=(25,8))
sns.distplot(x,bins=75,ax=ax_hist)
sns.boxplot(x,ax= ax_box,fliersize = 0)
ax_hist.set(title="Price distribution of car listings",xlabel = '');


# Looking at the above chart we find that the distribution is left-skewed and the boxplot presents itself neater than a list of quantiles.

# What about the distribution of manufacturing years?
# 
# First we have to remove a few listings which have no year or a year prior to 1960.

# In[ ]:


print('Unique years in dataset')
print(df1.year.unique())
df1.drop(df1[df1.year.isnull()].index, inplace = True)
df1.drop(df1[df1.year < 1960].index, inplace = True)


# In[ ]:


plt.figure(figsize=(24,10))
ax3 = sns.distplot(df1.year)
ax3.set(title="Distribution of Year in car listings");


# In[ ]:


print("Top 10 model years of listings")
print(df1.year.value_counts().iloc[:10])


# Out of curiosity, I'd like to see the listing count per state.

# In[ ]:


statecount = df1.state_code.value_counts()


# In[ ]:


datamap = dict(type='choropleth',
            colorscale = 'Reds',
            locations = statecount.index,
            z = statecount,
            locationmode = 'USA-states',
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Cars listed per State"}
            ) 


# In[ ]:


layout = dict(title = 'Cars listed per State',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )


# In[ ]:


choromap = go.Figure(data = [datamap],layout = layout)
iplot(choromap)


# We find out that the top 3 states for car listings on craigslist in decreasing order were: California, Florida and Texas.
# 
# What about the median car price?
# 

# In[ ]:


medpriceXX = df1.groupby('state_code')['price'].median()


# In[ ]:


datamap2 = dict(type='choropleth',
            colorscale = 'Portland',
            locations = medpriceXX.index,
            z = medpriceXX,
            locationmode = 'USA-states',
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Median Car Price per State"}
            ) 
layout2 = dict(title = 'Median Car Price per State',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )


# In[ ]:


choromap2 = go.Figure(data = [datamap2],layout = layout2)
iplot(choromap2)


# We find that WA and HI to be the highest in median car price. 

# That about completes it for my first EDA on Kaggle. 
# 
# This dataset was fun to visualize and while I wanted to attempt price prediction, I think other datasets would be more worthy.
