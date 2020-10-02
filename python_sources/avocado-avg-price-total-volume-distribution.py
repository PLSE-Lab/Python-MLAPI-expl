#!/usr/bin/env python
# coding: utf-8

# Hi ! I am Ajay. 
# 
# Thanks to Justin, I will use his  [Kernel](https://www.kaggle.com/neuromusic/avocado-prices-across-regions-and-seasons) as reference to explore following
# 
# *  Avocado Avg. price variation over Regions, Years and type(conventional/orgainic)
# *  Avocado Total Volume distribution over Regions, Years and type(conventional/orgainc)
# 
# First let's load this data
# 

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("../input/avocado.csv")
df.head()
# Any results you write to the current directory are saved as output.


# Time to explore "Total Volumes" per Region per type over the years
# 
# We will use Seaborn to generate these plots

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#sns.set_style()
#df.columns
sns.catplot(x='Total Volume',y='region',hue='year',col = 'type',data=df,kind = "point", height = 10,aspect = 0.5, join=False, sharex=False)


# Looks like Region column also contains 'TotalUS', we will drop this row and see regional variation.
# Also, let's sort regions in the order of Total Volume and re-plot.

# In[ ]:


df = df[df['region'] != 'TotalUS']
df.sort_values(by=['Total Volume'],inplace=True, ascending=False)
sns.catplot(x='Total Volume',y='region',hue='year',col = 'type',data=df,kind = "point", height = 10,aspect = 0.8, join=False,sharex=False)


# Great !
# Following are the observations from the plot:
# *  Total Volume for Conventional type is far higher than organic, this is true across all regions
# * Region data also contains major regions like West, South Central, California..
# *  Among major regions West and California far higher volumes.
# * Among Cities Great Lakes & Los Angeles have higher volumes while Syracuse has lower volumes.
# * Total volumes have increase over the years, this is true for all regions
# 
# 
# Now, We will explore same data for Average price

# In[ ]:


df.sort_values(by=['AveragePrice'],inplace=True, ascending=False)
sns.catplot(x='AveragePrice',y='region',hue='year',col = 'type',data=df,kind = "point", height = 10,aspect = 0.8, join=False,sharex=False)


# Above data shows  AveragePrice for Organic type is higher than Conventional type.
# 
# Sorting Average Price data with respect to Conventional category and year 2018 since Total Volume conventional category is higher ! 

# In[ ]:


sort_list_average_price = df[(df['type'] == 'conventional') & (df['year'] == 2018)].groupby('region')['AveragePrice'].mean().sort_values().index#sort_list_average_price = mask['region'].tolist()
sns.catplot(x='AveragePrice',y='region',hue='year',col = 'type',data=df,kind = "point", height = 10,aspect = 0.8, join=False,sharex=False,order=sort_list_average_price, col_order = ['conventional','organic'])


# Following are the observations for Average Price data:
# 
# *  Average Price for Avocado was highest in year 2017 and decreased during year 2018, this is true for most of the region
# 
# *  For Conventional Avocade type, Chicago has highest Average Price while Phoenix Tuxon has lowest.
# *  Average Price trend among Regions is not same for Conventional and Organic type Avocado.
# 
# 
# Please post your suggesstions in comment section.
# 
