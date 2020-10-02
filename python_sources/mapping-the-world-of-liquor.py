#!/usr/bin/env python
# coding: utf-8

# # Mapping the World of Liquor

# This notebook is based on the data set [Liquor for Days](https://www.kaggle.com/schmidtpalexander/worlds-largest-liquor-store-product-list) and made to display one way of analyzing the data. Please post your own data exploration if you want.

# ### Importing the needed packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Plotly imports
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Reading the data

# In[ ]:


df = pd.read_excel('../input/worlds-largest-liquor-store-product-list/Products-2020-jan-07-v1.xlsx')


# ### Checking the columns

# In[ ]:


df.columns


# ### Extracting the country names

# In[ ]:


countries = df['OriginCountryName']


# Checking unique values

# In[ ]:


countries.unique()


# Creating a data frame with the country and number of products

# In[ ]:


country_df = countries.value_counts().reset_index()
country_df


# Creating the data which will go into the choropleth map

# In[ ]:


data = dict(
        type = 'choropleth',
        autocolorscale=True,
        locations = country_df['index'],
        locationmode = "country names",
        z = country_df['OriginCountryName'],
        colorbar = {'title' : 'Number of products'},
      ) 


# Creating the dictionary for layout of the choropleth map

# In[ ]:


layout = dict(
    title = 'Countries with most products',
    geo = dict(
        showframe = False,
        projection = {'type':'equirectangular'}
    )
)


# Plotting the map

# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# ### Barplot of the countries with most products

# In[ ]:


sns.set(rc={'figure.figsize':(18,15)})
chart = sns.barplot(x='OriginCountryName',y='index', data=country_df)
plt.yticks(
    #rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small'  
)
plt.ylabel('Country')
plt.xlabel('Number of products')


# ### Barplot of the product groups

# In[ ]:


productGroup = df['ProductGroup']
productGroup_df = productGroup.value_counts().reset_index()
productGroup_df


# In[ ]:


sns.set(rc={'figure.figsize':(18,15)})
chart = sns.barplot(x='ProductGroup',y='index', data=productGroup_df)
plt.yticks(
    #rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small'  
)
plt.ylabel('Product Group')
plt.xlabel('Number of products')


# In[ ]:




