#!/usr/bin/env python
# coding: utf-8

# ## House Sales in Ontario 

# Information from the dataset...
# "This dataset includes the listing prices for the sale of properties (mostly houses) in Ontario. They are obtained for a short period of time in July 2016 and include the following fields: - Price in dollars - Address of the property - Latitude and Longitude of the address obtained by using Google Geocoding service - Area Name of the property obtained by using Google Geocoding service
# This dataset will provide a good starting point for analyzing the inflated housing market in Canada although it does not include time related information. Initially, it is intended to draw an enhanced interactive heatmap of the house prices for different neighborhoods (areas)
# However, if there is enough interest, there will be more information added as newer versions to this dataset. Some of those information will include more details on the property as well as time related information on the price (changes).
# This is a somehow related articles about the real estate prices in Ontario: http://www.canadianbusiness.com/blogs-and-comment/check-out-this-heat-map-of-toronto-real-estate-prices/
# I am also inspired by this dataset which was provided for King County https://www.kaggle.com/harlfoxem/housesalesprediction "
# Tasks and questions:
# Which region have more qtt of houses on sale?
# Which region have more valuable houses?

# ## Data Analysis

# In[1]:


#Importing Data Analysis Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Getting Data
df = pd.read_csv('../input/properties.csv')


# In[3]:


#Checking Dataset
df.head(5)


# In[4]:


df.columns


# In[5]:


#Changing some Column Names
df.rename(columns={'Unnamed: 0': 'ID', 'Price ($)': 'Price'}, inplace=True)
df.columns


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


#Eliminating houses with invalid lat and lng
df2 = df[(df['lat'] != -999 ) & (df['lng'] != -999)]
df2.shape


# In[9]:


#Eliminating houses with price <= 100000 and >= 10000000
df3 = df2[(df2['Price'] >= 100000) & (df2['Price'] <= 10000000)]
df3.shape


# In[10]:


df3.describe()


# ### Which area have more quantity of houses on sale?

# In[11]:


#Creating a TOP 20 Rank by area
qtHouses = pd.DataFrame(df3['AreaName'].value_counts())
qtHouses.sort_values(by='AreaName')
qtHouses[0:20].plot(kind='bar', title='Top 20 Areas by Number of Houses on Sale')


# Downtown is the area with most number of houses on sale

# ### Which area have the most expensive houses on sale ?

# In[12]:


#Boxplot 
topAreasNames = list(qtHouses[0:20].index)
area = df3[df3['AreaName'].isin(topAreasNames)]
#print(topAreasNames)
box_ax = area.boxplot(column='Price', by='AreaName', rot=90, grid=True)
box_ax.set_ylim(-1e5, 4e6)

plt.show()


# In[13]:


#Mean Price by Area(Top 20)
df3_dropped = df3.drop(['ID','Address','lat','lng'], axis=1)
meanPrices = df3_dropped.groupby(['AreaName']).mean().sort_values(by='Price', axis=0, ascending=False)
topMeanPrices = meanPrices[0:20] 
topMeanPrices.plot(kind='bar', title='Top 20 Mean Prices of Houses by Area')

