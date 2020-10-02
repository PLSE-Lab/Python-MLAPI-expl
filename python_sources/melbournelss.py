#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[ ]:


os.chdir('../input')


# In[ ]:


data= pd.read_csv('melbourne.csv') 


# In[ ]:


data


# In[ ]:


data.info() 


# In[ ]:


data.shape


# In[ ]:


data.median() 


# In[ ]:


data.describe() 


# In[ ]:


missing= data.isnull().sum() 
missing = missing[missing > 0]
missing.plot.bar() 


# In[ ]:


missing


# In[ ]:


#replacing missing values of Car,BuildingArea and YearBuilt with their mean values and CouncilArea with None
data['Car'].replace({np.nan:1.6},inplace= True) 
data['BuildingArea'].replace({np.nan:152},inplace= True)
data['YearBuilt'].replace({np.nan:1964},inplace= True)
data['CouncilArea'].replace({np.nan:'None'},inplace= True) 


# In[ ]:


data.head(3) 


# In[ ]:


qwe = data.corr()
plt.figure(figsize=(14,14))
sns.heatmap(qwe, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws= {'size': 12},
            xticklabels= qwe.columns.values,
            yticklabels= qwe.columns.values, linewidths= 0.5, linecolor= 'gold') 


# In[ ]:


num= [f for f in data.columns if data[f].dtype != 'object']
print("Numerical features are {}".format(len(num)))
cat= [f for f in data.columns if data[f].dtype == 'object']
print("Categorical features are {}".format(len(cat))) 


# In[ ]:


cat


# In[ ]:


num


# In[ ]:


X1= data[['Rooms','Distance','Postcode','Bedroom2','Price']] 
sns.set(style= 'ticks', palette= 'Dark2')
sns.pairplot(data, vars= X1)
plt.show() 


# In[ ]:


sns.lmplot(x= 'Rooms', y= 'Price', data= data) 


# In[ ]:


sns.lmplot(x= 'Distance', y= 'Price', data= data)


# In[ ]:


sns.lmplot(x= 'Postcode', y= 'Price', data= data)


# In[ ]:


sns.lmplot(x= 'Bedroom2', y= 'Price', data= data)


# In[ ]:


X2= data[['Bathroom','Car','Landsize','BuildingArea','YearBuilt','Price']] 
sns.set(style= 'ticks', palette= 'icefire')
sns.pairplot(data, vars= X2)
plt.show() 


# In[ ]:


sns.lmplot(x= 'Bathroom', y= 'Price', data= data)


# In[ ]:


sns.lmplot(x= 'Car', y= 'Price', data= data) 


# In[ ]:


sns.lmplot(x= 'Landsize', y= 'Price', data= data)


# In[ ]:


sns.lmplot(x= 'BuildingArea', y= 'Price', data= data)


# In[ ]:


sns.lmplot(x= 'YearBuilt', y= 'Price', data= data)


# In[ ]:


X3= data[['Lattitude','Longtitude','Propertycount','Price']]
sns.set(style= 'ticks', palette= 'mako')
sns.pairplot(data, vars= X3)
plt.show() 


# In[ ]:


sns.lmplot(x= 'Lattitude', y= 'Price', data= data)


# In[ ]:


sns.lmplot(x= 'Longtitude', y= 'Price', data= data)


# In[ ]:


sns.lmplot(x= 'Propertycount', y= 'Price', data= data)


# In[ ]:


sns.distplot(a= data['Propertycount'], kde= False)


# In[ ]:


sns.boxplot(x= 'Price', y= 'Regionname', data= data)


# In[ ]:


sns.stripplot(x= 'Price', y= 'Regionname', data= data)


# In[ ]:


sns.stripplot(x= 'Price', y= 'CouncilArea', data= data)


# In[ ]:


sns.stripplot(x= 'Price', y= 'Method', data= data)


# In[ ]:


sns.stripplot(x= 'Price', y= 'Type', data= data)


# The property rates in Melbourne depends on:
# Rooms: Number of rooms==> 2 to 4 rooms has higher price.
# Distance: Distance from CBD==> houses closer have more price.
# Bedroom2 : Scraped # of Bedrooms==> 2 to 5 bedrooms have higher price.
# Bathroom: Number of Bathrooms==> 2 to 4 bathrooms in a house have higher price.
# Car: Number of carspots==> space for 4 cars have higher demand and price.
# Landsize: The price is high for small sized plots.
# BuildingArea: The price is high for a small to medium sized building.
# Propertycount: The number of properties in each suburbs has made only a slight variation in price
# YearBuilt: Price is high for newer house.
# House in Southern Metropolitian have higher price.
# Method: S- property sold has higher price with  PI - property passed in and VB - vendor bid close to second highest.
# Type:  h - house,cottage,villa, semi,terrace==> are more preffered and costly.

# 

# In[ ]:




