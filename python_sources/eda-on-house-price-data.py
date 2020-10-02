#!/usr/bin/env python
# coding: utf-8

#  # **This notebook demonstrates Exploratory Data Analysis on the House Price dataset**

# Here we have explored house price data set. Impact of different features like number of bedrooms, bathrooms, area of basement, living room, rooftop has been visualized. We have also visualized those patterns in various cities in USA. Finally we also explored the impact of time on house price from the data set.

# The majority of the plots and visualizations will be generated using data stored in *pandas* dataframes.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Importing necessary Libraries and Packages**

# In[ ]:


import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


# **Importing the data into Pandas Dataframe**

# In[ ]:


house_data=pd.read_csv("../input/housedata/data.csv")
house_data.head()


# In[ ]:


house_data.info()


# Here type of date is object. We need to convert it into date-time.

# In[ ]:


house_data["date"]=pd.to_datetime(house_data['date'], infer_datetime_format=True)
house_data.info()


# Now we will try to investigate the relationship of the features with price.

# In[ ]:


house_data.corr(method ='pearson') 


# Here we can see that Area of living room (sqft_living)  has the hisghest correlation value with the value. Visualizing with regression plot will make it more clear. 

# In[ ]:


neumaric=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_above','sqft_basement','yr_built','yr_renovated']

for i in neumaric:
    sns.lmplot(x = i, y ='price', data = house_data)


# From above visualizations we can see that year renovated, year built, condition features don't have significant impact on price. sqft_living, bathrooms, sqft_above have strong influence.

# However we haven't observe the influence of categorical variables on price.

# In[ ]:


cities=house_data['city'].unique()
cities


# In[ ]:


house_data['city'].value_counts()


# We are seeing that some cities don't have sufficient entries. So we are ignoring the cities with less that 100 entries.

# In[ ]:


cities=['Seattle','Renton','Bellevue','Bellevue','Redmond','Kirkland','Issaquah','Kent','Auburn','Sammamish','Federal Way','Shoreline','Woodinville']
house_data_filtered=house_data[house_data.city.isin(cities)]
house_data_filtered.shape


# In[ ]:


sns.set(rc={'figure.figsize':(20,10)})
sns.boxplot(x="city", y="price", data=house_data_filtered)


# Here we can see that most of the cities have similar influences on price. But different cities can have different influence on different features.

# There are some outliers in dataset. Now we will clean those outliers

# In[ ]:


q = house_data_filtered["price"].quantile(0.75)
house_data_final = house_data_filtered[(house_data_filtered["price"] < q)]
house_data_final.shape


# # **Impact of numer of bedrooms in different cities**

# In[ ]:


sns.set_style('whitegrid') 
plot=sns.lmplot(x ='bedrooms', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5) 


# Here we can see that influence of number of bedrooms on house price is not same in all cities. In Shoreline,Kent,Redmond,Auburn,Federal Way,Woodinville,Renton,Issaquah and Kirkland bedrooms have strong positive influence. Wheras in Sammamish,Seattle bedrooms have a little positive influence. In Bellevue there is no significant impact of bedrooms in housing prises

# # **Impact of number of bathrooms in different cities**

# In[ ]:


sns.set_style('whitegrid') 
plot=sns.lmplot(x ='bathrooms', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5) 


# It is clearly seen that number of bathrooms on house price is different in different cities. In Shoreline,Kent, Redmond,Federal Way,Kirkland,Woodinville,Renton, Issaquah number of bathrooms has strong positive influence on house price. In Seattle and Auburn have a little positive impact whereas in Bellevue and Sammamish number of bathrooms has negative impact on house price.
# 

# # **Impact of area of living room in different ciites**

# In[ ]:


sns.set_style('whitegrid') 
plot=sns.lmplot(x ='sqft_living', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5) 


# Area of living room has the highest correlation among the features with price. However we can see that in Bellevue and Sammamish area of living room has negative impact on house price. In rest of the cities area of living room has strong positive influence on house price. 

# # **Impact of floor space in different cities**

# In[ ]:


sns.set_style('whitegrid') 
plot=sns.lmplot(x ='sqft_lot', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5)


# Floor space has a very small correlation value. It doesn't have much impact on house price. Most of the values are limited within a certain range.

# # **Impact of area of basement in different cities**

# In[ ]:


sns.set_style('whitegrid') 
plot=sns.lmplot(x ='sqft_basement', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5)


# We can see that in cities like Auburn,Kent, Renton area of the basement has no impact on house price. In Sammamish,Issaquah area of basement has negative impact on houseprice.  In other cities area of basement has little positive influence on price. 

# # **Impact of area of rooftop in different cities**

# In[ ]:


sns.set_style('whitegrid') 
plot=sns.lmplot(x ='sqft_above', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5)


# From above visualization we can see that area of rooftop can be a good predictor of house price. In Bellevue area of rooftop has negative impact on house price. In rest of the cities area of rooftop has a strong positive influence.

# # **Effect of time**

# Now we will see the effect of time in house price. For this we need to select same type of houses and analyze them with time.

# In[ ]:


house_data_final.describe()


# In[ ]:


house_data_t=house_data_final[(house_data_final['bedrooms']<(3.235870+0.883879)) & (house_data_final['bedrooms']>(3.235870-0.883879))]
house_data_t=house_data_final[(house_data_final['bathrooms']<(1.959058+0.687286)) & (house_data_final['bathrooms']>(1.959058-0.687286))]
house_data_t=house_data_final[(house_data_final['sqft_living']<(1798.087319+673.396039)) & (house_data_final['sqft_living']>(1798.087319-673.396039))]
house_data_t=house_data_final[(house_data_final['sqft_above']<(1527.289855+619.584131)) & (house_data_final['sqft_above']>(1527.289855-619.584131))]
house_data_t.shape


# Now we have selected same type of houses. 

# In[ ]:


features=['date','price','bedrooms','bathrooms','sqft_living','sqft_above','city']
house_data_t=house_data_t[features]
house_data_t.head()


# We have highest number of entries in Seattle city. So we will visualize price in different time on same type of houses in Seattle city to see the pattern

# In[ ]:


p=['date','price']
seattle_data=house_data_t[house_data['city']=='Seattle']
seattle_data=seattle_data[p]
seattle_data.set_index('date',inplace=True)
seattle_data.head()
seattle_data.plot(grid=True)


# From above visualization we can see that in Seattle city there is no significant pattern with time in that span of time.

# Thank you. Constructive comments are highly appreciated.
