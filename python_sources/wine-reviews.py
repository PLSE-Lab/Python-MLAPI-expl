#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sb
import matplotlib.pyplot as plt


# In[ ]:


wine_df =pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")


# In[ ]:


wine_df.head()


# In[ ]:


wine_df.info()


# In[ ]:


wine_df.tail()


# In[ ]:


wine_df.iloc[2, 2]


# In[ ]:


wine_df.drop('Unnamed: 0', axis = 1, inplace = True)


# In[ ]:


wine_df.shape


# In[ ]:


wine_df.duplicated().sum()


# In[ ]:


wine_df.describe()


# In[ ]:


wine_df['country'].nunique()


# In[ ]:


wine_df['country'].value_counts()


# In[ ]:





# In[ ]:


plt.figure(figsize=[8,8])
sb.countplot(y='country', data= wine_df)


# Highest number of wines found in the U.S.

# In[ ]:


wine_df.head()


# What are the main features of interest?

# The main features of interest are country, wine variety, price, points. I would like to look at the relationship between price and points and variety. Possibly also looking at the winery to see if there is any relationship there.

# In[ ]:


#Looking at the distribution of price
binsize= 20
bins= np.arange(0, wine_df['price'].max() +10, binsize)
plt.figure(figsize=[8, 5])
plt.hist(data = wine_df, x = 'price', bins = bins)
plt.title("Wine price")
plt.xlabel('Wine price distribution')
plt.xlim(0, 1000)
plt.show()


# The price has a skewed distribution. The range being 4 dollars to above 3000 dollars. 

# In[ ]:


#Look at the points distribution
sb.distplot(wine_df['points'], bins = 28);


# In[ ]:


#Look at the bivariate distributions of price and points
wine_df['variety'].nunique()


# In[ ]:


sb.regplot(data = wine_df, x= 'points', y ='price', fit_reg= False,
          x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/5});


# There is not a strong relation between price and points; the price does not vary much with points although there are 
# few high price values associated with higher points which is abve $1000

# Let us look at the wines in the U.S. Italy and France

# In[ ]:


some_countries=wine_df.loc[(wine_df['country']== 'US')  |(wine_df['country']=='Italy')| (wine_df['country']=='France')]


# In[ ]:


#Let us compare the price between these countries
#plt.figure(figsize=[10,10])
fig, ax =plt.subplots(ncols=2, figsize=[14,8])
sb.boxplot(data=some_countries, x= 'country', y ='price', ax= ax[0]);
 
sb.boxplot(data=some_countries, x= 'country', y ='points', ax= ax[1]);


# The max price is seen for French wines; the distribution for price is equally varied for the three countries. France 
# shows high number of outliers for the price. As for the points, US shows the higest range for the points with more varied 
# distribution.

# What are those varieties for which the price is above 1000 dollars?

# In[ ]:


some_countries.describe()


# In[ ]:


sb.regplot(data = some_countries, x= 'points', y ='price', fit_reg= False,
           scatter_kws = {'alpha' : 1/5});


# In[ ]:


variety =some_countries.loc[some_countries['price'] >=1000]


# In[ ]:


plt.figure(figsize=[12, 10])
sb.violinplot(x='variety',y = 'price', data = variety);


# So the pricey wines above 1000 dollars are Bordeau-style Red blends and even more pricier are the Pino Noir

# In[ ]:





# In[ ]:


some_countries.corr()


# Moderately positive correlation exists between price and points for countries US, Italy and France.

# In[ ]:


sb.boxplot(data = variety, x= 'province', y= 'price')


# Wines coming from the  province of Burgundy have a high median price of 2000 dollars or a little more.

# In[ ]:


for_heat= variety[['winery','province', 'price']]


# In[ ]:


for_heat_2= variety[['province', 'price','variety']]


# In[ ]:





# In[ ]:


#The wines priced above 1000 dollars come most  from Burgundy province and the winery :Dominane Du Comte
plt.figure(figsize=[10,10])
sb.countplot(data= for_heat, y= 'winery', hue='province');


# In[ ]:


#for_heat_set.winery =for_heat_set.winery.astype('category')


# In[ ]:


for_heat.info()


# df= pd.pivot_table(data=sb.load_dataset("for_heat_2"),
#                 index= "province",
#                 values="price", 
#                 columns = "variety")
# #Getting a urlERROR -- Was trying to do a heatmap

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:


#below_1000.describe()


# In[ ]:





# In[ ]:





# In[ ]:




