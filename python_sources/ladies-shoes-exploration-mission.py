#!/usr/bin/env python
# coding: utf-8

# **Some basic wowmen's shoes data exploration**

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


wshoes = pd.read_csv('/kaggle/input/womens-shoes-prices/7210_1.csv', low_memory=False)

wshoes.head(5)


# In[ ]:


# Get rid of the empty and not so useful columns

wshoes.drop(['asins', 'categories', 'count', 'descriptions', 'dimension', 'ean', 'features', 'flavors', 'imageURLs',
             'isbn', 'keys', 'manufacturer', 'manufacturerNumber', 'merchants', 'prices.availability', 'prices.color',
             'prices.condition', 'prices.count', 'prices.dateAdded', 'prices.dateSeen', 'prices.flavor', 'prices.merchant', 'prices.offer',
             'prices.returnPolicy', 'prices.size', 'prices.shipping', 'prices.source', 'prices.sourceURLs', 'prices.warranty', 'quantities', 'reviews', 'sizes',
             'skus', 'upc', 'vin', 'websiteIDs', 'weight'], axis=1, inplace=True)
wshoes = wshoes.iloc[:,:-4]


# In[ ]:


# In this data set there are a small number of items quoted in Canadian dollars. 
# We love Canada, but to keep things simple here, let's only keep the items priced in US dollars

wshoes = wshoes[wshoes['prices.currency'] == 'USD']


# In[ ]:


# We are given the maximum and the minimum prices, which we can use to get mean price

wshoes['PRICE'] = (wshoes['prices.amountMax'] + wshoes['prices.amountMax']) / 2
wshoes['PRICE'].sort_values().head(5)


# In[ ]:


wshoes['PRICE'].sort_values().tail(5)


# There seem to be some outliers. I doubt there are women's shoes prices 0.01 dollar. Also, I do not want
# my wife to notice that there are shoes for 4198.99 dollar. So, let's pretend they never existed and get rid of both the extremes by eliminating 0.5% from each end.

# In[ ]:


wshoes = wshoes[wshoes['PRICE'] > wshoes['PRICE'].quantile(0.005)]
wshoes = wshoes[wshoes['PRICE'] < wshoes['PRICE'].quantile(0.995)]


# In[ ]:


# data visualizatiom
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.hist(wshoes['PRICE'], bins=50)
plt.title('Price Frequency Distribution')
plt.xlabel('Prices')
plt.ylabel('Occurences')
plt.show()


# In[ ]:


# It looks like the price data has a very long right tail. Let's get rid of all the entries greater than 300 dollars

wshoes = wshoes[wshoes['PRICE'] < 250]

plt.hist(wshoes['PRICE'], bins=25)
plt.title('Price Frequency Distribution')
plt.xlabel('Prices')
plt.ylabel('Occurences')
plt.show()


# In[ ]:


# Top 10 best selling brands

top_brands = wshoes['brand'].value_counts().head(10)
print(top_brands)


# In[ ]:


# Average price for the top 10 best selling brands

top_brands_avg = wshoes[wshoes['brand'].isin(top_brands.index)].groupby('brand')['PRICE'].mean()

print(top_brands_avg)


# In[ ]:


# Price dispersion of the top 10 best selliong brands

top_brands_dispersion = wshoes[wshoes['brand'].isin(top_brands.index)].groupby('brand')['PRICE'].std()

top_brands_dispersion


# In[ ]:


# It is better to compare coefficient of variation, instead of simple standard deviation, since the means vary a lot across brands

top_brands_coeff_var = top_brands_dispersion / top_brands_avg

top_brands_coeff_var


# In[ ]:


# Most popular colors

popular_colors = wshoes['colors'].value_counts().head(10)

popular_colors


# In[ ]:


plt.figure(figsize=(10, 6))
plt.bar(popular_colors.index, popular_colors)
plt.title('Colour Popularity')
plt.show()


# In[ ]:


# Average price for each of the 10 most popular shoe colors

avg_price_by_color = wshoes[wshoes['colors'].isin(popular_colors.index)].groupby('colors')['PRICE'].mean()

avg_price_by_color


# In[ ]:


# Looks like brown shoes costs 72 dollars on average, while black-tan costs about half the price. Very interesting!


# In[ ]:


plt.figure(figsize=(10, 6))
plt.bar(avg_price_by_color.index, avg_price_by_color)
plt.title('Average Shoes Price By Colour')
plt.show()


# In[ ]:


# I'll be back to do some more exploratory analysis on this dataset :)

