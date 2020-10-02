#!/usr/bin/env python
# coding: utf-8

# # EDA for Avocado Prices Dataset
# 
# Here in this kernel, we will be exploring the Avocado Prices Dataset by using dedicated libraries and making visualizations.
# 
# Let's begin with importing the necessary libraries:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# And fetch the dataset into `df`.

# In[ ]:


df = pd.read_csv('../input/avocado.csv')


# To have a basic insight about the dataset, we have to use `df.info()` method.

# In[ ]:


df.info()


# There are 14 columns (attributes) in our dataset. Three of them (`Date`,`type` and `region`) are `object-type` (`string`), and the rest of them are non-numeric values.
# 
# Descriptions below give us a basic understanding of some of the columns in the dataset, borrowed from the [dataset overview](https://www.kaggle.com/neuromusic/avocado-prices/home):
# 
# * `Date` - The date of the observation
# * `AveragePrice` - the average price of a single avocado
# * `type` - conventional or organic
# * `year` - the year
# * `Region` - the city or region of the observation
# * `Total Volume` - Total number of avocados sold
# * `4046` - Total number of avocados with PLU 4046 sold
# * `4225` - Total number of avocados with PLU 4225 sold
# * `4770` - Total number of avocados with PLU 4770 sold

# Let's dive deeper into data by having statistical results by using `df.describe()` method.

# In[ ]:


df.describe()


# At the first glance, we can observe that dataset covers avocados observed from 2015 to 2018.

# Now it's time to take a look at data!

# In[ ]:


df.head()


# By examining a single row, we can say something further about the dataset. For example, avocados are packed up in bags and bags may have three types: `Small Bags`, `Large Bags` and `XLarge Bags`. `Total Bags` indicates the sum of bags used for packaging.

# We fetched the first 5 samples in the dataset by using `df.head()` method. Now let's get the *last* 15 samples by `df.tail(15)` to have a deeper insight.

# In[ ]:


df.tail(15)


# Strictly speaking, `Unnamed: 0` is an index feature which gets zero as `region` changes.
# 
# We can even make an intuitive comparison between two samples. Let's compare the samples with indexes `3` and `18240`. Fetching them into two distinct dataframes and then concatenating those frames will help us to see them sequently.

# In[ ]:


df_i3 = df.iloc[3]
#df_i3 = df_i3.to_frame()

df_i18240 = df.iloc[18240]
#df_i18240 = df_i18240.to_frame()

pd.concat([df_i3,df_i18240], axis = 1, ignore_index = True)


# We can come up with some conclusions here:  (sample on the left side is from Albany, the latter one is from West Texas, New Mexico)
# 
# * Avocados from Albany are conventional, where the ones from West Texas are organic.
# 
# * Avocados in West Texas are more expensive than in Albany, thanks to reason above.
# 
# * Avocados have higher volume in Albany.
# 
# * Both of the regions hate XLarge Bags.

# Now it is perfect time to examine the relationships between features, by using `df.corr()` method.

# In[ ]:


df.corr()


# A bunch of numbers scaled between 0 and 1, indicating the correlation between features. `seaborn` comes in handy here to visualize these correlations, via `sns.heatmap()` method.

# In[ ]:


f,axis = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.4, fmt= '.2f',ax=axis)
plt.show()


# As boxes goes lighter, correlation gets stronger. We can easily observe that `AveragePrice` has nothing to do with the type or number of bags.  But we can say, as `Total Volume` gets a higher value, the number of `Small Bags` increases faster than `Large Bags` and `XLarge Bags`.
# 
# It seems like examining the correlation between the volume of a single avocado (`Total Volume`) and the type of bag (`Small Bags`, etc) may take us somewhere.

# Let's visualize this correlation by using **scatter plot**.
# 
# Comparison between the feature `Total Volume` and features `Small Bags`, `Large Bags`, `XLarge Bags`, respectively:

# In[ ]:


df.plot(kind = 'scatter', x = 'Small Bags', y = 'Total Volume', color = 'magenta', alpha = '0.6')
plt.xlabel('Small Bags')
plt.ylabel('Total Volume')
plt.title("Comparison Between Total Volume and Small Bags")
plt.show()


# In[ ]:


df.plot(kind = 'scatter', x = 'Large Bags', y = 'Total Volume', color = 'red', alpha = '0.6')
plt.xlabel('Large Bags')
plt.ylabel('Total Volume')
plt.title("Comparison Between Total Volume and Large Bags")
plt.show()


# In[ ]:


df.plot(kind = 'scatter', x = 'XLarge Bags', y = 'Total Volume', color = 'blue', alpha = '0.6')
plt.xlabel('XLarge Bags')
plt.ylabel('Total Volume')
plt.title("Comparison Between Total Volume and XLarge Bags")
plt.show()


# ## Correlation Between Numeric and Non Numeric Feature

# By using `df.region.unique()` method, we will be listing the regions to choose our next location for data analysis.

# In[ ]:


df.region.unique()


# It is time to visit California! Let's examine samples from the region San Francisco.

# In[ ]:


sf = df['region'] == 'SanFrancisco'


# In[ ]:


df[sf].head()


# In[ ]:


df[sf].count()


# From the statistical analysis, we know the prices of the avocados vary between 0.44 and 3.25. But what is the effect of the type of avocados (conventional or organic) on the price? Filtering the dataset with arbitrary values may help us to get shallow observations.
# 
# Let's declare our arbitrary threshold as 2.0, so prices higher than this value will be claimed as expensive.

# In[ ]:


df_exp = df[sf & (df['AveragePrice'] > 2.0)]

df_exp.count()


# After filtering the dataset, we have 126 samples that are observed from San Francisco, and have average price higher than 2.0.
# 
# Now let's check how many of them are organic!

# In[ ]:


df_exp[df['type']=='organic'].count()


# Results are impressive! 122 of the expensive avocados (~%97) are organic. This observation may come handy for predicting the price of the avocados.
# 
# Now let's check the existence and the effect of outliers on data, via Visual EDA.

# ## Visual Exploratory Analysis

# In[ ]:


df.boxplot(column = 'AveragePrice', by = 'type', figsize = (10,10))
plt.show()


# Seems like we have a lot of outliers, especially on *organic* avocados. But saying 'a lot of outliers' is not enough, we need to have numeric results.

# In[ ]:


df.describe()


# **IQR** = Q3 - Q1 = 1.66 - 1.10 = **0.56 **
# 
# Q3 + (1.5).(IQR) = 1.66 + (1.5).(0.56) = **2.5** (upper threshold)
# 
# Q1 - (1.5).(IQR) = 1.10 - (1.5).(0.56) = **0.26** (lower threshold)
# 
# Avocados that have average price higher than 2.5 or lower than 0.26 are outliers.
# Let's check how many outliers do we have.

# In[ ]:


outliers = (df['AveragePrice'] > 2.5) | (df['AveragePrice'] < 0.26)
df[outliers].count()


# 203 in 18249, which means ~0.01 of the data consist of outliers.

# ## How prices change over time?
# 
# We can easily say time has a huge effect on prices (inflation, etc.). Let's take a look how it affects the average prices on avocados. We will use the technique **indexing pandas time series** to make it easy.
# 
# Dataset already has a column named `Date`, so all we have to do is converting this feature to proper type and implement it as an index.

# In[ ]:


df_ts = df.set_index('Date')
df_ts.head()
df_ts.tail()
df_ts.sort_index()
df_ts.tail()


# In[ ]:


#df_ts.resample("A").mean()


# In[ ]:


df.head(15)


# ## *...to be continued*
