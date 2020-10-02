#!/usr/bin/env python
# coding: utf-8

# 

# # Importing Data
# The dataset I am using is made available by Erik Bruin. We start by importing the data. Straight after, I like to get a quick preview by making histograms for all the features to get an initial impression of the dataset.

# In[ ]:


import pandas as pd
df = pd.read_csv("../input/airbnb-amsterdam/listings.csv")


# In[ ]:


df.hist(figsize= (16,9))


# # Data Cleansing
# Lets check for data quality. In this stage, we are mainly trying to answer the following questions:
# - Total number of records: How many records exist for each feature?
# - Data types: What are the data type of different features. How many numeric/categorical features are there in dataset?
# - Samples of records. What kind of information is in the feature column. Is the assigned data type appropriate for the features.
# - Empty records: Are there any? How do we adjust these values?

# In[ ]:


df.info()


# We cleaup the missing values as shown below. Our goal in this excercise to preserve as much information as possible and secondarily impute missing information with the right statistical measure.

# In[ ]:


df['reviews_per_month'].fillna(0,inplace=True)
del df['neighbourhood_group']
df['name'].fillna('Empty',inplace=True)
df['host_name'].fillna('Empty',inplace=True)
df[df.last_review.isnull()==False]
df['last_review'].fillna('Empty',inplace=True)


# In[ ]:


df.head()


# Now that the dataset is clean, lets see if we have any redundancy in the dataset.

# In[ ]:


import seaborn as sns
sns.heatmap(df.corr())


# We see that there is significant correlation between reviews_per_month and number_of_reviews. This is in confirming our understanding that these two variables are showing similar information. We purge this column from our dataset

# In[ ]:


del df['reviews_per_month']


# In[ ]:


df.info()


# We see that the dataset looks cleaner and ready for further analysis. We can start categorizing our categorical and numerical columns.

# In[ ]:


categorical_columns = [c for c in df.columns 
                       if df[c].dtype.name == 'object']
numerical_columns = [c for c in df.columns 
                     if df[c].dtype.name != 'object']
print('categorical: ',categorical_columns)
print('numerical: ',numerical_columns)


# In[ ]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')


# In[ ]:


figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
#cmap = sns.cubehelix_palette(as_cmap=True)
sns.scatterplot(x='number_of_reviews',y='price',data=df,alpha=0.5,                hue='room_type',                #palette=cmap,\
                legend="full")

plt.ylim(0, 2000)
#plt.xlim(-10, 400)


# We use violin plot to check the distribution of prices in different neighbourhoods. We can infer the following from this plot:

# In[ ]:


figure(num=None, figsize=(25, 12), dpi=80, facecolor='w', edgecolor='k')
#cmap = sns.cubehelix_palette(as_cmap=True)
sns.violinplot(y='price',x='neighbourhood',data=df[df.price < df['price'].quantile(.98)],)

plt.xticks(rotation=90)


# We can infer the following:
# - Bijlmer Centrum, Bijlmer Oost, Osdorp, Gaasperdam- Driemond have the lowest prices. Travellers can find cheaper rentals in these negihbourhoods.
# - Centrum Oost, Centrum West have relatively large spreads. The possibility of finding a good bargain in this region is therefore high.

# In[ ]:


sns.barplot(x='price', y = 'neighbourhood',data=df.groupby('neighbourhood').mean()['price'].unstack())


# In[ ]:


df.groupby('neighbourhood').mean()['price'].unstack()


# In[ ]:




