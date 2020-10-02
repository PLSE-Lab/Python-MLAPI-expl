#!/usr/bin/env python
# coding: utf-8

# # Purpose
# ### My purpose is to comparing room prices each region in Singapore

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set()


# # Raw Data

# In[ ]:


raw_data = pd.read_csv('../input/singapore-airbnb/listings.csv')
raw_data.info()


# # Pre-processing

# ### Does the minimum_nights and price of the rooms make sense to be rent?

# In[ ]:


sns.boxplot(x=raw_data['minimum_nights'])


# In[ ]:


sns.boxplot(x=raw_data['price'])


# ### Let's remove the outlier

# In[ ]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr #formula for find the outlier on the left side
    fence_high = q3+1.5*iqr #formula for find the outlier on the right side
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[ ]:


data_remove_1 = remove_outlier(raw_data,'minimum_nights')
data_remove_2 = remove_outlier(data_remove_1,'price')


# ### Let's see the Result

# In[ ]:


sns.distplot(data_remove_1['minimum_nights'], bins=10, kde=False)


# In[ ]:


sns.distplot(data_remove_2['price'], bins=10, kde=False)


# # Analyze

# ### The purpose is to compare avarage of price each region in Singapore

# In[ ]:


data = data_remove_2[['neighbourhood_group','price']]


# In[ ]:


data['neighbourhood_group'].value_counts()


# In[ ]:





# In[ ]:


data.groupby('neighbourhood_group')['price'].mean()


# In[ ]:


data['price'].mean()

