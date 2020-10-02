#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/winemag-data-130k-v2.csv')


# In[ ]:


data.info()


# In[ ]:


data.head()


# # Heat Map

# In[ ]:


heat = data.corr()
sns.heatmap(heat)
sns.heatmap(heat, annot = True)
plt.show()


# ## Total 'NaN' value for each features

# In[ ]:


data.isnull().sum()


# ## Total 'NaN' value for 'price'

# In[ ]:


data.price.isnull().sum()


# # Scatter Plot    
# ## Filtering  30<'price'<100  
# **Scatter plot 130 sample for 'price'**

# In[ ]:


data[(data['price'] >30) & (data['price'] <100)].sample(130).plot.scatter(x='price', y='points')
plt.show()


#  # Histogram
# 

# In[ ]:


data.price.plot.hist(bins=13,range=(0,130),figsize=(13,10))
plt.show()


# In[ ]:


data.columns


# ## First five country with highests wine review
# 

# In[ ]:


data.country.value_counts().head()


# # Wine Activity by country

# In[ ]:


plt.subplots(figsize=(13,7))
sns.countplot('country',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7),order=data['country'].value_counts().head(10).index)
plt.xticks(rotation=90)
plt.title('Number Of Wine Review By Country')
plt.show()


# ## Top 5 wine taster for  wine review

# In[ ]:


data.taster_name.value_counts().head()


# In[ ]:


plt.subplots(figsize=(17,7))
sns.countplot('taster_name',data=data,palette= 'plasma_r',edgecolor=sns.color_palette('dark',7),order=data['taster_name'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Wine Review By Wine Taster')
plt.show()


# # Cleaning Data

# In[ ]:


data1=data.drop(["Unnamed: 0"],axis=1)


# In[ ]:


data1.describe()


# # Box Plot and Violin Plot

# **Differences:**
# 
# Basically `violinplot` and `boxplot` shows the same data, but it is better to misinterpret and than the utilitarian `boxplot`

# In[ ]:


print(data['variety'].value_counts(dropna=False).head()) # dropna=True means: except nan values


# In[ ]:


df = data[data.variety.isin(data.variety.value_counts().head().index)]

sns.boxplot(x='variety',y='points',data=df)
plt.show()


# In[ ]:


sns.violinplot(x='variety',y='points',data=data[data.variety.isin(data.variety.value_counts()[:5].index)])
plt.show()


# ## Arrange Data

# We arrange data drop() and melt().
# 
# drop(): briefly provide remove the feature we do not want on data
# 
# melt(): Helps us examine specific features in data . Actually it can be confusing to explain. You will understand more cleary the next examples

# In[ ]:


new_data=data.drop(["Unnamed: 0"],axis=1).head()
new_data


# In[ ]:


melted_data=pd.melt(frame=new_data,id_vars='country',value_vars=['province','region_1'])
melted_data


# # Concatenating Data
# 
# We can concatenate two dataframe
# 
# 
# 

# **Adds Dataframes in row**

# In[ ]:


data1=data.head(3)
data2=data.tail(3)
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row


# **Adds Dataframes in column**

# In[ ]:


data1=data['country'].head()
data2=data['region_1'].head()
conc_data_col=pd.concat([data1,data2],axis=1)
conc_data_col

