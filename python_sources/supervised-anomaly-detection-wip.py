#!/usr/bin/env python
# coding: utf-8

# # Supervised anomaly detection: Suitability evaluation

# We start by loading the data as `data_raw` and taking a look at the features present in the dataset.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


data_raw = pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv')
data_raw.head()


# In[ ]:


data_raw.info()


# 
# To identify if the dataset is fit for supervised anomaly detection tasks, we have to identify the ratio of backordered items. This ratio should not be higher then one percent.

# In[ ]:


print("Ratio of backordered items: {0:8.6f}".format(data_raw.went_on_backorder.value_counts()['Yes'] / sum(data_raw.went_on_backorder.value_counts())))


# **The data seems to be appropriate for supervised anomaly detection.**

# One thing, that has to be clarified, is the meaning of the "sku" column.

# In[ ]:


data_raw.sku.value_counts()


# We can see, that this column acts as a transaction ID and is therefor not relevant for our purposes.

# # Data Transformation

# Although the data is in a very good state, a few things need to be transformed.

# Lets begin with dropping the 'sku' column and transforming the values 'Yes' and 'No' into boolean values.

# In[ ]:


data = data_raw.copy()
data = data.drop(labels = 'sku', axis=1)
data.head()


# In[ ]:


data[data.columns[(data.dtypes == 'object')]] = data[data.columns[(data.dtypes == 'object')]]  == 'Yes'
data.head()


# Looking at the data.info() table, we can see that the last entry in the dataset is a bad row, containing only NaN values.

# In[ ]:


data = data.iloc[:-1]
data.tail()


# ### Dealing with NaN values
# The column 'lead_time' has some NaN values. We have multiple possibilities of dealing with them.

# In[ ]:


data.lead_time.isnull().sum() / data.shape[0]


# The number of NaN values in this column is at roughly 6%. It is therefore 10x higher than the amount of positive entries. If we were to drop these values, we'd potentially loose a lot of positive entries, which is to be avoided at all costs. The actual way of dealing with NaN values is now dependant on model performance and has to be decided on model by mode

# In[1]:


print("Ratio of backordered items: {0:8.6f}".format(data_raw.went_on_backorder.value_counts()['Yes'] / sum(data_raw.went_on_backorder.value_counts())))

