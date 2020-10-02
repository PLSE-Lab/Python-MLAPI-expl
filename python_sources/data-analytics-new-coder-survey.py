#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv")


# In[ ]:


data.columns.values


# Above is a list of features for every individual coder.  

# In[ ]:


# Looking out for missing values and removing unwanted features


# In[ ]:


def feature_summary(data):
    n_row=data.shape[0]
    features=pd.DataFrame()
    features_names=[]
    features_counts=[]
    features_missing=[]
    names=data.columns.values
    for i in names:
        features_names.append(i)
        features_counts.append(data[i].value_counts().count())
        features_missing.append(data[data[i].isnull()].shape[0])
    features['name']=features_names
    features['value counts']=features_counts
    features['missing']=features_missing
    features['percentage_missing']=features['missing']/n_row
    return (features)
        


# In[ ]:


feature_table=feature_summary(data)


# In[ ]:


feature_table


# From the above, we can see that only a few columns with lesser percentage of missing values matter a lot as they have information that might be useful for analysis. We can play with these select few columns first and then move on to the columns with missing values

# In[ ]:


useful_columns=feature_table[feature_table['percentage_missing']<0.50]
# eliminating features with more than 50% missing or null values
useful_columns.shape


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data_IsSoftwareDev = data[data['IsSoftwareDev']==1]
data_IsSoftwareDev['Age'].sort_values(axis=0,ascending=False).value_counts().plot(kind='bar',figsize=(12,4))
# this could be skewed based on number of people who took the survey
# let us divide the series  by the total number of value counts and then plot a bar plot


# In[ ]:




