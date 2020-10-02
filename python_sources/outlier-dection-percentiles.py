#!/usr/bin/env python
# coding: utf-8

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


# # Outlier are unusual datapoints which are different from the rest

# In[ ]:


df=pd.read_csv('/kaggle/input/air-bnb-ny-2019/AB_NYC_2019.csv')
df


# # Here i am interested in room type and price for detecting the outliers

# In[ ]:


df1=df[['room_type','price']]
df1


# In[ ]:


dj=df1.loc[df1['room_type'] == 'Entire home/apt']
dj


# # now we will se the price distribution for entire home

# In[ ]:


import seaborn as sns
x=dj['price']
sns.distplot(x,bins=30,kde=False)


# In[ ]:


dj.describe()


# # Now we will use percentile feature of pandas dataframe

# In[ ]:


dj['price'].quantile(0.95)
#THE OUTPUT IS THE THRESHOLD VALUE, WHICH MEANS 95% OF ALL THE PRICES ARE BELOW 450$


# In[ ]:


max_threshold=dj['price'].quantile(0.95)
#Similarly we will define minimum threshold value
min_threshold=dj['price'].quantile(0.05)
min_threshold
#THE OUTPUT IS THE MINIMUM THRESHOLD VALUE, WHICH MEANS 5% OF ALL THE PRICES ARE BELOW 78$


# # Now we will see the data without outliers

# In[ ]:


dj[(dj['price']<max_threshold)&(dj['price']>min_threshold)]


# # Now we will see the data with outliers

# In[ ]:


dj[(dj['price']>max_threshold)|(dj['price']<min_threshold)]


# In[ ]:


#so there are 2495 outliers in the dataset as per the percentile we have defined

