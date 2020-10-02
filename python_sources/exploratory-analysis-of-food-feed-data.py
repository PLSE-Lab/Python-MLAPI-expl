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
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


fao = pd.read_csv("../input/FAO.csv", encoding='latin1')


# In[ ]:


print("Lets look at a few rows of this data:\n")
fao.head()


# In[ ]:


print("Column names:\n", fao.columns)


# The dataset contains food and feed data from 1961 to 2013. As we can see from the column names Geocode data is also available for this data set but let's go to that later.
# Let's look at the data first.

# In[ ]:


print("Dataset contains {} rows and {} columns".format(fao.shape[0], fao.shape[1]))


# In[ ]:


print(fao.dtypes)


# In[ ]:


fao.Element.unique()


# The dataset has food and feed elements only as was mentioned in the dataset information.
# Let's look at the how the data is distributed among these two categories

# In[ ]:


sns.catplot('Element', data = fao, kind = 'count');


# In[ ]:


print("The feed and food data is available for {} countries listed below:\n".format(len(fao.Area.unique())))
fao.Area.unique()


# In[ ]:


# The count plot for 174 countries won't be to clear hence lets look at the values here
print("Area wise row counts:\n", fao.Area.value_counts())


# In[ ]:


# To get the top 20 food/feed production for Y2013
fao_top_20 = fao.nlargest(20, 'Y2013')
fao_top_20.head()


# In[ ]:


# Let's look at the largest producers in 2013
fao_top_20.Area.unique()


# In[ ]:


fao_top_20.Element.unique()


# In[ ]:


# Let's plot the the top 20 productions for food and feed for 2013
sns.catplot(x= 'Area', data = fao_top_20, kind ='count', height = 6, aspect = 1.5);


# In[ ]:


# Let's take a look  at the food items in the dataset
print(fao.Item.unique())
# Let's look for the top 5 producers for each item in 2013
fao_wheat = fao[fao.Item == 'Wheat and products']
print(fao_wheat.head())


# In[ ]:


fao_wheat_area = fao_wheat.groupby(['Area'])['Y2013'].sum()


# In[ ]:


fao_wheat_area.shape


# Further updates to follow..

# In[ ]:




