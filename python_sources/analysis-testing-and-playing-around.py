#!/usr/bin/env python
# coding: utf-8

# Hello there, I am just playing around here..

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **To get a brief overview of the data:**

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/BlackFriday.csv')
df.head(10)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# I would only like to get statistics for purchases, so..

# In[ ]:


df['Purchase'].describe()


# The above seen in a histogram and a box plot:

# In[ ]:


df['Purchase'].hist(bins=50)


# There are clearly some groupings as seen from the above histogram.

# In[ ]:


df.boxplot(column='Purchase')


# I would like to get a glimpse of data above 20,000 value in purchase just to get the exact value at which the last grouping is.

# In[ ]:


df[df['Purchase']>20000]['Purchase'].hist(bins=50)


# The split is occuring at above 22,000 mark.

# In[ ]:


df[df['Purchase']>22000]['Purchase'].hist(bins=50)


# I am suspecting that this high value purchases are due to product type however, I will still run box plots for age and gender wise purchases:

# In[ ]:


df[df['Purchase']>22000].boxplot(column = 'Purchase', by = 'Age')


# In[ ]:


df[df['Purchase']>22000].boxplot(column = 'Purchase', by = 'Gender')


# There is no gender and age relation to the high value purchases, now just to see by product:

# In[ ]:


df[df['Purchase']>22000].boxplot(column = 'Purchase', by = 'Product_ID')


# Ok, just to break it down into categories now:

# In[ ]:


df[df['Purchase']>22000].boxplot(column = 'Purchase', by = 'Product_Category_1')


# Visualising the rest of the data set by product category 1:

# In[ ]:


df[df['Purchase']<22000].boxplot(column = 'Purchase', by = 'Product_Category_1')


# Let me filter the whole data set by Product Category where product category is 10.

# In[ ]:


df[df['Product_Category_1']==10]['Purchase'].hist(bins=50)


# In[ ]:


df[df['Product_Category_1']==10].boxplot(column = 'Purchase', by = 'Product_ID')


# I am guessing that the differences is because of differences in units purchased? I would have loved to get that particular data as well..
# 
# That's it.. for now..
