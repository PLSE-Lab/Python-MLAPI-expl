#!/usr/bin/env python
# coding: utf-8

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


df = pd.read_csv('/kaggle/input/isotc213/ISO_TC 213.csv')


# In[ ]:


df.head()


# In[ ]:


df.drop(columns=['corrected_version'])


# In[ ]:


df['Status'].unique()


# In[ ]:


import matplotlib.pyplot as plt


# Grouping the products by their publication status

# In[ ]:


count_df = df.groupby('Status')['price_CF'].count().reset_index()


# In[ ]:


plt.bar(count_df['Status'],count_df['price_CF'])


# In[ ]:


n_pages_max = df['Number of pages'].max()
df['Number of pages'] /= n_pages_max


# In[ ]:


status_coded = df['Status'].factorize()
status_coded[1]


# In[ ]:


s = [200*(n+0.00000000000001) for n in df['Number of pages']]
colors = ['b', 'g', 'y' ,'r'] #corresponding to 4 status
c = [colors[n] for n in status_coded[0]]


# on the X-axis we have all the ISO products and on the Y-axis we have their corresponding 'price_CF'

# size of the bubble is the number of pages and the color of the bubble corresponds to the status.

# 0 - 'Withdrawn' --- Blue
# 1 - 'Published' --- Green
# 2 - 'Under development' --- Yellow
# 3 - 'Deleted' --- Red

# In[ ]:


plt.figure(figsize=[10,8])
plt.scatter(df.index, df['price_CF'], s=s, color= c)
plt.xticks([])
plt.xlabel('ISO Products')
plt.ylabel('price_CF')
plt.show()


# In[ ]:


task_cols = ['title', 'Number of pages', 'price_CF']


# In[ ]:


df['Number of pages'] *= n_pages_max


# In[ ]:


task_df = df[df['Status']=='Published'][task_cols]


# In[ ]:


task_df.head()


# In[ ]:


task_df['price_CF_to_npages_ratio'] = task_df['price_CF']/task_df['Number of pages']


# In[ ]:


mean_price_CF_to_npages_ratio = task_df['price_CF_to_npages_ratio'].mean()
mean_price_CF_to_npages_ratio


# list of top 5 ISO standard products with a published status and with price_CF to number of pages ratio greater than its mean

# In[ ]:


task_df[task_df['price_CF_to_npages_ratio']>mean_price_CF_to_npages_ratio][task_cols].sort_values(by=['price_CF'], ascending = False).head()


# In[ ]:




