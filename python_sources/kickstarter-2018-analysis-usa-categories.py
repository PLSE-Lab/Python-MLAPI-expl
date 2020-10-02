#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/ks-projects-201801.csv')


# In[ ]:


df.info()


# In[ ]:


df.head(5)


# In[ ]:


group_by_countries = df[['country','usd_goal_real']].groupby(['country']).count()['usd_goal_real'].reset_index(name='counts')


# In[ ]:


group_by_countries


# In[ ]:


plt.plot(group_by_countries['country'],group_by_countries['counts'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Title')


# In[ ]:


usa_data = df[df['country']=='US']


# In[ ]:


usa_data.head()


# In[ ]:


group_category_usa = usa_data.groupby('main_category').agg({'main_category':'count', 'usd_goal_real': 'mean'}).sort_values(by=['usd_goal_real'], ascending=False)


# In[ ]:


group_category_usa.rename(columns={'main_category':'main_category_count'}, inplace=True)


# In[ ]:


group_category_usa.reset_index(inplace=True)


# In[ ]:


group_category_usa.head(5)


# In[ ]:


fig = plt.figure(figsize=(9,3), dpi=200)

ax = fig.add_axes([0,0,1,1])
ax.plot(group_category_usa.head(10)['main_category'],group_category_usa.head(10)['usd_goal_real'])


# In[ ]:




