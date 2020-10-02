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


data = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv',encoding='ISO-8859-1')


# In[ ]:


data.head(5)


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


data['DISTRICT'].unique()


# In[ ]:


with open('/kaggle/input/crimes-in-boston/crime.csv') as file:
    print(file)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


top_crimes = data['OFFENSE_CODE_GROUP'].value_counts().head(5)


# In[ ]:


top_crimes


# In[ ]:


x = top_crimes.index


# In[ ]:


plt.bar(x,height=top_crimes)
plt.xticks(rotation=90)


# In[ ]:


list(data['DAY_OF_WEEK'].value_counts().index)


# In[ ]:


index = np.arange(7)
bar_width=0.95
plt.bar(data['DAY_OF_WEEK'].value_counts().index,height=data['DAY_OF_WEEK'].value_counts(),color = ['b','r','g','yellow','k', 'magenta', 'orange'])
plt.xlabel('Week days')
plt.ylabel('crime count')
#plt.xticks(rotation=90)
plt.legend()
plt.xticks(rotation=90)

