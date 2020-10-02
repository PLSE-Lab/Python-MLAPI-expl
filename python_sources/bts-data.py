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


get_ipython().system('wget https://s3.amazonaws.com/imcbucket/data/flights/2008.csv')


# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('2008.csv')


# In[ ]:


df.shape


# In[ ]:


print(df.columns)


# In[ ]:


df.head


# In[ ]:


df2 = df[['UniqueCarrier','ArrDelay']]


# In[ ]:


df2.head


# In[ ]:


df3 = df2.groupby(['UniqueCarrier']).mean()


# In[ ]:


df3.head


# In[ ]:


df3 = df3.sort_values(by=['ArrDelay'])
# df3 = df3.sort_values(by=['ArrDelay'], ascending=False)


# In[ ]:


df3.head()


# In[ ]:


df3.shape


# In[ ]:


import matplotlib.pyplot as plt
plt.figure()
df3.plot.bar()
# plt.axhline(0, color='k')

