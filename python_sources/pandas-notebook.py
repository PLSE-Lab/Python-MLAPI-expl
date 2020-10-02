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


df = pd.read_csv('../input/parks.csv', index_col=['Park Code'])


# In[ ]:


df.head(3)


# In[ ]:


df.iloc[2]


# In[ ]:


df.loc['BADL']


# In[ ]:


df.loc[['BADL', 'ARCH', 'ACAD']]


# In[ ]:


df.iloc[[2,1,0]]


# In[ ]:


df[:3]


# In[ ]:


df[3:6]


# In[ ]:


df['State']


# In[ ]:


df['State'].head(3)


# In[ ]:


df.State.head(3)


# In[ ]:


df.Park Code


# In[ ]:


df.columns = [col.replace(' ', '_').lower() for col in df.columns]


# In[ ]:


print(df.columns)


# In[ ]:


df[['state', 'acres', 'park_name']][:3]


# In[ ]:


df.state.iloc[2]


# In[ ]:


df[df.state != 'UT']


# In[ ]:


df[(df.latitude > 60) | (df.acres > 10*6)]


# In[ ]:


df[df['park_name'].str.split().apply(lambda x: len(x) == 3)].head(3)


# In[ ]:




