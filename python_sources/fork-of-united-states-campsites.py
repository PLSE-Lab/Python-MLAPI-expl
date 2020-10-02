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


df = pd.read_csv("../input/fed_campsites.csv")


# In[ ]:


print (df)


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.head(5)


# In[ ]:


df.tail(7)


# In[ ]:


df[df.AddressStateCode == 'CA']


# In[ ]:


states = df['AddressStateCode'].unique()
print (states)


# In[ ]:


agg = df.groupby(['AddressStateCode']).count()
print (agg.loc['NJ'],  agg.loc['NY'], agg.loc['CA'])


# In[ ]:




