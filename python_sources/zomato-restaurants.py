#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas.

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir(r"../input/zomato-bangalore-restaurants/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv', header=0)

# data["location"][df['Banashankari']]


# In[ ]:


# data.head()

# data.columns
print("all addresses count is", data["address"].size)
# 560034

# type(data.location)

d = data.location
print("HSR addresses count is", d.value_counts()['HSR'])

print("% of addresses in HSR is", d.value_counts()['HSR']/data["address"].size)


# In[ ]:




