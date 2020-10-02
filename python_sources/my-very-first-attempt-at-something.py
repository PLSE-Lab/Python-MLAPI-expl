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

input_file = os.listdir("../input")

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/master.csv")
df.head(10)


# In[ ]:


list(df)


# In[ ]:


print(df.info())


# The above shows that there are no null entries in any fields except HDI for year.

# In[ ]:


df['suicides_no'].hist(bins=50)


# In[ ]:


df['suicides/100k pop'].hist(bins=50)


# In[ ]:


df.boxplot(column='suicides/100k pop', by = 'year')


# In[ ]:


df.boxplot(column='suicides/100k pop', by = 'age')

