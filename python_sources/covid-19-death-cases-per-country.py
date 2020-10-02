#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# task_2-COVID-19-death_cases_per_country_after_fifth_death-till_26_June.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/hackathon/task_2-COVID-19-death_cases_per_country_after_fifth_death-till_26_June.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'COVIV-19-deaths-after-fifth-case.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.head()


# In[ ]:


df2 = df1[[ 'deaths_per_million_10_days_after_fifth_death', 'deaths_per_million_20_days_after_fifth_death', 'deaths_per_million_30_days_after_fifth_death', 
           'deaths_per_million_40_days_after_fifth_death', 'deaths_per_million_50_days_after_fifth_death', 'deaths_per_million_60_days_after_fifth_death',
          'deaths_per_million_70_days_after_fifth_death', 'deaths_per_million_80_days_after_fifth_death', 'deaths_per_million_90_days_after_fifth_death',
          'deaths_per_million_100_days_after_fifth_death']]
df2


# In[ ]:


df3 = df2.transpose()

df3


# In[ ]:


# select few countries to compare the result
df4 = df3[[203, 204, 205, 206]]
df4


# In[ ]:


df4.plot.line()

