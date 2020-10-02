#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`.

# In[ ]:


import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There is 1 csv file in the current version of the dataset:

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Let's check the file: /kaggle/input/covid19-coronavirus-romania/covid-19RO.csv

# In[ ]:


df = pd.read_csv('/kaggle/input/covid19-coronavirus-romania/covid-19RO.csv', delimiter=',')
df.dataframeName = 'covid-19RO.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df.tail(5)


# The date column should be transformed to the proper format.

# In[ ]:


df['date'] =  pd.to_datetime(df['date'])

df.info()


# In[ ]:


plt.plot(df['cases'])


# In[ ]:


plt.plot(df['recovered'])


# In[ ]:


plt.plot(df['deaths'])


# In[ ]:


plt.plot(df['tests'])

