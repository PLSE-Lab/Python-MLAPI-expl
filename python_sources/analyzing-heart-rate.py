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


# Read the heart rate file & store it into a dataframe

# In[ ]:


import json
fileName='../input/HR2019-04-10.json'
with open(fileName) as json_file:  
            json_data = json.load(json_file)
df = pd.DataFrame(json_data['activities-heart-intraday']['dataset'])
df.head(5)


# Plot the whole day heart rate

# In[ ]:


df.plot()


# Lets analyse heart rate during sleep & find the min & max & the time when it occured

# In[ ]:


time = '07:00:00'
dfn=df.loc[df['time'] <= time]
print(dfn.loc[dfn['value'].idxmin()])
print(dfn.loc[dfn['value'].idxmax()])


# Lets plot a histogram

# In[ ]:


dfn.hist()


# Lets plot the heart rate during sleep

# In[ ]:


dfn['time'] = pd.to_datetime(dfn['time'])
dfn = dfn.set_index('time')
dfn.plot()


# As we cannot make out much from this. Lets aggregate it to every 30m mins & see the plot

# In[ ]:



df1 = dfn.resample('30min').mean()
df1.plot()


# We can observe that heart rate went down from 00:00 AM & reached a minimum at around 3:30 AM before beginning to raise again as the body prepares to wake up. A U shaped curve is considered a good sleep

# In[ ]:


dfa = df
dfa['time'] = pd.to_datetime(dfa['time'])
dfa = dfa.set_index('time')


# In[ ]:


df2 = dfa.resample('30min').mean()
df2.plot()

