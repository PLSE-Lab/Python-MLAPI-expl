#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# 

# Training data
# The total size of the train data is almost 9 GB and we don't want to wait too long just for a first impression, let's load only some rows:

# In[4]:


train = pd.read_csv("../input/train.csv", nrows=10000000,
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.head(5)


# We can see two columns: Acoustic data and time_to_failure. The further is the seismic singal and the latter corresponds to the time until the laboratory earthquake takes place. Ok, personally I like to rename the columns as typing "acoustic" every time is likely for me to produce errors:

# In[5]:


train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
train.head(5)


# We can see that the quaketime of these first rows seems to be always the same. But is this really true?

# In[6]:


for n in range(5):
    print(train.quaketime.values[n])


# Aha! We can see that they are not the same and that pandas has rounded them off. And we can see that the time seems to decrease. Let's plot the time to get more familiar with this pattern:

# In[ ]:




