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


# In[2]:


# to have autocompletion
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import pandas as pd
import numpy as np


# In[6]:


data = pd.read_csv("../input/FRvideos.csv")


# In[7]:


test = data["trending_date"].str.extractall("([0-9]{2}\.[0-9]{2}\.[0-9]{2})")

if test.shape != data["trending_date"].shape:
    print("something weird")
    setA = set(data["trending_date"].values)
    setB = set(test.values[:,0])
    diff = setA.difference(setB)
    for elem in diff:
        print( "problem with value ", elem, " at rows:", np.where( data["trending_date"] ==  elem))

