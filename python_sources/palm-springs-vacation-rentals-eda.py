#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## import data and investigate data##

# In[ ]:


df = pd.read_csv('../input/Datafiniti - Palm Springs Vacation Rentals.csv')
df.head()


# In[ ]:


print(df.shape)


# GENERAL ANALYSIS & DATA CLEANSING
# ---------------------------------
# Before we start delving into the visuals and stats for the data, let's run some general data analysis as well as statistics to get a feel for what we have. Let's start by checking if there exists any nulls in the dataframe by calling the method "isnull().any()" as such:
# 

# In[ ]:


df.isnull().any()


# Well well, let's start by getting rid of the nulls via the "dropna" call

# In[ ]:


df = df.dropna(axis=0)


# By calling the dataframe method "info()", we can discover is a numerical column contains strings. Therefore we convert that column type to string:

# In[ ]:


df.info()

