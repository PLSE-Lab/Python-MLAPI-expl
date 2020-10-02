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


# **Objective**
# A very beginner level data analysis 

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


filename="../input/bd-food-rating.csv"


# In[ ]:


df=pd.read_csv(filename)


# In[ ]:


df.head(20)


# In[ ]:


df.dtypes


# In[ ]:


for column in df.columns.values.tolist():
    print(column)
    print (df[column].value_counts())
    print("")    


# This gives us an overall summary of the data frame where we can see that out of 57 restuarants inspected 37 were given A rating , 18 were given A+ rating and 2 were given B rating it also shows that maximum of the inspection were done in Dhaka paltan area, where 21 restuarants were inspected.

# In[ ]:


rating = {'A+': 2,'A': 1,'B':0}
df.bfsa_approve_status = [rating[item] for item in df.bfsa_approve_status] 
df


# For proper analysis of the data were changing the data type of the column bfsa_approve_status (which is of object type) to int type by assigning our own rating points **A+= 2 points, A= 1 points and B= 0 points**

# In[ ]:


df.loc[df['bfsa_approve_status'] == 2]


# Dataframe of the restuarants with A+ rating having 2 points

# In[ ]:


df.loc[df['bfsa_approve_status'] == 1]


# Dataframe of the restuarants with A rating having 1 points

# In[ ]:


df.loc[df['bfsa_approve_status'] == 0]


# Dataframe of the restuarants with B rating having 0 points

# In[ ]:




