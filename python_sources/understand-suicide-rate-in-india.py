#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# we have a dataset of all type suicide happend in india since 2001,now we are going to analyze this data set according to our thinking that what actually are happening in our socity, why and people are attempting sucide :(
#  

# Load the data set using pandas library

# In[2]:


data = pd.read_csv('../input/Suicides in India 2001-2012.csv')


# Ok! csv file (data set) finally loaded to  variable "**data**"
# Now we will check first thier features 

# In[3]:


columns = data.columns
print(columns)
#'State', 'Year', 'Type_code', 'Type', 'Gender', 'Age_group', 'Total'


# Here, you can see columns value as
# State : Name of the indian state including union territory 
# Year  : Years 2001 to 2018
# Type_code : suicide code 
# Type : Name of sucide as dowry, family pressure etc
# Gender : Gender of person who had been suicide
# Age_group : range of age 
# Toatl : Number of sucide(Label)
# all the columns are very import we will analyize according these columns

# In[4]:


#shape of dataset
data.shape


# In[5]:


#extract state columns from data
state_col = data.State

#Get state name 
state = state_col.unique()
state


# In[6]:


#lenth of state array
len(state)


# Cool!! 38 state are here.
# How?
# 28 : All state
# 7 : UTS
# 1 : All india
# 1 : All State in one(28)
# 1 : All uts in one(7)
# Total = 28+7+1+1+1 = 38

# Same as state, lets discuss about **"Type"** and **"Type_code"** features or coulmns

# In[7]:


type_col      = data.Type
type_code_col = data.Type_code

#unique value of Type and Type_code

Type = type_col.unique()
Type_code = type_code_col.unique()

#print the lenth of Type and Type_code
print(len(Type))
print(len(Type_code))


# In[8]:


#first print type an type_code columns and see what actually this are
print(Type)


# In[9]:


print(Type_code)


#  we have three feature to analyze 
# * State
# * Type
# * Type_code
# * 
# Now I am going plot somthing with state and Type_code and then state and Type
# For ploting  I use matplotlib 

# In[10]:


import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:




