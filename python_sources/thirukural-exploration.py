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
#change current directory to dataset directory
os.chdir("../input")

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('Thirukural.csv')


# In[ ]:


df_exp=pd.read_csv('Thirukural With Explanation.csv')


# In[ ]:


#explore
df.head()


# In[ ]:


#replacing tabs with spaces to read clearly
df['Verse']=df['Verse'].str.replace('\t',' ')
df.head()


# In[ ]:


#section name is Athigaram


# In[ ]:


#Capture how many sections in each chapter
df['Chapter Name'].value_counts()


# In[ ]:


df_exp.head(2)


# In[ ]:


# I dont see more difference between df and df_exp than an Explanation column.

#Adding the Explanation column to df.
df.loc[:,'Explanation']=df_exp.loc[:,'Explanation']


# In[ ]:


df.head(2)


# In[ ]:




