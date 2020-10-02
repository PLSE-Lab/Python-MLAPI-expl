#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#understanding pictureof the data
df=pd.read_csv("../input/master.csv")
df.head()
print("the number of rows are: "+str(df.shape[0]))
print("the number of column are: "+str(df.shape[1]))


# In[3]:


#finding unique countries entry
unique_country=df['country'].unique()
print(unique_country)
print("the number of unique countries in the dataset are: "+str(len(unique_country)))


# In[4]:


#finding correlation between data
correlation_between_attributesofData=sns.heatmap(df.corr(),annot=True)


# considering 0.5-0.8 to be medium correlation btw variables.Two sets of vars which are more correlated,first:(HDI for the year) and (gdp_per_capita), second: population and suicide_no.

# **will try to analyse the suicide number among male and female.**

# In[15]:


plt.figure(figsize=(10,5))
p = sns.barplot(x='age', y='suicides/100k pop', hue='sex', data=df)


# **suicide among male is more than female in every age group.**

# In[12]:


g=sns.lineplot(x='year',y='suicides/100k pop',hue='sex',data=df.groupby(['year','sex']).sum().reset_index()).set_title('graph')


# **suicide rate among male has been higher than female, but in the recent years the decrease in suicide rate in male is more than the female, as is evident from the slope of the graph after 2014.**

# In[11]:


p=sns.barplot(x='sex',y='suicides/100k pop',hue='age',data=df)


# **suicide in the age group of of 75+years is max in both the male and female.**

# 

# I am a beginner in data Analysis.this is my first kernel on kaggle.I have tried to do some very basic visualisation.Please privide your feedback.
