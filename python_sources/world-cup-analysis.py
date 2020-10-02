#!/usr/bin/env python
# coding: utf-8

# **WORLD CUP ANALYSIS**

# ![](https://iasbh.tmgrup.com.tr/1c123a/0/0/0/0/0/0?u=https://isbh.tmgrup.com.tr/sb/album/2018/06/09/dunya-kupasi-tarihine-gecen-enler-1528547678820.jpg&mw=752&mh=700)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualizations
import seaborn as sns # data visualizations

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Any results you write to the current directory are saved as output.


# In[ ]:


# reading the dataset

data=pd.read_csv('../input/fifa-world-cup/WorldCups.csv')
data.shape


# In[ ]:


# checking the head of the data

data.head(10)


# In[ ]:


# checking the tail of the data

data.tail(10)


# In[ ]:


# information the data

data.info()


# In[ ]:


# describing the data

data.describe()


# In[ ]:


# checking if there are any null values

data.isnull().sum()


# In[ ]:


# data features

data.columns


# In[ ]:


#data types

data.dtypes


# In[ ]:


# Year feature and GoalsScored feature concat

data1= data['Year']
data2= data['GoalsScored']
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col


# **Year of least goals : 1930, 1934**
# 
# **Year of most goals : 1998, 2014**

# In[ ]:


# scatter plot
# goal frequency by year
data.plot(kind = "scatter",x="Year",y = "GoalsScored",color='green')
plt.show()


# In[ ]:


# world cup the most goal 

data['GoalsScored'].max()


# In[ ]:


# world cup the least goal

data['GoalsScored'].min()


# In[ ]:


# repeated goal counts
# ristogram plot

data.GoalsScored.plot(kind = 'hist',color='green',bins = 50,figsize = (10,10))
plt.show()


# In[ ]:


print(data['Winner'].value_counts(dropna =False)) 


# In[ ]:


print(data['Runners-Up'].value_counts(dropna =False))


# In[ ]:


print(data['Third'].value_counts(dropna =False))


# In[ ]:


print(data['Country'].value_counts(dropna =False))


# **Most winning country : Brazil**
# 
# **Most runner-up countries : Argentina, Germany FR, Netherlands**
# 
# **Most third country : Germany**
# 
# **Nost organizing countries : Germany, Brazil, France, Italy, Mexico**
