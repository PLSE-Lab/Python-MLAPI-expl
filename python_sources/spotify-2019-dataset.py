#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #matplot library
plt.style.use('fivethirtyeight')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Reading the file into our DataFrame**
# * We will use pd.read_csv for reading the csv file into our Dataframe

# In[ ]:


df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding = "ISO-8859-1")


# * We will check the content of the Dataframe and have a look at the data .We are checking the head and tails of the data.Head() will give us first 5 data included in the dataframe which is being analyzed .Similarly Tail() will give us last five data in our dataframe.
# 
# * ** We can use Head(10) or Tail(8) to change the default size of preview we want to see **

# In[ ]:


df.head()


# In[ ]:


df.tail()


# * We are using columns to check the column names of Data frame 

# In[ ]:


df.columns


# shape function is used for checking the data size i.e length and width of data

# In[ ]:


df.shape


# **We use info to check the datatypes of the predictor variables**

# In[ ]:


df.info()


# * From above analysis, we can see 3 object type predictor variables and 11 integer type predictor variables.Now we want to use statistical analysis to check how data is distributed and outliers in our data.

# # Statistical Analysis

# * We will use describe to get the count,mean,quartiles,maximum and minimum of the numerical predictor variables

# In[ ]:


df.describe()


# * We will have to use describe method on object type variables individually to get the information about the data.

# In[ ]:


df['Track.Name'].describe()


# * We are using unique method to get the unique values of Track

# In[ ]:


df['Track.Name'].unique()


# In[ ]:


df['Artist.Name'].describe()


# In[ ]:


df['Artist.Name'].unique()


# In[ ]:


df['Genre'].describe()


# In[ ]:


df['Genre'].unique()


# # Bar Graph of Popularity and Track Name

# In[ ]:


df.plot(y='Popularity',x= 'Track.Name',kind='bar',figsize=(25,6),legend =True,title="Popularity Vs Track Name",
        fontsize=18,stacked=True,color=['r', 'g', 'b', 'r', 'g', 'b', 'r'])
plt.ylabel('Popularity')
plt.xlabel('Track Name')
plt.show()


# # Box Plot

# In[ ]:


df.plot(y='Beats.Per.Minute', kind='box')


# In[ ]:


df.plot(kind='box',subplots=True, layout=(4,3),figsize=(35,20),color='r',fontsize=22,legend = True)


# In[ ]:


df[df['Popularity'] == df['Popularity'].max()] 


# In[ ]:


df[df['Popularity'] == df['Popularity'].min()] 


# # Top 5 Popular Songs of the Year

# In[ ]:


df['Popularity'].nlargest(5)


# In[ ]:


df[df['Popularity'] == 94]


# In[ ]:


df[df['Popularity'] == 93]


# In[ ]:


df[df['Popularity'] == 92]

