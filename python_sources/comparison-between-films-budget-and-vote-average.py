#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/tmdb_5000_movies.csv')


# We have run the dataset for analyzing information

# In[ ]:


data.info()


# **There are too many columns which are waiting for correlation and comparison**
#      Now let me look at first 15

# In[ ]:


data.head(15)


# In[ ]:


data[['budget','vote_average','original_title']].max()


# In[ ]:


data[['budget','vote_average','original_title']].head()


# In[ ]:


data[['budget','vote_average','original_title']].tail()


# In[ ]:


#simple filtering
first_filter = data.budget > 200000000
second_filter = data.vote_average > 6.5
data[first_filter & second_filter]


# In[ ]:


#filtering based on other values
data.vote_average[data.budget> 200000000]


# In[ ]:


# adding new column
# data['....'] = ... + .... (columns)  or data['....'] = 
#


# In[ ]:


data.budget.mean()


# In[ ]:


data.vote_average.mean()


#  I think the most accurate comparison will be between BUDGET and VOTE AVERAGE 

# In[ ]:


data.describe()


# Before starting visualization lets check frequency of budget and vote average

# In[ ]:


plt.hist(data.budget, bins =30)
plt.xlabel('Budget')
plt.ylabel('Frequency')
plt.show()
plt.hist(data.vote_average, bins =30)
plt.xlabel('Vote Average')
plt.ylabel('Frequency')
plt.show()


# Now, i can visualize . Firstly, scatter style will be suitable for understanding relation between budget and vote avarage then line plot.

# In[ ]:


data.plot(kind = 'scatter' , x= 'budget' , y = 'vote_average' , grid = True , color = 'r', alpha=0.3)
plt.title('BUDGET x VOTE AVERAGE')
plt.show()


# In[ ]:


plt.plot(data.budget,data.vote_average,color='r',alpha = 0.4)
plt.title('BUDGET x VOTE AVERAGE')
plt.show()


# **CONCLUSION**
# The result i understood from graphics , there is no relationship between film budget and vote average . 
# But the view of film commentators may affect the result.
