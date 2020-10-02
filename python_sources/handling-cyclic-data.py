#!/usr/bin/env python
# coding: utf-8

# # **Handling Cyclic Data **

# **What is Cyclic data??**
# 
# Data which has a unique set of values which repeat in a cycle are known as cyclic data. Time related features  are mainly cyclic in nature. For example, months of a year, day of a week, hours of time, minutes of time etc..These features have a set of values and all the observations will have a value from this set only. In many ML problems we encounter such features. But most of us ignore it. I suggest to handle them separately as handling such features properly have proved to help in improvement of the accuracy. Let us see how to handle them.  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
train.columns


# Here month and day are cyclic. Lets handle them
# 

# In[ ]:


df = train[['month','day']].copy()


# In[ ]:


print('Unique values of month:',df.month.unique())
print('Unique values of day:',df.day.unique())


# Lets first handle na. We will replace them with mode.

# In[ ]:


for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)


# In[ ]:


print('Unique values of month:',df.month.unique())
print('Unique values of day:',df.day.unique())


# So we see no nan is present now.
# 
# Also, we can understand that month is from 1-12
# And day is weekday from 1-7
# 
# These are cyclic data.
# 
# Let us handle them now.

# In[ ]:


import numpy as np

df['day_sin'] = np.sin(df.day*(2.*np.pi/7))
df['day_cos'] = np.cos(df.day*(2.*np.pi/7))
df['month_sin'] = np.sin((df.month-1)*(2.*np.pi/12))
df['month_cos'] = np.cos((df.month-1)*(2.*np.pi/12))


# In[ ]:


print(df.head(10))


# **The logic**
# 
#  We map each cyclical variable onto a circle such that the lowest value for that variable appears right next to the largest value. We compute the x- and y- component of that point using sin and cos trigonometric functions.
#  
#  For handling month we consider them from 0-11 and refer the below figure. 

# ![i](https://i.ibb.co/hF8yWhT/cyclic.png)

# Go on and implement this in your codes.
# 
# [Reference](http://blog.davidkaleko.com/feature-engineering-cyclical-features.html)
