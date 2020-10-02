#!/usr/bin/env python
# coding: utf-8

# In this Kernel I will analyse the Airbnb data in Singapore using some data visualisation tools.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


Singapore_data=pd.read_csv("../input/singapore-airbnb/listings.csv")


# The first step before starting any visualization is to get an understanding of the type of data vaiable, this includes understanding of number of data points, amount of columns, type of the data, missing values...

# In[ ]:


# visualisation of the top 5 rows

Singapore_data.head()


# In[ ]:


Singapore_data.info()


# This shows that the data frame contains 7907 entries and 15 columns (not including the id). 6 of the columns have categorical values, wheres the rest value numerical values. Most of the columns have values for all the data points, however for colums last_review and reviews_per_month there are only 5149 entries.

# In[ ]:


#Description of values in each of the numerical columns

Singapore_data.describe()


# The code below creats a heatmap, which gives information on variables that are highly correlated:

# In[ ]:


f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(Singapore_data.corr(),annot=True)#,linewidths=5,fmt='.1f',ax=ax)
#plt.show()


# The heat map only provides information on the correlation for columns with numerical values. This shows that the variables that have the highest correlation ( avaue close to 1 or -1, 1 if positively correlated. -1 if negatively correlated) are the i and  host_id columns and the number_of_reviews and Reviews_per_month. This was expected as these columns refer to very similar things.
# 
# In this case, I am going to examine further the variables that are affecting the price. From the heatmap above none of the numerical values seem to have a high correlation with the proce. Therefore, the next step will will to understand whether the price is affected by the categorical values (neighbourhood_group, neighbourhood, room_type). However, before that I want to see what the expected valiability of the price is.

# In[ ]:


Singapore_data.columns


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(a=Singapore_data['price'])#, kde=False)


# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Average price per night for different neighbourhood group")
sns.barplot(x=Singapore_data['neighbourhood_group'], y=Singapore_data['price'])


# In[ ]:


plt.figure(figsize=(15,10))
plt.title("Average price per night for different neighbourhoods")
sns.barplot(x=Singapore_data['price'], y=Singapore_data['neighbourhood'])


# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Average price per night for different room types")
sns.barplot(x=Singapore_data['room_type'], y=Singapore_data['price'])


# The 3 plots shown above show some important information. This shows that the price per night mainly depends on the type of the room and the location or neighbourhood of the accommocation.

# The next step could be to use this data to create a machine learning model based on these variables to predict the proce per night.
