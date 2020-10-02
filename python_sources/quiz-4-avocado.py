#!/usr/bin/env python
# coding: utf-8

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


avocado = pd.read_csv('../input/avocados/Avocado.csv')
avocado


# If you want to print out the first n rows of the data set, use .head(n) method. On the other hand, if you want to display the last n rows, use tail(n) method. If there is no input for n, the method will display the first/last five rows of the data.

# In[ ]:


# display the first 5 rows
avocado.head()


# In[ ]:


# display the last 3 rows
avocado.tail(3)


# **There are three data attributes that you have to know:**
# 1. `.index`      : gives you information about the rows
# 2. `.columns`    : gives you the list of column labels
# 3. `.shape`      : gives you the shape of the data frame (rows x columns)

# In[ ]:


# prints information about the rows from the starting column (start = 0) to the end (stop=18249) consecutively.
avocado.index


# In[ ]:


# prints out a list of column labels in the data frame
avocado.columns


# In[ ]:


# prints out the shape of the data frame (rows x columns)
avocado.shape


# You could split, apply, and come the data frame. A good example of this is using the `.groupby()` function. Here are some rules that you have to follow if you want to use the function:
# * - You have to put a list of column labels that you want to group the data frame by
# * - You have to follow the groupby function with a column label that you want to display for each cell

# In[ ]:


# For each year, display the total sum of small_bags. Store the new data frame into a variable.
avo_data = avocado.groupby(['year']).Small_Bags.sum()
avocado.groupby(['year']).Small_Bags.sum()


# Notice how the function above has a .sum() method at the end. That is to specify what value to display in each cell. If you do not specify the values you want to display, the expression will only display the memory address.

# There are 5 other methods other than .sum() that you can use to display different values. They are:
# 1. .mean()
# 2. .median()
# 3. .std()
# 4. .size()
# 5. .value_counts()

# In[ ]:


# Figure out the mean number of small bags sold per day for each year
avocado.groupby(['year']).Small_Bags.mean()


# In[ ]:


# Figure out the median number of small bags sold for each year
avocado.groupby(['year']).Small_Bags.median()


# In[ ]:


# Figure out the standard deviation of small bags sold for each year
avocado.groupby(['year']).Small_Bags.std()


# In[ ]:


# .size() gives you the number of elements in the frame
# In this case, it will display the number of dates listed per year
avocado.groupby(['year']).Small_Bags.size()


# In[ ]:


# value_counts() will will display the number of counts of quantity of small_bags sold for each year
avocado.groupby(['year']).Small_Bags.value_counts()


# In[ ]:


# Let's try grouping by years and display the total numbers of small bags and large bags 
new_avo= avocado.groupby('year')['Small_Bags', 'Large_Bags'].sum()
new_avo


# There are two methods for selecting rows:
# - `.loc[]` is for index
# - `.iloc[]` is for row number
# 
# Try associating .loc[] with dictionaries while associng .iloc[] with lists

# In[ ]:


# .loc[] method with a input of a specific index that we are looking for will return associated information about that index
new_avo.loc[2015] 


# In[ ]:


# .iloc[row_index, column_index] method will return information when row_index and column index are specified.
new_avo.iloc[1,1]


# There are different sort methods. They are:
# * 1. `.sort_values()`
# 2. `.sort_index()`
# 3. `.max()`
# 4. `.min()`
# 5. `.idxmax()`
# 6. `.idxmin()`

# In[ ]:


# The sort_values will take in a column labeland order the data frame accordingly\
# The second input, ascending =True is optional. You can also make the input = False in order to reverse the order.
avo_sorted_small = avocado.sort_values('Small_Bags', ascending = True)
avo_sorted_small


# In[ ]:


# .sort_index() method will sort the data frame according to the indices
avo_sorted_small.sort_index()


# In[ ]:


# .max() method will give you the highest value for all the columns available
avo_sorted_small.max()

# Similarly, .min() will give you the lowest value for all the columns
# avo_sorted_small.min()


# Let's plot the data for further analysis.

# In[ ]:


# Group the number of small bags of avocadoes sold per year.
# Make a bar graph for it.
avocado.groupby('year')['Small_Bags'].sum().plot.bar()

