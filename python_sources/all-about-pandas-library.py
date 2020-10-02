#!/usr/bin/env python
# coding: utf-8

# # Introduction to Pandas
# 
# This is a short intoduction to pandas, geared mainly for new user.
# 
# We will import as follows:

# In[ ]:


import pandas as pd
import numpy as np


# # Object creation
# 
# Creating a **Series** by passing a list of values, letting pandas create a default integer index:

# In[ ]:


s = pd.Series([1 , 3, 5, np.nan, 6, 8])
s


# Creating a **DataFrame** by passing a Numpy array, with a datatime index and labeled columns:

# In[ ]:


dates = pd.date_range('20130101' , periods=6)
dates


# In[ ]:


df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns= list('ABCD'))
df


# Creating A **DataFrame** by passing a dict of objects that can be converted to series-like:

# In[ ]:


df2 =pd.DataFrame({'A':1.,
              'B': pd.Timestamp('20130102'),
              'C': pd.Series(1, index=list(range(4)), dtype='float32'),
              'D': np.array([3] * 4, dtype='int32'),
              'E': pd.Categorical(["test", "train", "test", "train"]),
              'F': 'foo'})

df2


# The columns of the resulting DataFrame have different dtypes.

# In[ ]:


df2.dtypes


# # Viewing Data
# 
# Here is how to view the top and bottom rows of the frame:

# In[ ]:


df.head()


# In[ ]:


df.tail(3)


# Display the index, columns:

# In[ ]:


df.index


# In[ ]:


df.columns


# **DataFrame.to_numpy()** gives a Numpy representation of the underlying data. Note that this can be expensive operation when your DataFrame has columns with different data types, which comes down to a fundamental difference between pandas and Numpy: **Numpy arrays have one dtype for the entire array, while pandas DataFrames have one dtype per column.**
# 
# Whwn you call **DataFrame.to_numpy()**, pandas will find the Numpy dtype that can hold **all** of the dtypes in the DataFrame. This may end up being *object* , which requires  casting every value to a Python object.
# 
# For df, our DataFrame of all floating-point values.
# 
# **DataFrame.to_numpy()** is fast and doesn't require copying data.

# In[ ]:


df.to_numpy()


# For df2, the Dataframe with multiple dtypes, **DataFrame.to_numpy()** is relatively expensive.

# In[ ]:


df2.to_numpy()


# **Take note that DataFrame.to_numpy() does not include the index or column labels in the output.**

# **describe()** shows a quick statistic summary of your data:

# In[ ]:


df.describe()


# Transposing your data:

# In[ ]:


df.T


# Sorting by axis:

# In[ ]:


df.sort_index(axis = 1, ascending=False)


# Sorting by values:

# In[ ]:


df.sort_values(by='C')


# # Getting
# 
# Selecting a single column, which yeilds a Series, equivalent to df.A:

# In[ ]:


df['A']


# Selecting via [ ], which slices the rows.

# In[ ]:


df[0:3]


# In[ ]:


df['20130102':'20130104']


# # Selection by label
# 
# For getting a cross section using a label:

# In[ ]:


df.loc[dates[0]]


# Selecting on a multi-axis by label:

# In[ ]:


df.loc[:,['A','B']]


# Showing label slicing , both endpoints are included:

# In[ ]:


df.loc['20130102':'20130104',['A','B']]


# For getting a scalar value:

# In[ ]:


df.loc[dates[0], 'A']


# # Setting
# 
# Setting a new column automatically aligns the data by the indexes.

# In[ ]:


s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102' , periods=6))
s1


# In[ ]:


df['F'] = s1
df


# Setting by assigning with a NumPy array:

# In[ ]:


df.loc[:, 'D'] = np.array([5]*len(df))
df


# A where opertion with setting.

# In[ ]:


df2 = df.copy()

df2[df2 > 0] = -df2

df2


# # Missing Data
# 
# Pandas primarily uses the value np.nan to represent missing data. It is by default not included in computations.
# 
# Reindexing allows you to change/add/delete the index on a specified axis.This returns a copy of the data.

# In[ ]:


df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])


# In[ ]:


df1.loc[dates[0]:dates[1], 'E'] = 1
df1


# To drop any rows that have missing data.

# In[ ]:


df1.dropna(how='any')


# Filling missing data.

# In[ ]:


df1.fillna(value=5)


# To get boolean mask where values are nan.

# In[ ]:


pd.isna(df1)


# # Stats
# 
# Performing a descriptive statistic:

# In[ ]:


df.mean()


# Same operation on the other axis:

# In[ ]:


df.mean(1)


# # Apply
# Applying fuctions to the data:

# In[ ]:


df.apply(np.cumsum)


# In[ ]:


df.apply(lambda x: x.max() - x.min())


# # Concat
# 
# Pandas provides various facilities for easily combining together Series and DataFrame objects with various kinds of set logic for the indexes and relational algebra functionality in the case of join/ merge-type operations:
# 
# concatenating pandas objects together with **concat()**:

# In[ ]:


df = pd.DataFrame(np.random.randn(10, 4))
df


# In[ ]:


pieces = [df[:3], df[3:7], df[7:]]

pd.concat(pieces)


# # Grouping
# 
# By "Group by" we are referring to a process involving one or more of the following steps:
# 
# * **Splitting** the data into groups based on some criteria.
# * **Applying** a fuction to each group indeoendently.
# * **Combining** the results into a data structure.
# 

# In[ ]:


df = pd.DataFrame({'A': ['foo' , 'bar','foo' , 'bar','foo' , 'bar','foo' , 'foo'],
                   'B' :['one', 'one', 'two', 'three','one', 'two', 'three','one'],
                   'C' :np.random.randn(8),
                   'D' :np.random.randn(8)})

df


# Grouping and then applying the **sum()** fuction to the resuting groups.

# In[ ]:


df.groupby('A').sum()


# Grouping by multiple columns forms a hierarchical index, and again we can apply the sum fuction.

# In[ ]:


df.groupby(['A','B']).sum()


# # Pivot Table

# In[ ]:


df = pd.DataFrame({'A': ['foo' , 'bar','foo' , 'bar','foo' , 'bar','foo' , 'foo'],
                   'B': ['A','B','C','D'] * 2,
                   'C' :['one', 'one', 'two', 'three','one', 'two', 'three','one'],
                   'D' :np.random.randn(8),
                   'E' :np.random.randn(8)})
df


# In[ ]:


pd.pivot_table(df, values='D', index=['A','B'], columns=['C'])


# If you have reached till here, So i hope you liked my notebook.
# 
# If you learned anything new from this notebook then do give it a upvote.
# 
# I'm a rookie and any suggestion in the comment box is highly appreciated.
# 
# If you have any doubt reagrding any part of the notebook, feel free to comment your doubt in the comment box.
# 
# Thank you!

# 
