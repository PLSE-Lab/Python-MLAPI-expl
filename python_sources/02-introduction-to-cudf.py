#!/usr/bin/env python
# coding: utf-8

# <a id="introduction"></a>
# ## Introduction to cuDF
# #### by Paul Hendricks
# #### modified by Beniel Thileepan 
# -------
# 
# This work is modified inorder to run in Kaggle with additional rapids dataset.
# 
# In this notebook, we will show how to work with cuDF DataFrames in RAPIDS.
# 
# **Table of Contents**
# 
# * [Introduction to cuDF](#introduction)
# * [Setup](#setup)
# * [cuDF Series Basics](#series)
# * [cuDF DataFrame Basics](#dataframes)
# * [Input/Output](#io)
# * [cuDF API](#cudfapi)
# * [Conclusion](#conclusion)

# <a id="setup"></a>
# ## Setup and install RAPIDs
# 
# 

# In[ ]:


get_ipython().system('nvidia-smi')


# Next, let's see what CUDA version we have:

# In[ ]:


get_ipython().system('nvcc --version')


# In[ ]:


import sys
get_ipython().system('rsync -ah --progress ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('rsync -ah --progress /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# <a id="series"></a>
# ## cuDF Series Basics
# 
# First, let's load the cuDF library.

# In[ ]:


import cudf; print('cuDF Version:', cudf.__version__)


# There are two main data structures in cuDF: a `Series` object and a `DataFrame` object. Multiple `Series` objects are used as columns for a `DataFrame`. We'll first explore the `Series` class and build upon that foundation to later introduce how to work with objects of type `DataFrame`.
# 
# We can create a `Series` object using the `cudf.Series` class.

# In[ ]:


column = cudf.Series([10, 11, 12, 13])
column


# We see from the output that `column` is an object of type `cudf.Series` and has 4 rows.
# 
# Another way to inspect a `Series` is to use the Python `print` statement.

# In[ ]:


print(column)


# We see that our `Series` object has four rows with values 10, 11, 12, and 13. We also see that the type of this data is `int64`. There are several ways to represent data using cuDF. The most common formats are `int8`, `int32`, `int64`, `float32`, and `float64`.
# 
# We also see a column of values on the left hand side with values 0, 1, 2, 3. These values represent the index of the `Series`. 

# In[ ]:


print(column.index)


# We can create a new column with a different index by using the `set_index` method.

# In[ ]:


new_column = column.set_index([5, 6, 7, 8]) 
print(new_column)


# Indexes are useful for operations like joins and groupbys.

# <a id="dataframes"></a>
# ## cuDF DataFrame Basics
# 
# As we showed in the previous tutorial, cuDF DataFrames are a tabular structure of data that reside on the GPU. We interface with these cuDF DataFrames in the same way we interface with Pandas DataFrames that reside on the CPU - with a few deviations.
# 
# In the next several sections, we'll show how to create and manipulate cuDF DataFrames. For more information on using cuDF DataFrames, check out the documentation: https://docs.rapids.ai/api/cudf/stable/

# #### Creating a cudf DataFrame using lists
# 
# There are several ways to create a cuDF DataFrame. The easiest of these is to instantiate an empty cuDF DataFrame and then use Python list objects or NumPy arrays to create columns. Below, we create an empty cuDF DataFrame.

# In[ ]:


df = cudf.DataFrame()
print(df)


# Next, we can create two columns named `key` and `value` by using the bracket notation with the cuDF DataFrame and storing either a list of Python values or a NumPy array into that column.

# In[ ]:


import numpy as np; print('NumPy Version:', np.__version__)


# here we create two columns named "key" and "value"
df['key'] = [0, 1, 2, 3, 4]
df['value'] = np.arange(10, 15)
print(df)


# #### Creating a cudf DataFrame using a list of tuples or a dictionary
# 
# Another way we can create a cuDF DataFrame is by providing a mapping of column names to column values, either via a list of tuples or by using a dictionary. In the below examples, we create a list of two-value tuples; the first value is the name of the column - for example, `id` or `timestamp` - and the second value is a list of Python objects or Numpy arrays. Note that we don't have to constrain the data stored in our cuDF DataFrames to common data types like integers or floats - we can use more exotic data types such as datetimes or strings. We'll investigate how such data types behave on the GPU a bit later.

# In[ ]:


from datetime import datetime, timedelta


ids = np.arange(5)
t0 = datetime.strptime('2018-10-07 12:00:00', '%Y-%m-%d %H:%M:%S')
timestamps = [(t0+ timedelta(seconds=x)) for x in range(5)]
timestamps_np = np.array(timestamps, dtype='datetime64')


# In[ ]:


df = cudf.DataFrame()
df['ids'] = ids
df['timestamp'] = timestamps_np
print(df)


# Alternatively, we can create a dictonary of key-value pairs, where each key in the dictionary represents a column name and each value associated with the key represents the values that belong in that column.

# In[ ]:


df = cudf.DataFrame({'id': ids, 'timestamp': timestamps_np})
print(df)


# #### Creating a cudf DataFrame from a Pandas DataFrame
# 
# Pandas DataFrames are a first class citizen within cuDF - this means that we can create a cuDF DataFrame from a Pandas DataFrame and vice versa.

# In[ ]:


import pandas as pd; print('Pandas Version:', pd.__version__)


pandas_df = pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'b': [0.0, 0.1, 0.2, None, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
print(pandas_df)


# We can use the `cudf.from_pandas` or `cudf.DataFrame.from_pandas` functions to create a cuDF DataFrame from a Pandas DataFrame.

# In[ ]:


df = cudf.from_pandas(pandas_df)
# df = cudf.DataFrame.from_pandas(pandas_df)  # alternative
print(df)


# #### Creating a cuDF DataFrame from cuDF Series
# 
# We can create a cuDF DataFrame from one or more cuDF Series objects by passing the Series objects in a dictionary mapping each Series object to a column name.

# In[ ]:


column1 = cudf.Series([1, 2, 3, 4])
column2 = cudf.Series([5, 6, 7, 8])
column3 = cudf.Series([9, 10, 11, 12])
df = cudf.DataFrame({'a': column1, 'b': column2, 'c': column3})
print(df)


# #### Inspecting a cuDF DataFrame
# 
# There are several ways to inspect a cuDF DataFrame. The first method is to enter the cuDF DataFrame directly into the REPL. This shows us information about the type of the object, and metadata such as the number of rows or columns.

# In[ ]:


df = cudf.DataFrame({'a': np.arange(0, 100), 'b': np.arange(100, 0, -1)})


# In[ ]:


df


# A second way to inspect a cuDF DataFrame is to wrap the object in a Python `print` function. This results in showing the rows and columns of the dataframe.

# In[ ]:


print(df)


# For very large dataframes, we often want to see the first couple rows. We can use the `head` method of a cuDF DataFrame to view the first N rows.

# In[ ]:


print(df.head())


# #### Columns
# 
# cuDF DataFrames store metadata such as information about columns or data types. We can access the columns of a cuDF DataFrame using the `.columns` attribute.

# In[ ]:


print(df.columns)


# We can modify the columns of a cuDF DataFrame by modifying the `columns` attribute. We can do this by setting that attribute equal to a list of strings representing the new columns.

# In[ ]:


df.columns = ['c', 'd']
print(df.columns)


# #### Data Types
# 
# We can also inspect the data types of the columns of a cuDF DataFrame using the `dtypes` attribute.

# In[ ]:


print(df.dtypes)


# We can modify the data types of the columns of a cuDF DataFrame by passing in a cuDF Series with a modified data type. Be warned that silent errors may be introduced from nonsensical type conversations - for example, changing a float to an integer or vice versa.

# In[ ]:


df['c'] = df['c'].astype(np.float32)
df['d'] = df['d'].astype(np.int32)
print(df.dtypes)


# #### Series
# 
# cuDF DataFrames are composed of rows and columns. Each column is represented using an object of type `Series`. For example, if we subset a cuDF DataFrame using just one column we will be returned an object of type `cudf.dataframe.series.Series`.

# In[ ]:


print(type(df['c']))
print(df['c'])


# #### Index
# 
# Like `Series` objects, each `DataFrame` has an index attribute.

# In[ ]:


df.index


# We can use the index values to subset the `DataFrame`.

# In[ ]:


print(df[df.index == 2])


# #### Converting a cudf DataFrame to a Pandas DataFrame
# 
# We can convert a cuDF DataFrame back to a Pandas DataFrame using the `to_pandas` method.

# In[ ]:


pandas_df = df.to_pandas()
print(type(pandas_df))


# #### Converting a cudf DataFrame to a NumPy Array
# 
# Often we want to work with NumPy arrays. We can convert a cuDF DataFrame to a NumPy array by first converting it to a Pandas DataFrame using the `to_pandas` method followed by accessing the `values` attribute of the Pandas DataFrame.

# In[ ]:


numpy_array = df.to_pandas().values
print(type(numpy_array))


# #### Converting a cudf DataFrame to Other Data Formats
# 
# We can also convert a cuDF DataFrame to other data formats. 
# 
# For more information, see the documentation: https://docs.rapids.ai/api/cudf/stable/api.html#dataframe

# <a id="io"></a>
# ## Input/Output
# 
# Before we process data and use it in machine learning models, we need to be able to load it into memory and write it after we're done using it. There are several ways to do this using cuDF.

# #### Writing and Loading CSV Files
# 
# At this time, there is no direct way to use to cuDF to write directly to CSV. However, we can conver the cuDF DataFrame to a Pandas DataFrame and then write it directly to a CSV.

# In[ ]:


df.to_pandas().to_csv('./dataset.csv', index=False)


# Perhaps one of the most common ways to create cuDF DataFrames is by loading a table that is stored as a file on disk. cuDF provides a lot of functionality for reading in a variety of different data formats. Below, we show how easy it is to read in a CSV file:

# In[ ]:


df = cudf.read_csv('./dataset.csv')
print(df)


# CSV files come in many flavors and cuDF tries to be as flexible as possible, mirroring the Pandas API wherever possible. For more information on possible parameters for working with files, see the cuDF IO documentation: 
# 
# https://rapidsai.github.io/projects/cudf/en/stable/api.html#cudf.io.csv.read_csv

# <a id="cudfapi"></a>
# ## cuDF API
# 
# The cuDF API is pleasantly simple and mirrors the Pandas API as closely as possible. In this section, we will explore the cuDF API and show how to perform common data manipulation operations.

# #### Selecting Rows or Columns
# 
# We can select rows from a cuDF DataFrame using slicing syntax. 

# In[ ]:


df = cudf.DataFrame({'a': np.arange(0, 100).astype(np.float32), 
                     'b': np.arange(100, 0, -1).astype(np.float32)})


# In[ ]:


print(df[0:5])


# There are several ways to select a column from a cuDF DataFrame.

# In[ ]:


print(df['a'])
# print(df.a)  # alternative


# We can also select multiple columns by passing in a list of column names.

# In[ ]:


print(df[['a', 'b']])


# We can select specific rows and columns using the slicing syntax as well as passing in a list of column names.

# In[ ]:


print(df.loc[0:5, ['a']])
# print(df.loc[0:5, ['a', 'b']])  # to select multiple columns, pass in multiple column names


# #### Defining New Columns
# 
# We often want to define new columns from existing columns.

# In[ ]:


df = cudf.DataFrame({'a': np.arange(0, 100).astype(np.float32), 
                     'b': np.arange(100, 0, -1).astype(np.float32), 
                     'c': np.arange(100, 200).astype(np.float32)})


# In[ ]:


df['d'] = np.arange(200, 300).astype(np.float32)

print(df)


# In[ ]:


data = np.arange(300, 400).astype(np.float32)
df.add_column('e', data)

print(df)


# #### Dropping Columns
# 
# Alternatively, we may want to remove columns from our `DataFrame`. We can do so using the `drop_column` method. Note that this method removes a column in-place - meaning that the `DataFrame` we act on will be modified.

# In[ ]:


df = cudf.DataFrame({'a': np.arange(0, 100).astype(np.float32), 
                     'b': np.arange(100, 0, -1).astype(np.float32), 
                     'c': np.arange(100, 200).astype(np.float32)})


# In[ ]:


df.drop_column('a')
print(df)


# If we want to remove a column without modifying the original DataFrame, we can use the `drop` method. This method will return a new DataFrame without that column (or columns).

# In[ ]:


df = cudf.DataFrame({'a': np.arange(0, 100).astype(np.float32), 
                     'b': np.arange(100, 0, -1).astype(np.float32), 
                     'c': np.arange(100, 200).astype(np.float32)})


# In[ ]:


new_df = df.drop('a')

print('Original DataFrame:')
print(df)
print(79 * '-')
print('New DataFrame:')
print(new_df)


# We can also pass in a list of column names to drop.

# In[ ]:


new_df = df.drop(['a', 'b'])

print('Original DataFrame:')
print(df)
print(79 * '-')
print('New DataFrame:')
print(new_df)


# #### Missing Data
# 
# Sometimes data is not as clean as we would like it - often there wrong values or values that are missing entirely. cuDF DataFrames can represent missing values using the Python `None` keyword.

# In[ ]:


df = cudf.DataFrame({'a': [0, None, 2, 3, 4, 5, 6, 7, 8, None, 10],
                     'b': [0.0, 0.1, 0.2, None, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                     'c': [0.0, 0.1, None, None, 0.4, 0.5, None, 0.7, 0.8, 0.9, 1.0]})
print(df)


# We can also fill in these missing values with another value using the `fillna` method. Both `Series` and `DataFrame` objects implement this method.

# In[ ]:


df['c'] = df['c'].fillna(999)
print(df)


# In[ ]:


new_df = df.fillna(-1)
print(new_df)


# #### Boolean Indexing
# 
# We previously saw how we can select certain rows from our dataset by using the bracket `[]` notation. However, we may want to select rows based on a certain criteria - this is called boolean indexing. We can combine the indexing notation with an array of boolean values to select only certain rows that meet this criteria.

# In[ ]:


df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                     'b': np.random.randint(2, size=100).astype(np.int32), 
                     'c': np.arange(0, 100).astype(np.int32), 
                     'd': np.arange(100, 0, -1).astype(np.int32)})


# In[ ]:


mask = df['a'] == 3
mask


# In[ ]:


df[mask]


# #### Sorting Data
# 
# Data is often not sorted before we start to work with it. Sorting data is is very useful for optimizing operations like joins and aggregations, especially when the data is distributed.
# 
# We can sort data in cuDF using the `sort_values` method and passing in which column we want to sort by. 

# In[ ]:


df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                     'b': np.random.randint(2, size=100).astype(np.int32), 
                     'c': np.arange(0, 100).astype(np.int32), 
                     'd': np.arange(100, 0, -1).astype(np.int32)})
print(df.head())


# In[ ]:


print(df.sort_values('d').head())


# We can also specify if the column we're sorting should be sorted in ascending or descending order by using the `ascending` argument and passing in `True` or `False`.

# In[ ]:


print(df.sort_values('c', ascending=False).head())


# We can sort by multiple columns by passing in a list of column names. 

# In[ ]:


print(df.sort_values(['a', 'b']).head())


# We can also specify which of those columns should be sorted in ascending or descending order by passing in a list of boolean values, where each boolean value maps to each column, respectively.

# In[ ]:


print('Sort with all columns specified descending:')
print(df.sort_values(['a', 'b'], ascending=False).head())
print(79 * '-')
print('Sort with both a descending and b ascending:')
print(df.sort_values(['a', 'b'], ascending=[False, True]).head())


# #### Statistical Operations
# 
# There are several statistical operations we can use to aggregate our data in meaningful ways. These can be applied to both `Series` and `DataFrame` objects.

# In[ ]:


df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                     'b': np.random.randint(2, size=100).astype(np.int32), 
                     'c': np.arange(0, 100).astype(np.int32), 
                     'd': np.arange(100, 0, -1).astype(np.int32)})


# In[ ]:


df['a'].sum()


# In[ ]:


print(df.sum())


# #### Applymap Operations
# 
# While cuDF allows us to define new columns in interesting ways, we often want to work with more complex functions. We can define a function and use the `applymap` method to apply this function to each value in a column in element-wise fashion. While the below example is simple, it can be very easily extended to more complex workflows.

# In[ ]:


df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                     'b': np.random.randint(2, size=100).astype(np.int32), 
                     'c': np.arange(0, 100).astype(np.int32), 
                     'd': np.arange(100, 0, -1).astype(np.int32)})


# In[ ]:


def add_ten_to_x(x):
    return x + 10

print(df['c'].applymap(add_ten_to_x))


# #### Histogramming
# 
# We can access the value counts of a column using the `value_counts` method. Note that this is typically used with columns representing discrete data i.e. integers, strings, categoricals, etc. We may not be as interested in the value counts of numerical data e.g. how often the value 2.1 appears. The results of the `value_counts` method can be used with Python plotting libraries like Matplotlib or Seaborn to generate visualizations such as histograms.

# In[ ]:


df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                     'b': np.random.randint(2, size=100).astype(np.int32), 
                     'c': np.arange(0, 100).astype(np.int32), 
                     'd': np.arange(100, 0, -1).astype(np.int32)})


# In[ ]:


result = df['a'].value_counts()
print(result)


# #### Concatenations
# 
# In everyday data science, we typically work with multiple sources of data and wish to combine these data into a single more meaningful representation. These operations are often called concatenations and joins. We can concatenate two or more dataframes together row-wise or column-wise by passing in a list of the dataframes to be concatenated into the `cudf.concat` function and specifying the axis along which to concatenate these dataframes.
# 
# If we want to concatenate the dataframes row-wise, we can specify `axis=0`. To concatenate column-wise, we can specify `axis=1`.

# In[ ]:


df1 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                      'b': np.random.randint(2, size=100).astype(np.int32), 
                      'c': np.arange(0, 100).astype(np.int32), 
                      'd': np.arange(100, 0, -1).astype(np.int32)})
df2 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                      'b': np.random.randint(2, size=100).astype(np.int32), 
                      'c': np.arange(0, 100).astype(np.int32), 
                      'd': np.arange(100, 0, -1).astype(np.int32)})


# In[ ]:


df = cudf.concat([df1, df2], axis=0)
df


# In[ ]:


df1 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                      'b': np.random.randint(2, size=100).astype(np.int32), 
                      'c': np.arange(0, 100).astype(np.int32), 
                      'd': np.arange(100, 0, -1).astype(np.int32)})
df2 = cudf.DataFrame({'e': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                      'f': np.random.randint(2, size=100).astype(np.int32), 
                      'g': np.arange(0, 100).astype(np.int32), 
                      'h': np.arange(100, 0, -1).astype(np.int32)})


# In[ ]:


df = cudf.concat([df1, df2], axis=1)
df


# #### Joins / Merges
# 
# Multiple dataframes can be joined together using a single (or multiple) column(s). There are two syntaxes for performing joins:
# 
# * One can use the `DataFrame.merge` method and pass in another dataframe to join, or
# * One can use the `cudf.merge` function and pass in which dataframes to join.
# 
# Both syntaxes can also be passed a list of column names to an additional keyword argument `on` - this will specify which columns the dataframes should be joined on. If this keyword is not specified, cuDF will by default join using column names that appear in both dataframes.

# In[ ]:


df1 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                      'b': np.random.randint(2, size=100).astype(np.int32), 
                      'c': np.arange(0, 100).astype(np.int32), 
                      'd': np.arange(100, 0, -1).astype(np.int32)})
df2 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                      'b': np.random.randint(2, size=100).astype(np.int32), 
                      'e': np.arange(0, 100).astype(np.int32), 
                      'f': np.arange(100, 0, -1).astype(np.int32)})


# In[ ]:


df = df1.merge(df2)
print(df.head())


# In[ ]:


df = df1.merge(df2, on=['a'])
print(df.head())


# In[ ]:


df = df1.merge(df2, on=['a', 'b'])
print(df.head())


# In[ ]:


df = cudf.merge(df1, df2)
print(df.head())


# In[ ]:


df = cudf.merge(df1, df2, on=['a'])
print(df.head())


# In[ ]:


df = cudf.merge(df1, df2, on=['a', 'b'])
print(df.head())


# #### Groupbys
# 
# A useful operation when working with datasets is to group the data using a specific key and aggregate the values mapping to those keys. For example, we might want to aggregate multiple temperature measurements taken during a day from a specific sensor and average those measurements to find avergage daily temperature at a specific geolocation.
# 
# cuDF allows us to perform such an operation using the `groupby` method. This will create an object of type `cudf.groupby.groupby.Groupby` that we can operate on using aggregation functions such as `sum`, `var`, or complex aggregation functions defined by the user.
# 
# We can also specify multiple columns to group on by passing a list of column names to the `groupby` method.

# In[ ]:


df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 
                     'b': np.random.randint(2, size=100).astype(np.int32), 
                     'c': np.arange(0, 100).astype(np.int32), 
                     'd': np.arange(100, 0, -1).astype(np.int32)})
print(df.head())


# In[ ]:


grouped_df = df.groupby('a')
print(grouped_df)


# In[ ]:


aggregation = grouped_df.sum()
print(aggregation)


# In[ ]:


aggregation = df.groupby(['a', 'b']).sum()
print(aggregation)


# #### One Hot Encoding
# 
# Data scientists often work with discrete data such as integers or categories. However, this data can be represented using a One Hote Encoding format.
# 
# cuDF allows us to convert these discrete datas to a One Hot Encoding format using the `one_hot_encoding` method. We can pass this method the column name to convert, a prefix with which to prepend to each newly created column, and the categories of data to create new columns for. We can pass in all the categories in the discrete data or a subset - cuDF will flexibly handle both and only create new columns for the categories specified.

# In[ ]:


categories = [0, 1, 2, 3]
df = cudf.DataFrame({'a': np.repeat(categories, 25).astype(np.int32), 
                     'b': np.arange(0, 100).astype(np.int32), 
                     'c': np.arange(100, 0, -1).astype(np.int32)})
print(df.head())


# In[ ]:


result = df.one_hot_encoding('a', prefix='a_', cats=categories)
print(result.head())
print(result.tail())


# In[ ]:


result = df.one_hot_encoding('a', prefix='a_', cats=[0, 1, 2])
print(result.head())
print(result.tail())


# <a id="conclusion"></a>
# ## Conclusion
# 
# In this notebook, we showed how to work with cuDF DataFrames in RAPIDS.
# 
# To learn more about RAPIDS, be sure to check out: 
# 
# * [Open Source Website](http://rapids.ai)
# * [GitHub](https://github.com/rapidsai/)
# * [Press Release](https://nvidianews.nvidia.com/news/nvidia-introduces-rapids-open-source-gpu-acceleration-platform-for-large-scale-data-analytics-and-machine-learning)
# * [NVIDIA Blog](https://blogs.nvidia.com/blog/2018/10/10/rapids-data-science-open-source-community/)
# * [Developer Blog](https://devblogs.nvidia.com/gpu-accelerated-analytics-rapids/)
# * [NVIDIA Data Science Webpage](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/)
