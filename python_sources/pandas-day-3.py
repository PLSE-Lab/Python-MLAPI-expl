#!/usr/bin/env python
# coding: utf-8

# # Pandas tutorial : Day 3
# 
# Selecting, Slicing and Filtering data in a Pandas DataFrame
# 
# * [Selecting](#1)
#  1. [Select rows and columns using labels](#2)
#    * [To select a single column](#3)
#    * [To select multiple columns](#4)
#    * [Select a row by it's label](#5)
#    * [Select multiple row by it's label](#6)
#    * [Accessing values by row label and column name](#7)
#    * [Accessing values from multiple columns of same row](#8)
#    * [Accessing values from multiple rows but same columns](#9)
#    * [Accessing values from multiple rows and multiple columns](#10)
#  2. [Select by index position](#11)
#    * [Select a row by index location](#12)
#    * [Select a column by index location](#13)
#    * [Select data at specified row and column location](#14)
#    * [Select multiple rows and columns](#15)
#  3. [Selecting top n largest values of given column](#16)
#  4. [Selecting top n samllest values of given column](#17)
#  5. [Selecting random sample from the dataset](#18)
#  6. [Conditional selection of columns](#19)
# * [Slicing](#20)
#  1. [Slicing rows and columns using labels.](#21)
#    * [Slice row by label](#22)
#    * [Slice columns by label](#23)
#    * [Slice row and columns by label](#24)
#  2. [Slicing rows and columns by position.](#25)
#    * [To slice rows by index position](#26)
#    * [To slice columns by index position](#27)
#    * [To slice row and columns by index position](#28)
# * [Subsetting by boolean conditions](#29)
#  1. [Select rows based on column value](#30)
#    * [To select all rows whose column contain the specified value(s)](#31)
#    * [Rows that match multiple column conditions](#32)
#    * [Select rows whose column DOES NOT contain specified values](#33)
#  2. [Select columns based on row value](#34)
#  3. [Subsetting using filter method](#35)
#  
#  
# Let's gets started!
#  
# [Data for daily news for stock market prediction](https://www.kaggle.com/aaron7sun/stocknews)

# In[ ]:


# import library
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# import data
df = pd.read_csv('/kaggle/input/stocknews/upload_DJIA_table.csv')


# In[ ]:


df.head()


# # Selecting<a id='1'></a>

# ## 1. Select rows and columns using labels<a id='2'></a>
# You can select rows and columns in a Pandas DataFrame by using their corresponding labels.

# ### 1.1 To select a single column<a id='3'></a>
# Syntax 1 : `df.loc[:, 'column_name']`
# 
# Syntax 2 : `df['column_name']`
# 
# Syntax 3 : `df.column_name`

# In[ ]:


df.loc[:, 'Open']


# In[ ]:


df['Open']


# In[ ]:


df.Open


# ### 1.2 To select multiple columns<a id='4'></a>
# Syntax 1 : `df.loc[:, ['column1', 'column2', ...]]`
# 
# Syntax 2 : `df[['column1', 'column2', ...]]`

# In[ ]:


df.loc[:, ['Open', 'Close']]


# In[ ]:


df[['Open', 'Close']]


# ### 1.3 Select a row by it's label<a id='5'></a>
# Syntax : `df.loc[row_label]`

# In[ ]:


df.loc[0]


# ### 1.4 Select multiple row by it's label<a id='6'></a>
# Syntax : `df.loc[[row_label1, row_label2, ...]]`

# In[ ]:


df.loc[[0,1,10]]


# ### 1.5 Accessing values by row label and column name<a id='7'></a>
# Syntax : `df.loc[row_label, 'column_name']`

# In[ ]:


df.loc[0, 'Open']


# ### 1.6 Accessing values from multiple columns of same row<a id='8'></a>
# Syntax : `df.loc[row_label, ['column_name1', 'column_name2']]`

# In[ ]:


df.loc[1, ['Open', 'Close']]


# ### 1.7 Accessing values from multiple rows but same columns<a id='9'></a>
# Syntax : `df.loc[[row_label1, row_label2], 'column_name']`

# In[ ]:


df.loc[[0, 1], ['Open']]


# ### 1.8 Accessing values from multiple rows and multiple columns<a id='10'></a>
# Syntax : `df.loc[[row_label1, row_label2, ...], ['column_name1, column_name2, ...']]`

# In[ ]:


df.loc[[0, 1], ['Open', 'Close']]


# ## 2. Select by index position<a id='11'></a>
# You can select data from a Pandas DataFrame by its location. Note, Pandas indexing starts from zero.

# ### 2.1 Select a row by index location<a id='12'></a>
# Syntax : `df.iloc[index]`

# In[ ]:


df.iloc[0]


# ### 2.2 Select a column by index location<a id='13'></a>
# Syntax : `df.iloc[:, index]`

# In[ ]:


df.iloc[:, 5]


# ### 2.3 Select data at specified row and column location<a id='14'></a>
# Syntax : `df.iloc[row_index, column_index]`

# In[ ]:


df.iloc[0, 0]


# ### 2.4 Select multiple rows and columns<a id='15'></a>
# Syntax : `df.iloc[[row_index1, row_index2, ...], [column_index1, column_index2, ...]]`

# In[ ]:


df.iloc[[0, 1, 3], [0, 1]]


# ## 3. Selecting top n largest values of given column<a id='16'></a>
# Syntax : `df.nlargest(n, 'column_name')`

# In[ ]:


df.nlargest(3,'Open') 


# ## 4. Selecting top n smallest values of given column<a id='17'></a>
# Syntax : `df.nsmallest(n, 'column_name')`

# In[ ]:


df.nsmallest(3,'Open') 


# ## 5. Selecting random sample from the dataset<a id='18'></a>
# Syntax 1 : `df.sample(n)` 
# 
# Syntax 2 : `df.sample(frac = n)` 

# In[ ]:


df.sample(3)


# In[ ]:


df.sample(frac = 0.3)


# ## 6. Conditional selection of columns<a id='19'></a>
# Syntax 1 : `df[df.column_name < value]`
# 
# Syntax 2 : `df[df.column_name > value]`
# 
# Syntax 3 : `df[df.column_name == value]`
# 
# Syntax 4 : `df[df.column_name <= value]`
# 
# Syntax 5 : `df[df.column_name >= value]`

# In[ ]:


df[df.Open >= 18281.949219]


# # **Slicing**<a id='20'></a>
# Slicing in Python is a feature that enables accessing parts of sequences like strings, tuples, and lists. You can also use them to modify or delete the items of mutable sequences such as lists. Slices can also be applied on third-party objects like NumPy arrays, as well as Pandas series and data frames.
# 
# Slicing enables writing clean, concise, and readable code.

# ## 1. Slicing rows and columns using labels<a id='21'></a>
# You can select a range of rows or columns using labels or by position. To slice by labels you use **loc** attribute of the DataFrame.

# ### 1.1 Slice row by label<a id='22'></a>
# Syntax : `df.loc[starting_row_label : ending_row_label, :]`

# In[ ]:


df.loc[1:5, :]


# ### 1.2 Slice columns by label<a id='23'></a>
# Syntax : `df.loc[:, 'starting_column_name' : 'ending_column_name']`

# In[ ]:


df.loc[:, 'Open' : 'Close']


# ### 1.3 Slice row and columns by label<a id='24'></a>
# Syntax : `df.loc[starting_row_label : ending_row_label, 'starting_column_name' : 'ending_column_name']`

# In[ ]:


df.loc[1:3, 'Open' : 'Close']


# ## 2. Slicing rows and columns by position<a id='25'></a>
# To slice a Pandas dataframe by position use the iloc attribute. Remember index starts from 0 to (number of rows/columns - 1).

# ### 2.1 To slice rows by index position<a id='26'></a>
# Syntax : `df.iloc[starting_row_index : ending_row_index, :]`

# In[ ]:


df.iloc[0:2, :]


# ### 2.2 To slice columns by index position<a id='27'></a>
# Syntax : `df.iloc[:, starting_column_index : ending_column_index]`

# In[ ]:


df.iloc[:, 1:4]


# ### 2.3 To slice row and columns by index position<a id='28'></a>
# Syntax 1 : `df.iloc[starting_row_index : ending_row_index, starting_column_index : ending_column_index]`
# 
# Syntax 2 : `df.iloc[:starting_row_index, :ending_column_index]`

# In[ ]:


df.iloc[0:2, 0:2]


# In[ ]:


df.iloc[:2, :2]


# ## Subsetting by boolean conditions<a id='29'></a>
# You can use boolean conditions to obtain a subset of the data from dataframe.

# ## 1. Select rows based on column value<a id='30'></a>

# ### 1.1 To select all rows whose column contain the specified value(s)<a id='31'></a>
# Syntax 1 : `df.column_name == value`
# 
# Syntax 2 : `df.loc[df.column_name == value]`
# 
# Syntax 3 : `df[df.Open == value]`
# 
# Syntax 4 : `df[df.column_name.isin([value1, value2, ...])]`

# In[ ]:


df.Open == 17355.210938


# In[ ]:


df[df.Open == 17355.210938]


# In[ ]:


df.loc[df.Open == 17355.210938]


# In[ ]:


df[df.Open.isin([17355.210938])]


# ### 1.2 Rows that match multiple column conditions<a id='32'></a>
# Syntax : `df[(df.column_name == value) | (df.column_name == value)]`

# In[ ]:


df[(df.Open == 17355.210938) | (df.Close == 17949.369141)]


# ### 1.3 Select rows whose column DOES NOT contain specified values<a id='33'></a>
# Syntax : `df[~df.column_name.isin([value])]`

# In[ ]:


# row 0 and 2 contains these values so it is not in output
df[~df.Open.isin([17924.240234, 17456.019531])]


# ## 2. Select columns based on row value<a id='34'></a>
# To select columns where rows contain the specified value.
# 
# Syntax : `df.loc[:, df.isin([value]).any()]`

# In[ ]:


df.loc[:, df.isin([17456.019531]).any()]


# ## 3. Subsetting using filter method<a id='35'></a>
# Subsets can be created using the filter method like below.
# 
# Method 1 : `df.filter(items=['column_name1', 'column_name2'])`
# 
# Method 2 : `df.filter(like='row_index/label', axis=0)`
# 
# Method 3 : `df.filter(regex='[^column_letter]')`
# 
# Method 4 : `df[(df['column_name'] > value) & (df['column_name'] > value)]`

# In[ ]:


df.filter(items=['Open', 'Close'])


# In[ ]:


df.filter(like="2", axis=0)


# In[ ]:


df.filter(regex="[^OpenCloseHigh]")


# In[ ]:


df[(df['Open'] > 18281.949219) & (df['Date'] > '2015-05-20')]

