#!/usr/bin/env python
# coding: utf-8

# **Introduction to Pandas**
# 
# In this section of the course we will learn how to use pandas for data analysis. You can think of pandas as an extremely powerful version of Excel, with a lot more features. In this section of the course, you should go through the notebooks in this order:
# 
# * Introduction to Pandas
# * Series
# * DataFrames
# * Missing Data
# * GroupBy
# * Merging,Joining,and Concatenating
# * Operations
# * Data Input and Output

# **Series**
# 
# The first main data type we will learn about for pandas is the Series data type. Let's import Pandas and explore the Series object.
# 
# A Series is very similar to a NumPy array (in fact it is built on top of the NumPy array object). What differentiates the NumPy array from a Series, is that a Series can have axis labels, meaning it can be indexed by a label, instead of just a number location. It also doesn't need to hold numeric data, it can hold any arbitrary Python Object.
# 
# Let's explore this concept through some examples:

# In[ ]:


import numpy as np 
import pandas as pd


# **Creating a Series**
# 
# You can convert a list,numpy array, or dictionary to a Series:

# In[ ]:


labels=['a', 'b','c']
my_data=[1,2,3]
arr=np.array(my_data)
d={'a': 1,'b':2, 'c':3}


# **Using Lists**

# In[ ]:


pd.Series(my_data)


# In[ ]:


pd.Series(data=my_data,index=labels)


# **NumPy Arrays**

# In[ ]:


pd.Series(arr)


# In[ ]:


pd.Series(arr,labels)


# **Dictionary**

# In[ ]:


pd.Series(d)


# **Data in a Series**
# 
# A pandas Series can hold a variety of object types:

# In[ ]:


#a series can hold almost any type of data objects
pd.Series(labels)


# In[ ]:


#we can also pass some built in function
# Even functions (although unlikely that you will use this)
pd.Series(data=[sum,print,len]) #to show the flexibility of the pandas series


# **Using an Index**
# 
# The key to using a Series is understanding its index. Pandas makes use of these index names or numbers by allowing for fast look ups of information (works like a hash table or dictionary).
# 
# Let's see some examples of how to grab information from a Series. Let us create two sereis, ser1 and ser2:

# In[ ]:


ser1=pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])
ser1


# In[ ]:


ser2=pd.Series([1,2,3,4],['USA','Germany','Italy','Japan'])
ser2


# In[ ]:


ser1['USA']


# **Operations are then also done based off of index:**

# In[ ]:


ser1 + ser2


# **PANDAS DATAFRAME**
# 
# DataFrames are the workhorse of pandas and are directly inspired by the R programming language. We can think of a DataFrame as a bunch of Series objects put together to share the same index. Let's use pandas to explore this topic!

# In[ ]:


from numpy.random import randn


# In[ ]:


np.random.seed(101)


# In[ ]:


df=pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])


# **Selection and Indexing**
# 
# Let's learn the various methods to grab data from a DataFrame

# In[ ]:


df['W']


# DataFrame Columns are just Series

# In[ ]:


type(df['W'])


# In[ ]:


df.W # SQL Syntax (NOT RECOMMENDED!)


# In[ ]:


# Pass a list of column names
df[['W','X']]


# **Creating a new column:**

# In[ ]:


df['new']=df['W']+df['Y']


# In[ ]:


df


# **Removing columns**

# In[ ]:


df.drop('new',axis=1,inplace=True)
df


# Can also drop rows this way

# In[ ]:


df.drop('E')


# In[ ]:


df.shape  # is a tuple
  


# **Selecting Rows**

# In[ ]:


df.loc['A']


# Or select based off of position instead of label

# In[ ]:


df.iloc[0]


# **Selecting subset of rows and columns**

# In[ ]:


df.loc['B','Y']


# In[ ]:


df.loc[['A','B'],['W','Y']]


# In[ ]:


df.loc[['A','B'],['W']]


# In[ ]:




