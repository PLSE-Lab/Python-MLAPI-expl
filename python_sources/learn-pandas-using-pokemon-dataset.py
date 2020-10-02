#!/usr/bin/env python
# coding: utf-8

# # <font color=green>Pandas </font>

# ###  <font color=blue>- Pandas is an open-source Python Library providing high-performance data manipulation and analysis tool using   its powerful data structures. </font> 
# 
# ### <font color=blue> - We can use Pandas for different tasks like processing , analysis , manipilation of data etc. </font> 
# 
# ### <font color=blue>- Python with Pandas is used in a wide range of fields including academic and commercial domains including finance, economics, Statistics, analytics, etc.</font> 
# 
# ### <font color=blue>- Derived from the word panel data</font> 
# 

# ## <font color=red> _Pandas Datastructures_ </font>
# 
# ### <font color=blue> - Series </font> : <font color=green> It is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). It is size immutable</font>
# 

# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:





# In[ ]:


list =  ["Brazil", "Russia", "India", "China", "South Africa"]
places = pd.Series(list)
places.head()


# In[ ]:


places = pd.Series(list,index=["BR", "RU", "IN", "CH", "SA"])
places.head()
# places.index = ["BR", "RU", "IN", "CH", "SA"]


# ### <font color=blue> - Panel </font> : <font color=green> Panel is a three-dimensional data structurel </font>
# 
# 

# ### <font color=blue> - DataFrame </font> : <font color=green> DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table, or a dict of Series objects </font>
# 
# ### <font color=purple>- DataFrame(data=None, index=None, columns=None, dtype=None) </font>
# 
# ### <font color=blue> - DataFrame accepts many different kinds of input</font>
# #### <font color=green>lists, dicts</font>
# #### <font color=green>ndarray</font>
# #### <font color=green>A Series</font>
# #### <font color=green>Another DataFrame</font>

# In[ ]:


dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

info = pd.DataFrame(dict) #change index
info.head()


# In[ ]:


df_from_series = pd.DataFrame(places,columns=["a"]) #df_from_series.columns
df_from_series.head()


# In[ ]:


print(os.listdir("../input"))


# In[ ]:



df = pd.read_csv("../input/pokemon.csv") # header=None
df.head(10)


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:





# In[ ]:


df[['Name','Speed']]


# In[ ]:


del(df['Form'])


# In[ ]:


df.head()


# In[ ]:


#slicing
df[0:4]


# In[ ]:


#loc vs iloc
df.iloc[:2,:4]


# In[ ]:


df.loc[:2]


# ### Transpose : To change columns to rows and rows to columns

# In[ ]:


df.T


# ### axes :  To get x axis and y axis names or range i.e get rows and columns names or range
# 

# In[ ]:


df.axes


# ### dtypes : To get the data type of every column
# 

# In[ ]:


df.dtypes


# ### Empty : return true if dataset is empty else false
# 

# In[ ]:


df.empty


# ### ndim : return number of dimentions dataset has
# 

# In[ ]:


df.ndim


# ### shape : returns a tuple containing number of rows and columns
# 

# In[ ]:


df.shape


# ### size : returns total size of dataset
#  

# In[ ]:


df.size


# ### values : returns the values of dataset as an array
# 

# In[ ]:


df.values


# ### head : returns top 5 values from dataset  -> try using head( any nymber less than rows )
# 

# In[ ]:


df.head()


# ### tail: return  last 5 rows of the dataset

# In[ ]:


df.tail(100)


# In[ ]:


df.sum()


# In[ ]:





# In[ ]:


df.mean()


# In[ ]:


df.corr()


# In[ ]:


df.count()


# In[ ]:


df.max()


# In[ ]:


df.min()


# In[ ]:


df.median()


# In[ ]:


df.std()


# In[ ]:


df.describe() # 25%,50%,35%  min+(max-min)*percentile 


# **A Cheat Sheet For Pandas**
# __ https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf __

# In[ ]:


df = pd.DataFrame([0,1,2,3],index = ['a','b','c','d'])
df


# In[ ]:


df.reindex(columns = [0,1,2,3])


# In[ ]:


df.reindex(columns = [0,1,2,3],fill_value = "ml class")


# In[ ]:


df.reindex(index = ['a','c','d'])


# In[ ]:





# In[ ]:





# In[ ]:




