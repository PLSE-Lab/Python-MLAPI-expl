#!/usr/bin/env python
# coding: utf-8

# ### Introduction

# Pandas is one of the python's library that is used for the data analysis.
# It plays a vital role in exploratory data analysis , from reading csv file to modify the values in the dataframe. I have been listed out the ways that will be help easy to use and understand . You can take this simply as a tutorial for beginners and I try to make it a well formated notebook , that works as complete clean code for the user.

# <h3 style="color:red">Please Upvote the Kernel. If you like this, its free. </h3>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express  as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from matplotlib import cm
plt.style.use('ggplot')
pd.__version__


# If we go with the simple definition of the **pandas series object**.<br/>
# It is a one-dimensional array of indexed data. It can be created from a
# list or array.

# In[ ]:


data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print("-"*50)
# used to print the values
print(data.values)
print("-"*50)
# print particular value
print(data.values[0])
print("-"*50)
# slice with indexes
print(data[1:3])
print("-"*50)
# print index 
print(data.index)


# ### Explict Index Definition with Pandas Series

# We can even define an explict indexs with pandas series objects. Also we can access the values with its index.

# In[ ]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
index=['a', 'b', 'c', 'd'])
print(data)
print("-"*50)
print(data['a'])
print("-"*50)
print(data['a':'c'])


# In[ ]:


# convert a dict to pandas Series
age_dict = {'Ram': 17,
'Mathew': 20,
'Ali': 17,
'Raj': 19,
'Sharad': 15}
age = pd.Series(age_dict)
print(age)


# In[ ]:


# scalar 
data = pd.Series(2, index=['a','b','c'])
print(data)


# In[ ]:


# convert a dict to pandas Series
marks_dict = {'Ram': 90,
'Mathew': 80,
'Ali': 67,
'Raj': 49,
'Sharad': 35}
marks = pd.Series(marks_dict)
print(marks)


# Now that we have this along with the age Series and marks Series, we can use a
# dictionary to construct a single two-dimensional object containing this information:

# In[ ]:


student_data = pd.DataFrame({'age': age, 'marks': marks})
print(student_data)
print("-"*50)
# print the index
print(student_data.index)
print("-"*50)
print(student_data.columns)


# ### Pandas Index Object

# The Series and DataFrame objects contain an explicit
# index that lets you reference and modify data. This Index object is an interesting
# structure in itself, and it can be thought of either as an immutable array or as an
# ordered set (technically a multiset, as Index objects may contain repeated values).

# In[ ]:


ind = pd.Index([2, 3, 5, 7, 11])
print(ind)


# In[ ]:


# if you try to modify the value , you can't . it will throw an exception here.
# ind[0] = 1
print("-"*50)
print(ind[0])
print("-----size-----shape-------ndim------dtype-----")
# it has also attributes that are similar to numpy array.
print(ind.size, ind.shape, ind.ndim, ind.dtype)


# In[ ]:


indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])


# In[ ]:


# intersection
print(indA & indB)
print("-"*50)
# union
print(indA | indB)

# symmetric
print("-"*50)
print(indA ^ indB)


# In[ ]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
index=['a', 'b', 'c', 'd'])
print(data)
print("-"*50)
print(data.keys())
print("-"*50)
print(data['a'])


# In[ ]:


print('a' in data)
print("-"*50)
print(list(data.items()))
print("-"*50)
data['e'] = 1.25
print(data)
print("-"*50)
# slicing by explicit index
print(data['a':'c'])
# slicing by implicit integer index
print("-"*50)
print(data[0:2])
print("-"*50)
# masking
print(data[(data > 0.3) & (data < 0.8)])
# fancy indexing
print("-"*50)
print(data[['a', 'e']])


# ### Merge, Concat,  Join

# ### Merge
# 
# Merge data frame merges the two different dataframes having different values

# In[ ]:


df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
                    'value': [1, 2, 3, 5]})
df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
                    'value': [5, 6, 7, 8]})
df1


# In[ ]:


df2


# In[ ]:


df1.merge(df2, left_on='lkey', right_on='rkey')


# ### Reading a csv file

# In[ ]:


netflix_data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')


# In[ ]:


# it will display the with rows , if not mentioned it will display only 5 rows.
netflix_data.head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})


# In[ ]:


# display the numerical values count, mean etc
netflix_data.describe()


# ### Get the Object data using describe 

# In[ ]:


I = netflix_data.dtypes[netflix_data.dtypes == 'object'].index


# In[ ]:


I


# In[ ]:


netflix_data[I].head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})


# In[ ]:


# displays count, unqiue, top and freq
netflix_data[I].describe()


# In[ ]:


netflix_data.loc[0]


# In[ ]:


netflix_data.iloc[0]


# In[ ]:


# finding out the null values
print(netflix_data.isnull().sum())
print("-"*50)
print(netflix_data.notnull().sum())


# In[ ]:


# this will drop the null values rows completly
netflix_data = netflix_data.dropna()
# netflix_data.dropna(axis='columns', how='all')


# In[ ]:


netflix_data['type'].unique()


# In[ ]:


temp_df = pd.DataFrame(netflix_data['type'].value_counts()).reset_index()

fig = go.Figure(data=[go.Pie(labels=temp_df['index'],
                             values=temp_df['type'],
                             hole=.7,
                             title = '% of Netflix by Type',
                             marker_colors = px.colors.sequential.Blues_r,
                            )
                     
                     ])
fig.update_layout(title='Netflix Shows')
fig.show()


# In[ ]:


netflix_data['rating']


# In[ ]:


ax = sns.countplot(x="rating", data=netflix_data)


# In[ ]:


ax = sns.countplot(x="type", data=netflix_data)


# In[ ]:


netflix_data.head()


# In[ ]:


# get movies with duration
movie_len = netflix_data[netflix_data['type'] == 'Movie']


# In[ ]:


# find movie duration 
ax = sns.countplot(x="release_year", data=movie_len)


# # more cool stuff to come....
