#!/usr/bin/env python
# coding: utf-8

# # Pandas equivalent of 10 useful SQL queries
# ### ... or Pandas for SQL developers

# In case you don't know, pandas is a python library for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. The name is derived from the term "panel data", an econometrics term for data sets that include observations over multiple time periods for the same individuals.[[1]](https://en.wikipedia.org/wiki/Pandas_(software) Basically, it is a way of working with tables in python. In pandas tables of data are called `DataFrame`s.  
# As the title suggests, in this article I'll show you the pandas equivalents of some of the most useful SQL queries. This can serve both as an introduction to pandas for those who already know SQL or as a cheat sheet of common pandas operations you may need.

# For the examples below I will use [this](https://www.kaggle.com/datasnaek/youtube-new#USvideos.csv) dataset which consists of data about trending YouTube videos in the US.

# In[ ]:


import numpy as np
import pandas as pd

# Reading the csv file into a DataFrame
df = pd.read_csv('../input/youtube-new/USvideos.csv')
df


# Pandas operations, by default, don't modify the data frame which you are working with; they just return other data frames which you need to assign to a variable if you want to save the changes. For most examples below we don't change our original data frame, we just show the returned result.

# ## 1. SELECT

# `SELECT col1, col2, ... FROM table`

# The SELECT statement is used to select columns of data from a table.  
# To do the same thing in pandas we just have to use the array notation on the data frame and inside the square brackets pass a list with the column names you want to select.

# In[ ]:


df[['video_id', 'title']]


# The same thing can be made with the following syntax which makes easier to translate WHERE statements later:

# In[ ]:


df.loc[:, ['video_id', 'title']]


# `SELECT DISTINCT col1, col2, ... FROM table`

# The SELECT DISTINCT statement returns only unique rows form a table.  
# In a data frame there may be duplicate values. If you want to get only distinct rows (remove duplicates) it is as simple as calling the `.drop_duplicates()` method. Judging based on this method's name you may think that it removes duplicate rows from your initial data frame, but what it actually does is to return a new data frame with duplicate rows removed.

# In[ ]:


df.loc[:, ['channel_title']].drop_duplicates()


# `SELECT TOP number col1, col2 FROM table`  
# or  
# `SELECT col1, col2, ... FROM table LIMIT number`

# The TOP or LIMIT keyword in SQL is used to limit the number of returned rows from the top of the table.  
# In pandas this is very easy to do with `.head(number)` method. Pandas also has the `.tail(number)` method for showing the rows from the end of data frame.

# In[ ]:


df.loc[:, ['video_id', 'title']].head(5)


# In[ ]:


df.loc[:, ['video_id', 'title']].tail(5)


# SQL's MIN(), MAX(), COUNT(), AVG(), and SUM() functions are pretty straightforward to translate to pandas:

# `SELECT MIN(col) FROM table`

# In[ ]:


df.loc[:, ['views']].min()


# `SELECT MAX(col) FROM table`

# In[ ]:


df.loc[:, ['views']].max()


# `SELECT COUNT(col) FROM table`

# In[ ]:


df.loc[:, ['views']].count()


# `SELECT AVG(col) FROM table`

# In[ ]:


df.loc[:, ['views']].mean()


# `SELECT SUM(col) FROM table`

# In[ ]:


df.loc[:, ['views']].sum()


# Now, what if we want to do something like this:  
# `SELECT MAX(likes), MIN(dislikes) FROM table`?  
# We need to do this in more steps:

# In[ ]:


new_df = df.loc[:, ['likes']].max().rename({'likes': 'MAX(likes)'})
new_df['MIN(dislikes)'] = df.loc[:, ['dislikes']].min().values[0]
new_df


# ## 2. WHERE

# `SELECT col1, col2, ... FROM table WHERE condition`

# The WHERE clause is used to extract only the rows that fulfill a specified condition.

# Recall the syntax we used so far for selecting columns:  
# `df.loc[:, ['col1', 'col2']]`  
# Inside the square brackets of `.loc` there is place for two parameters; so far we only used the second one which is used to specify what columns you want to select. Guess for what is the first parameter? Is for selecting rows. Pandas data frames expect a list of row indices or boolean flags based on which it extracts the rows we need. So far we used only the `:` symbol which means "return all rows". If we want to extract only rows with indices from 50 to 80 we can use `50:80` in that place. For extracting rows based on some condition, most often we will pass there an array of boolean flags returned by some (vectorized) boolean operation. The rows on positions where we will have False will not be included in the result, only those rows with True on their positions will be returned.

# Using equality and inequality operators **==, <, <=, >, >=, !=** in conditions is straightforward. For example, to return only rows that have number of likes >= 1000000 we can use:

# In[ ]:


df.loc[df['likes'] >= 1000000, ['video_id', 'title']]


# Note that the reason for which we could do what we did above (`df['likes'] >= 1000000`) is that pandas has overwritten the default behavior for >= operator so that it applies the operator element-wise and returns an array of booleans of the shape that we need (number of rows).  
# But the operators **and, or, not** don't work like that. So, we will use **&** instead of **and**, **|** instead of **or**, **~** instead of **not**.

# In[ ]:


df.loc[(df['likes'] >= 1000000) & (df['dislikes'] <= 5000), ['video_id', 'title']].drop_duplicates()


# `SELECT col1, col2, ... FROM table WHERE colN IS NOT NULL`

# In SQL you can use `IS NULL` or `IS NOT NULL` to get rows that contain/don't contain null values.

# How to check for null values in pandas?  
# We will use `isnull(array-like)` function from pandas package to do that. Note that this is not a method of data frame objects, don't use `df.isnull(...)`; instead do `pd.isnull(df['column'])`. So be careful.

# In[ ]:


df.loc[~pd.isnull(df['description']), ['video_id', 'title']].drop_duplicates()


# `SELECT col1, col2, ... FROM table WHERE colN LIKE pattern`

# The LIKE keyword can be used in a WHERE clause to test if a column matches a pattern.  
# In pandas we can use python's native re module for regular expressions to accomplish the same thing, or even more as the python's re module allows for a richer set of patterns to be tested rather than SQL's LIKE.   
# 
# We will create a function `like(x, pattern)` where x is an array-like object and pattern is a string containing the pattern which we want to test for. This function will first compile the pattern into a regular expression object, then we can use the `.fullmatch(val)` method to test the `val`'s value against our pattern. In order to apply this test to each element in our x vector we will use numpy's `vectorize(func)` function to create a vector equivalent for our operation of regex matching. Finally we apply this vectorized function to our x input vector. Then all we need to do is to pass `like(df['column'], pattern)` as pirst parameter in `.loc[]`.   
#   
# As an example the below code returns all videos that contains the word 'math' in their description.

# In[ ]:


import re

def like(x, pattern):
    r = re.compile(pattern)
    vlike = np.vectorize(lambda val: bool(r.fullmatch(val)))
    return vlike(x)

df_notnull = df.loc[~pd.isnull(df['description']), :]
df_notnull.loc[like(df_notnull['description'], '.* math .*'), ['video_id', 'title']].drop_duplicates()


# ## 3. ORDER BY

# `SELECT col1, col2, ... FROM table ORDER BY col1, col2 ASC|DESC`

# This SQL keyword is used to sort the results in ascending or descending order.  
# It is straightforward to translate this to pandas, you just call the `.sort_values(by=['col1', ...], ascending=True/False)` method on a dataframe.

# In[ ]:


df.loc[df['likes'] >= 1000000, ['video_id', 'title']].sort_values(by=['title'], ascending=True).drop_duplicates()


# ## 4. GROUP BY

# `SELECT col1, col2, ... FROM table GROUP BY colN`

# The GROUP BY statement groups rows that have the same value for a specific column. It is often used with aggregate functions (MIN, MAX, COUNT, SUM, AVG).  
# In pandas it is as simple as calling the `.groupby(['col1', ...])` method, followed by a call to one of `.min()`, `.max()`, `.count()`, `.sum`, `.mean()` methods.

# In[ ]:


df.loc[:, ['channel_title', 'views', 'likes', 'dislikes']].groupby(['channel_title']).sum()


# ## 5. HAVING

# `SELECT col1, col2, ... FROM table GROUP BY colN HAVING condition`

# The HAVING keyword is used to filter the results based on group-level conditions.  
# In pandas we have the `.filter(func)` method that can be called after a `groupby()` call. We need to pass to this method a function that takes a data frame of a group as a parameter and returns a boolean value that decides whether this group is included in the results or not.   
# But if we want to do more things at once in pandas, e.g. apply aggregate functions on columns and filter results based on group-level conditions, we need to do this in more steps. Whereas in SQL we could have done this in only one query.  
# In the example below we want to group by *channel_title*, allow only channels that have at least 100 different videos in the table, and apply average function on *views*, *likes*, and *dislikes*.  
# 
# In SQL this would be:  
# ```sql
# SELECT channel_title, AVG(views), AVG(likes), AVG(dislikes)
# FROM videos_table
# GROUP BY channel_title
# HAVING COUNT(video_id) > 100;
# ```

# In[ ]:


g = df.groupby(['channel_title'])
g = g.filter(lambda x: x['video_id'].count() > 100)
g = g.loc[:, ['channel_title', 'views', 'likes', 'dislikes']].groupby(['channel_title']).mean()
g


# ## 6. INSERT

# `INSERT INTO table (column1, column2, ...) VALUES (value1, value2, ...)`

# This SQL statement is used to insert new rows in the table.  
# In pandas we can use the `.append()` method to append a new data frame at the end of an existing one. We will use `ignore_index=True` in order to continue indexing from the last row in the old data frame.

# In[ ]:


new_row = pd.DataFrame({'video_id': ['EkZGBdY0vlg'],
                        'channel_title': ['Professor Leonard'],
                        'title': ['Calculus 3 Lecture 13.3: Partial Derivatives']})
df = df.append(new_row, ignore_index=True)
df


# ## 7. DELETE

# `DELETE FROM table WHERE condition`

# DELETE statement is used to delete existing rows from a table based on some condition.  
# In pandas we can use `.drop()` method to romove the rows whose indices we pass in. Unlike other methods this one doesn't accept boolean arrays as input. So we must convert our condition's output to indices. We can do that with `np.where()` function.  
# In the example below we deleted all the rows where *channel_title != '3Blue1Brown'*.

# In[ ]:


df.drop(np.where(~(df['channel_title'] == '3Blue1Brown'))[0])


# ## 8. ALTER

# `ALTER TABLE table ADD column`

# This SQL statement adds new columns.  
# In pandas we can do this by: `df['new_column'] = array-like`.   
# 
# Below we add a new column 'like_ratio':

# In[ ]:


df['like_ratio'] = df['likes'] / (df['likes'] + df['dislikes'])


# In[ ]:


df


# `ALTER TABLE table DROP COLUMN column`

# This SQL statement deletes a column.  
# `del df['column']` is how we do this in pandas.

# In[ ]:


del df['comments_disabled']


# In[ ]:


df


# ## 9. UPDATE

# ```sql
# UPDATE table_name
# SET column1 = value1, column2 = value2, ...
# WHERE condition;
# ```

# The UPDATE statement is used to change values in our table based on some condition.  
# For doing this in python we can use numpy's `where()` function. We also saw this function a few lines above when we used it to convert boolean array to indices array. That is what this function does when given just one parameter. This function can receive 3 arrays of the same size as parameters, first one being a boolean array. Let's call them c, x, y. It returns an array of the same size filled with elements from x and y choosen in this way: if c[i] is true choose x[i] else choose y[i].  
# To modify a data frame column we can do: `df['column'] = np.where(condition, new_values, df['column'])`.  
# In the example below we increase the number of likes by 100 where channel_title == 'Veritasium'.

# This is how the data looks before:

# In[ ]:


df.loc[df['channel_title'] == 'Veritasium', ['title', 'likes']]


# In[ ]:


df['likes'] = np.where(df['channel_title'] == 'Veritasium', df['likes']+100, df['likes'])


# And after:

# In[ ]:


df.loc[df['channel_title'] == 'Veritasium', ['title', 'likes']]


# ## 10. JOIN

# A JOIN clause is used to combine rows from two or more tables based on a related column between them.

# In order to show examples of joins I need at least two tables, so I will split the data frame used so far into two smaller tables.

# In[ ]:


df_titles = df.loc[:, ['video_id', 'title']].drop_duplicates()
df_titles


# In[ ]:


df_stats = df.loc[:, ['video_id', 'views', 'likes', 'dislikes']].groupby('video_id').max()
df_stats = df_stats.reset_index()
df_stats


# Doing joins in pandas is straightforward: it has a `.join()` method that we can use like this:   
# `df1.join(df2.set_index('key_column'), on='key_column')`

# There are more types of joins: inner, full, left, and right joins.  
# - INNER JOIN: returns rows that have matching values in both tables
# - FULL (OUTER) JOIN: returns rows that have matching values in any of the tables
# - LEFT JOIN: returns all rows from the left table, and the matched rows from the right one
# - RIGHT JOIN: returns all rows from the right table, and the matched rows from the left one  

# To specify which type of join you want in pandas you can use the **how** parameter in `.join()` method. This parameter can be one of: 'inner', 'outer', 'left', 'right'.

# Below are examples of these types of joins of the two data frames above on 'video_id' column.

# ```sql
# SELECT column_name(s)
# FROM table1
# INNER JOIN table2
# ON table1.column_name = table2.column_name;
# ```

# In[ ]:


df_titles.join(df_stats.set_index('video_id'), on='video_id', how='inner')


# ```sql
# SELECT column_name(s)
# FROM table1
# FULL OUTER JOIN table2
# ON table1.column_name = table2.column_name
# WHERE condition;
# ```

# In[ ]:


df_titles.join(df_stats.set_index('video_id'), on='video_id', how='outer')


# ```sql
# SELECT column_name(s)
# FROM table1
# LEFT JOIN table2
# ON table1.column_name = table2.column_name;
# ```

# In[ ]:


df_titles.join(df_stats.set_index('video_id'), on='video_id', how='left')


# ```sql
# SELECT column_name(s)
# FROM table1
# RIGHT JOIN table2
# ON table1.column_name = table2.column_name;
# ```

# In[ ]:


df_titles.join(df_stats.set_index('video_id'), on='video_id', how='right')


# If you want to learn more about pandas please refer to their [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html).  
# 
# I hope you found this information useful and thanks for reading!
