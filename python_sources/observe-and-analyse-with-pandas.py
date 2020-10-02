#!/usr/bin/env python
# coding: utf-8

# Importing libraries and reading file. 
# Display top 5 rows using head() method.
# 

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df=pd.read_csv('../input/chess/games.csv')
df.head(5)


# info() method displays number of rows, columns and datatype of each column

# In[ ]:


df.info()


# With describe method we can analyse the mean, median, quartiles of int / float type columns

# In[ ]:


df.describe()


# Bottom 75% of the data can be described as follows:
# -maximum turns a player has taken is 79
# -maximum black and white rating is around 1700
# -Players have made maximum 6 moves in the opening phase
# 

# We can alalyse the 'victory_status' and 'winner' columns. unique() displays unique values and value_counts displays the counts of each unique value

# In[ ]:


df['victory_status'].unique()


# In[ ]:


df['victory_status'].value_counts()


# In[ ]:


df['winner'].unique()


# In[ ]:


df['winner'].value_counts()


# We can analyse data by'victory_status' column. For this we create a group object using groupby() function and can apply methods to the object such as mean() and std()  

# In[ ]:


group=df.groupby('victory_status')
group.mean()


# Let's check the number of unique values in opening_name and opening_ply

# In[ ]:


df['opening_name'].nunique()


# In[ ]:


df['opening_ply'].nunique()


# So let's see what are the top 50 most common openings that the players have used.

# In[ ]:


df['opening_name'].value_counts().head(50)

