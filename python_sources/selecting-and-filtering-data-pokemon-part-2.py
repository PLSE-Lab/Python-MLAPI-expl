#!/usr/bin/env python
# coding: utf-8

# ## In this post, we'll still use the Pokemon dataset (cleaned version), but we'll focus on filtering and selecting data, two really important things that we need to learn when using Pandas. Fortunately, Pandas makes it really easy for us for doing these.
# 
# ## Table of Content
# * [Selecting subset of data](#select)
# * [Filtering dataframe ](#filter)
# * [Finding the strongest and lowest Pokemon with idxmax/idxmin and loc method](#find)

# Let's get started as usual by loading the important package,  loading the dataset from our locak disk, and read the first 5 rows of the dataframe

# In[ ]:


import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


pokedata = pd.read_csv("../input/pokemon-all-cleaned/pokemon_cleaned.csv")


# In[ ]:


pokedata.head()


# <a id='select'></a>
# ## Selecting subset of data
# 
# `pandas.DataFrame.head`is a Pandas function that  return the first *n* rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it. By default, it will show the first 5 rows, but we can just put a number inside the bracket and it will show .
# 
# Now what if we want to get the data between the 10th row and the 15th row. One of the easiest way to select a subset of data in Pandas is using the `square bracket []`. For example, if we want to select the data between 10th and 15th row, then we should write:

# In[ ]:


pokedata[10:15]


# Notice that even though we write 15 in the square bracket, it only returns data until the 14th row and the 15th row is excluded. Another way to select subset of data is by using `loc` method which we'll cover later.

# <a id='filter'></a>
# ## Filtering dataframe 
# 
# Pandas also makes it easy for us if we want to select some data with certain condition. Let's say you want to take a look at Charizard stats, the easiest way to do this is:

# In[ ]:


pokedata[pokedata['Name']=='Charizard']


# We can also use mathematical operator as a filter condition, let's say we want to see Pokemon with more than 150 speed,

# In[ ]:


pokedata[pokedata['Speed']>150]


# Look at how simple that is, you can also use different column as the filter conditions, say we want to see non legendary Electric-type Pokemon with more than 500 total stats, 

# In[ ]:


pokedata[(pokedata['Type 1']=='Electric') & (pokedata['Total']>500) & (pokedata['Legendary']==False)]


# <a id='find'></a>
# ## Finding the strongest and lowest Pokemon with idxmax/idxmin and loc method
# The `pandas.DataFrame.idxmax()` and `pandas.DataFrame.idxmin()` method can be used to return the index with the highest or lowest value. For example. if we want to see the index of the Pokemon with highest and lowest total stats, we should write:

# In[ ]:


pokedata['Total'].idxmax()


# In[ ]:


pokedata['Total'].idxmin()


# So now we know that the Pokemon with the highest total stats has index number of 492, while the Pokemon with lowest total stats has index number of 190.
# 
# The `pandas.DataFrame.loc()` method allows us to select a group of rows and columns by label(s) or a boolean array. For example,  `pokedata.loc[[50]]` will return the 50th row of the dataframe. And `pokedata.loc[10:15]`will return data between 10th and 15th row *(the 15th row will be included)*

# In[ ]:


pokedata.loc[[50]]


# In[ ]:


pokedata.loc[10:15]


# Now we can combine the `loc` and `idxmax` methods to find the Pokemon with highest and lowest Total stats

# In[ ]:


pokedata.loc[[pokedata['Total'].idxmax()]]


# In[ ]:


pokedata.loc[[pokedata['Total'].idxmin()]]

