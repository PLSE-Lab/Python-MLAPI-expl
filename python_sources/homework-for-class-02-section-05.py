#!/usr/bin/env python
# coding: utf-8

# Data Cleaning(Class 02 Section 05) is to be reviewed here
# 
# ## DATA CLEANING
# We may need to clear and transform data from
# * Lacking data in rows
# * Col name inconsistency
# * Bad format(string values where they should be numbers)
# 
# ### Importing and taking a glance
# We need to inspect our data if it needs a cleaning.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("../input/pokemon.csv")
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()


# ## EXPLORATORY DATA ANALYSIS
# ### VALUE_COUNTS()
# Frequency of values in a series

# In[ ]:


data["Type 1"].value_counts()


# ## QUARTILES - OUTLIERS
# Quartiles are certain middle grounds in our data.
# 
# Assume we have a serie. Sorted by it values,
# * %50 or **The Median** or **The Second Quartile** is the number in middle of a serie
# * %25 or **The First Quartile** is the one between the minimum and %50
# * %75 or **The Third Quartile** is the one between the %50 and maximum

# In[ ]:


serie = pd.DataFrame([9, 5, 6, 1, 3, 8, 12, 100, 11])
# sorted it is: 1 3 5 6 8 9 11 12 100
serie.describe()


# * Interquartile Range(**IQR**) is:
#     * IQR = Q3-Q1
# * An outlier is 
#   * smaller than Q1 - (1.5 * IQR)
# **OR** 
#   * bigger than Q3 + (1.5 * IQR)
# 

# ## VISUAL EXPLORATORY DATA ANALYSIS(EDA)

# In[ ]:


import matplotlib.pyplot as plt

data.boxplot(column="HP")
plt.show()


# In graphic above, the box is median, the lines are min and max, the circles are outliers.

# ## TIDY DATA WITH MELT
# 
# we'll simplify the data first, to see more clearly

# In[ ]:


summary = data.head()
summary


# Now we melt the data. **Melting** means, taking only the columns we state.

# In[ ]:


melted = pd.melt(frame=summary, id_vars="Name", value_vars = ["Attack", "Defense"])
melted


# ## PIVOTING THE MELTED DATA
# Pivot creates some sort of table which is 2 dimensional. Both columns and the rows are metadata, and values are the data.

# In[ ]:


melted.pivot(index='Name', columns='variable', values='value')


# ## CONCATENATING THE DATA
# We concat dataframes with concat function.

# In[ ]:


concated = pd.concat([data.head(), data.tail()], axis=0, ignore_index=True)
concated


# In[ ]:


concated_h = pd.concat([data["Attack"].head(), data["Defense"].head()], axis=1)
concated_h


# ## MISSING DATA
# you can 
# * leave as-is
# * drop them
# * fill them

# In[ ]:


# here we fill all the empty columns in data with string 'empty'
data.fillna('empty')


# In[ ]:


# here we drop every row that contains any empty cell
data.dropna()


# ## ASSERTION
# **Assertion** is a general programming concept. With it, we validate an expression. I will be using try-except for better output.

# In[ ]:


# in python, assertion returns nothing if it passes.
try:
    assert 1 == 1
    print("Assertion passed")
except AssertionError as e:
    print("ASSERTION FAILED")


# In[ ]:


# it throws AssertionError if it does not pass

#i use try-catch so jupyter wont stop on assertion error
try:
    assert 1 == 2
except AssertionError as e:
    print("ASSERTION FAILED")


# Lets make an assertion in a serie

# In[ ]:


dt = pd.DataFrame({"Nums":[1,2,3,None,5,6,7]})

# returns if serie does not contain any null values. will return false
dt["Nums"].notnull().all()


# In[ ]:


# lets drop null values
dt.Nums.dropna(inplace=True)
dt


# In[ ]:


# and assert if there is no null value remaining after dropping
try:
    assert dt.Nums.notnull().all()
    print("Assertion passed")
except AssertionError as e:
    print("ASSERTION FAILED")
# Passed!

