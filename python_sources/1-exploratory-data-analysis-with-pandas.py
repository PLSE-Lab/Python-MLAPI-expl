#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# import the dataset

# In[ ]:


df = pd.read_csv("../input/telecom_churn.csv")
df.head()


# Get the datatype of each column, number of rows, columns and missing values in each column

# In[ ]:


df.info()


# Get the unique values available in each columns. it helps to identify which column can be a categorical one.

# In[ ]:


df.nunique()


# In[ ]:


Analyze the values displaying in boolean columns


# In[ ]:


df['Churn'].value_counts()


# In[ ]:


df['Churn'].value_counts(normalize=True)


# In[ ]:


df.describe(include=['object','bool'])


# To explore the sorting function

# In[ ]:


df.sort_values(by='Total day charge',ascending=False).head()


# Sorting with Multiple columns

# In[ ]:


df.sort_values(by=['Total day charge','Total night charge','Total intl charge'],
               ascending=[True,False,True]).reset_index().head(10)


# Average value of dataframe for churn =1

# In[ ]:


avg_day_charge = df[df['Churn'] == False].mean()['Total day charge']


# There are 298 churn users who paid more than non churn user

# In[ ]:


df[(df['Churn'] == True) & (df['Total day charge'] > avg_day_charge)]


# In[ ]:


df[df['Churn'] == 1].mean()


# DataFrames can be indexed by column name (label) or row name (index) or by the serial number of a row. The loc method is used for indexing by name, while iloc() is used for indexing by number.
# 
# In the first case below, we say "give us the values of the rows with index from 0 to 5 (inclusive) and columns labeled from State to Area code (inclusive)". In the second case, we say "give us the values of the first five rows in the first three columns" (as in a typical Python slice: the maximal value is not included).

# Exploring data retrieval

# In[ ]:


df.loc[0:5,'State':'Area code']


# In[ ]:


df.iloc[0:6,0:3]


# If we need the first or the last line of the data frame, we can use the df[:1] or df[-1:] construct:

# In[ ]:


df.iloc[:1]


# In[ ]:


df[:1]


# In[ ]:


df[-1:]


# In[ ]:


df.iloc[-1:]


# Applying Functions to Cells, Columns and Rows
# To apply functions to each column, use apply():

# To find maximum value in each column

# In[ ]:


df.apply(np.max)


# In[ ]:


df.max()


# The apply method can also be used to apply a function to each row. To do this, specify axis=1. Lambda functions are very convenient in such scenarios. For example, if we need to select all states starting with W, we can do it like this:

# In[ ]:


df[df['State'].apply(lambda x:x[0:1]=='W')].head()


# The map method can be used to replace values in a column by passing a dictionary of the form {old_value: new_value} as its argument:
# 
# Series.map(arg, na_action=None)[source]
# Map values of Series according to input correspondence.
# 
# Used for substituting each value in a Series with another value, that may be derived from a function, a dict or a Series.

# In[ ]:


d = {'No' : False, 'Yes' : True}
df['International plan'] = df['International plan'].map(d)
df.head()


# The same thing can be done with the replace method:

# In[ ]:


df = df.replace({'Voice mail plan': d})
df.head()


# **We can create a summary, by 3 different functions.**
# 1. Group By
# 2. Cross tab
# 3. Pivot
# Please refer the following page https://pbpython.com/pandas-crosstab.html to get more insight. Now we will discuss each.

# Grouping
# In general, grouping data in Pandas works as follows:
# 
# df.groupby(by=grouping_columns)[columns_to_show].function()
# 
# First, the groupby method divides the grouping_columns by their values. They become a new index in the resulting dataframe.
# 
# Then, columns of interest are selected (columns_to_show). If columns_to_show is not included, all non groupby clauses will be included.
# 
# 

# Here we applied only sum function to the columns to show

# 

# In[ ]:


df.groupby(['State','Area code','International plan','Voice mail plan'])['Number vmail messages','Total day minutes'].sum().head(20)


# If we want to display more aggregate function in each columns, we need to apply them explicitly by using numpy function

# In[ ]:


df.groupby(['State','Area code','International plan','Voice mail plan'])['Number vmail messages','Total day minutes'].agg([np.mean,np.min,np.max]).head(20)


# To display the result of aggregate function in table format.

# In[ ]:


df.groupby(['State','Area code','International plan','Voice mail plan'])['Number vmail messages','Total day minutes'].agg([np.mean,np.min,np.max]).reset_index().head(20)


# In[ ]:




