#!/usr/bin/env python
# coding: utf-8

# # Introduction to Panda: Basic commands
# This is my very first post / kernel on kaggle, and I'm entirely new to Data Science / Data Analytics, so there might be mistakes here and there throughout this article. Inputs and comments are greatly appreciated, and I'll try to improve this post overtime. Since I'm new to this, I'll post comments and explanation on basic commands that might seem obvious by nature, but its more for my sake of learning. I'll start off with importing the 2 basic libraries that will be used throughout.

# In[ ]:


import pandas as pd
import numpy as np


# I would then fill in the proprietary panda command to configure the display option. The appropriate path / directory to the relevant csv file: train.csv is also set.

# In[ ]:


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
data = pd.read_csv('../input/train.csv')


# The function '-.shape' is used to tell use the number of rows and columns.

# In[ ]:


data.shape


# The function '-.head(n)' is used to show the first n rows, with the default value for n being 5. The opposite is true for '-.tail(n)', to show the last n rows.

# In[ ]:


print(data.head()) #similar case to data.tail(), with output being the last 5 rows.


# The **data types** for each individual column can be checked with 'data.dtypes'.

# In[ ]:


print(data.dtypes)


# To **check the missing values**, I can use the '-.isnull()' command, and combine it with '-.sum()' to count the amount of null data in every column.

# In[ ]:


data.isnull().sum()


# To check for **duplicate data**, I consider PassengerId as the key, and checked to see if they're unique with below code:

# In[ ]:


print(data.PassengerId.nunique())


# The value matches with the amount of row (recall that the data shape was 891 x 12), which means there are no duplicate PassengerId in the file. 
# 
# Now, we'll move on to the **data distribution check**. To begin, I will first see the overall view, using the command '-.describe()' as shown below.

# In[ ]:


print(data.describe())


# I could then view various properties for each column. The formatting of the code is simply '[DatasetName].[ColumnName].[Command()]'. For example, the code below classify the numbers of people in their corresponding ship class.

# In[ ]:


print(data.Pclass.value_counts())


# For **Data sorting**, we could use the command '-.sort_values('ColumnName', ascending=(True/False), inplace=True)'. The inplace argument is merely there to substitute the other way to run this code, both will be shown below:

# In[ ]:


data.sort_values('PassengerId', ascending=False, inplace=True)
#data = data.sort_values('PassengerId', ascending=False) #other alternative, same output
print(data.head())


# As you can see, the head() function is now listed in descending order, since we sorted the PassengerId to descend in value. We specifued the sort_values ascending argument to be False (ascending = false).
# **Dropping Duplicate Data** is simply done by using the command '-.drop_duplicates('ColumnName', keep= 'first/last'. In the code snippet below, I will drop duplicate Sex (which is not the best idea, but its for the sake of learning).

# In[ ]:


data.drop_duplicates('Sex', keep='first', inplace=True)
#data = data.drop_duplicates('Sex', keep='first') #also alretnatives for the absence of 'inplace=True'
print(data.shape)
print("Here are all the data left:\n")
print(data)


# As you can see, 'data' now only contains two people, with different sex. This is because every same sexes are dropped after the first two unique sexes are found. The argument 'keep=first' makes sure that the program will keep the first unique row. The other possible argument is 'keep=last', which will let the computer drop every other duplicates except for the last one. 
# 
# **Filling missing values** can be done using the '-.fillna(SubsValue, inplace=True), example shown below. But first, I'd have to re-import 'data' since it was left with only 2 rows from last command. In the code below, I will replace every null values with the average age of the rest of the passengers.

# In[ ]:


data = pd.read_csv('../input/train.csv')
print("Number of null entries in Age column before fill in is: "+ str(data['Age'].isnull().sum()) + "\n")
avr_age = float(data['Age'].mean())
data['Age'].fillna(avr_age, inplace=True) #avr_age arguments contains the float value for the replacement
print("Number of null entries in Age column after fill in is: "+ str(data['Age'].isnull().sum()))


# **Group By** in Panda is somewhat similar to the method I've used in mySQL. I will explain the code below on the following paragraph.

# In[ ]:


data_gp = data.groupby(['Survived', 'Sex']).agg({'PassengerId': 'count',
                                      'Age': 'mean'
                                      })
data_gp.reset_index(inplace=True) #to retain the index numbering for the resulting table
print(data_gp)


# The 'Survived' value above is user-defined, and the data author uses 0 to signify death and 1 means alive.  Notice the arrangement of the groupby function; we put in 'Survived' and 'Sex' column respectively. This order will classify the number of Females & Males that died and survived.

# **End of Note** that is all that I've learned for today, hope you learned someting new from this kernel.
