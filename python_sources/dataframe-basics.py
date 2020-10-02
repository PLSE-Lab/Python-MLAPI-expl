#!/usr/bin/env python
# coding: utf-8

# ## Please upvote and comment if you find the notebook helpful

# <a id = "table_of_content"></a>
# 
# # Table of contents
# [1. Basic exploratory analysis](#Index1)
# 
# [2. Indexing a DataFrame](#Index2)
# 
# [3. Filter a DataFrame](#Index3)
# 
# [4. Group By DataFrame](#Index4)

# ## Import required packages

# In[ ]:


import pandas as pd
import numpy as np


# ## Create a dataframe from dictionary

# In[ ]:


data = {'weekday': ['Sun','Sun','Mon','Tue'],
        'city': ['Austin','Dallas','Pittsburgh','California'],
        'vistors': [145,135,676,456]
       }
users = pd.DataFrame(data)
print(users)


# <a id = "Index1"></a>
# 
# # Basic exploratory analysis
# 
# [Go back to the Table of Contents](#table_of_content)

# In[ ]:


# Read dataset
titanic_data = pd.read_csv("../input/train.csv")


# In[ ]:


# print type of the object
type(titanic_data)


# In[ ]:


# Shape of the dataframe
titanic_data.shape


# In[ ]:


# show the columns in the dataframe
titanic_data.columns


# In[ ]:


# show first n records
titanic_data.head(2)


# In[ ]:


# show last n records
titanic_data.tail(2)


# In[ ]:


# Change column names of the dataframe
# titanic_data.columns = ['a', 'b']
# titanic_data.rename(columns={'Name': 'PName', 'Ticket': 'Number'}, inplace=True)


# In[ ]:


# show the data types and missing values in the dataset
titanic_data.info()


# #### Age: 177 null values; Cabin: 687 null values ; Embarked: 2 null values

# In[ ]:


# Summary statistics of the dataframe - For only numerical variables (Integer and float)
titanic_data.describe()


# In[ ]:


# Summary stats for categorical variables
titanic_data.describe(include = ['object','bool'])


# In[ ]:


# Convert datatype of a column in the dataframe
titanic_data['Pclass'] = titanic_data['Pclass'].astype('object')
titanic_data.dtypes


# <a id = "Index2"></a>
# 
# ## Indexing a dataframe
# [Go back to the Table of Contents](#table_of_content)

# In[ ]:


# Indexing using square barckets
titanic_data['Name'][0]
#titanic_data.Name[0]


# In[ ]:


# Select only few columns in the data
titanic_data[['Survived', 'Pclass', 'Name', 'Sex']].head()


# In[ ]:


# Drop few columns in the data
titanic_data.drop(['Ticket','Cabin'],axis = 1).head()


# In[ ]:


# slicing the df using loc (loc is used with column  names)
titanic_data.loc[0,'Fare']


# In[ ]:


# slicing all the rows but few columns in df
titanic_data.loc[:,'Survived':'Sex'].head(2)


# In[ ]:


# Slicing few rows but all columns in the df
titanic_data.loc[0:4,:].head(2)


# In[ ]:


# Slicing selected rows and columns
titanic_data.loc[0:3,['Sex','Age']]


# In[ ]:


# Slicing the df using iloc (iloc is used with index numbers)
titanic_data.iloc[0,9]


# In[ ]:


# Slicing the data frame rows by iloc
titanic_data.iloc[5:8,:]


# In[ ]:


# Slicing selected rows and columns using iloc
titanic_data.iloc[[0,4,6], 0:2]


# In[ ]:


# Slicing the last n rows of the dataframe using iloc
titanic_data.iloc[-5:,:]


# <a id = "Index3"></a>
# ## Filtering a data frame
# [Go back to the Table of Contents](#table_of_content)

# In[ ]:


# Filter only male records from df
titanic_data[titanic_data['Sex'] == 'male'].shape


# In[ ]:


# Filtering with a boolean series
titanic_data[titanic_data.Age > 50].shape


# In[ ]:


# Filter by a string pattern in a column
titanic_data[titanic_data['Name'].str.contains('Sir')]


# In[ ]:


# Multiple conditions within a filter
titanic_data[(titanic_data.Age >= 50) & (titanic_data.Fare > 30)].shape


# In[ ]:


# Filter on multiple conditions within same column
titanic_data[titanic_data['Embarked'].isin(['C','Q'])].shape


# In[ ]:


# Drop any rows with missing values
titanic_data.dropna(how = 'any').shape


# <a id = "Index4"></a>
# ## Group by - DataFrame 
# [Go back to the Table of Contents](#table_of_content)

# In[ ]:


# Group by a column (Avg survival rate by gender)
titanic_data.groupby("Sex")["Survived"].mean()


# In[ ]:


# Group by jointly on two columns (Avg survival rate by gender and Pclass)
titanic_data.groupby(["Sex","Pclass"])["Survived"].mean()


# In[ ]:


# In the Pandas version, the grouped-on columns are pushed into the MultiIndex of the resulting Series by defaul
# More closely emulate the SQL result and push the grouped-on columns back into columns in the result, 
# you an use as_index=False
titanic_data.groupby(["Sex","Pclass"],as_index = False)["Survived"].mean()


# In[ ]:


# Average fare by passenger class and sort it by descending order
Fare_by_class = titanic_data.groupby("Pclass",as_index = False)["Fare"].mean()
Fare_by_class.sort_values(['Fare'],ascending = False).head()


# # Appendix
# https://realpython.com/pandas-groupby/

# In[ ]:




