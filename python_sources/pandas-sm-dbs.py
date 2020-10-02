#!/usr/bin/env python
# coding: utf-8

# # 1.3 - Pandas 

# Pandas is built on Numpy. It has two types of easy to use and fast data structures , which helps to do Data Analysis with ease.
# 
# Data Structures:
# 
# - Series : One Dimensional array.
# - Dataframe : Two dimenionsal data structure with Row , Columns of different types.

# In[ ]:


import pandas as pd # import pandas Library
import numpy as np


# #### Series 

# In[ ]:


print(pd.Series([1,2,3]))
ser = pd.Series([1,2,3] , index = ['first', 'second','third']) #Index acts as reference 
print(se)


# In[ ]:


ser['first'] #Using index to access Data elements


# In[ ]:


se[0] #Using postion to access data elements


# #### DataFrame 

# In[ ]:


# Python Dictionary
p_dict = {'name' : ['Sachin','Sourav','Rahul'], 
         'country' : ['Ind','Ind','Ind'],
         'Runs': ['10000','10500','10100']}
print(p_dict)


# In[ ]:


#Dataframe - This will convert the dictionary to DataFrame
df = pd.DataFrame(p_dict) 
print(df.head())  #by default shows the first five rows
print('\n')
print(df.head(2))


# In[ ]:


df.tail(2) #last two rows


# Here we can see that Index is by default created.

# #### Slice & Dice in Dataframe

# In[ ]:


#Selecting a specific column
#Below are the two ways which produces the same result

print(df.Runs)
print(df['Runs'])


# iloc is used for integer-location based indexing / selection by position.Here we select rows and columns by number.
# 
# df.iloc[row_selecton , column_selection]

# In[ ]:


print(df.iloc[0]) #First row of the dataframe
print('\n')
print(df.iloc[-1]) #Only the last row
print('\n')
print(df.iloc[ :,1]) #all the rows and first column


# Please note a selection e.g.[1:6], will run from the first number to one minus the second number. e.g. [1:6] will go 1,2,3,4,5. (excludes the last number)

# In[ ]:


print(df.iloc[ :,0:2]) #All rows and 1st to 2nd column , : -> this will retrieve all rows/columns


# Few Common Dataframe Operations, we will perform on famous <b> Kaggle Titanic dataframe </b> - https://www.kaggle.com/c/titanic/overview

# In[ ]:


df = pd.read_csv('train.csv') # read_csv is the function to read csv file


# In[ ]:


df.shape # To check the count of rows and columns


# In[ ]:


df.columns #to check the column names


# In[ ]:


df.dtypes # to see datatypes of the columns


# In[ ]:


df["Survived"].value_counts() # to check unique values present in a column


# Boolean Indexing
# - or -> condition 1 | condition 2
# - and -> condition 1 & condition 2
# - Not -> ~ (not condition)
# - equal -> == (equal criteria)

# In[ ]:


# No. of females who survived?
df[(df.Sex =='female') & (df.Survived ==1)].shape[0] #() are mandatory for multiple conditions,()head -first 5 rows


# In[ ]:


# % of Male survivors
(len(df[(df.Sex =='male') & (df.Survived ==1)])/len(df))*100


# In[ ]:


# % of Female survivors
(len(df[(df.Sex =='female') & (df.Survived ==1)])/len(df))*100


# In[ ]:


# To check if there is any duplicate PassengerID
sum(df.PassengerId.duplicated())


# In[ ]:


#Average Age of passengers grouped by Gender and Survival Status
df.groupby(by = ['Sex','Survived']).mean().Age

