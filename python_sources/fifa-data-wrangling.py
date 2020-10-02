#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Q1: Read data from fifa_ranking.csv file
import os  #for setting working directory 


# In[ ]:


os.chdir("../input")
os.listdir()


# In[ ]:


import pandas as pd  # for reading csv files
data = pd.read_csv("../input/fifa_ranking.csv")


# In[ ]:


# Lets Start Now 

data.shape              # list the Row = 57793 & columns = 16


# In[ ]:


data.head()             # List top 5 records


# In[ ]:


data.tail()             # List last 5 records 


# In[ ]:


# Q2: Print full summary
data.describe           # List few records from top & Bottom. Rows = 57793, Columns = 16


# In[ ]:


# Q3: How many rows are there?
data.shape              # list the Row = 57793 & columns = 16


# In[ ]:


# Q4: How many columns are there?
data.shape              # list the Row = 57793 & columns = 16


# In[ ]:


# Q5: Get the row names
list(data.index)


# In[ ]:


# Q6: Get the column names
data.columns            # List index of all column names 
data.columns.values     # List an array of all column names 
data.dtypes             # List data type for each column 
data.info()             # Like summary & str function in R 


# In[ ]:


# Q7: Change the name of all-columns to uppercase using rename() and str.upper()

data.rename(columns = {'rank':'RANK','country_full':'COUNTRY_FULL','country_abrv':'COUNTRY_ABRV', 'total_points':'TOTAL_POINTS',
       'previous_points':'PREVIOUS_POINTS', 'rank_change':'RANK_CHANGE', 'cur_year_avg':'CUR_YEAR_AVG',
       'cur_year_avg_weighted':'CUR_YEAR_AVG_WEIGHTED', 'last_year_avg':'LAST_YEAR_AVG', 'last_year_avg_weighted':'LAST_YEAR_AVG_WEIGHTED',
       'two_year_ago_avg':'TWO_YEAR_AGO_AVG', 'two_year_ago_weighted':'TWO_YEAR_AGO_WEIGHTED', 'three_year_ago_avg':'THREE_YEAR_AGO_AVG',
       'three_year_ago_weighted':'THREE_YEAR_AGO_WEIGHTED', 'confederation':'CONFEDERATION', 'rank_date':'RANK_DATE'})


# In[ ]:


data.columns
data.columns.values   


# In[ ]:


# Q8: Order the rows of data by 'previous_points' and 'rank_change'

data.sort_values(by = ['previous_points', 'rank_change'])


# In[ ]:


# Q9: Get the 2nd column

newdf = data[data.columns[1]]
print(newdf)


# In[ ]:


#or 

data[data.columns[1]]


# In[ ]:


#or

df1 = data[['country_full']] 
print(df1)


# In[ ]:


# Q10: Get the country_abrv array
import numpy as np
df2 = data[['country_abrv']].copy()
df2.values


# In[ ]:


# Q11: Subset rows 1, 3, and 6

data.loc[[1,3,6],:]


# In[ ]:


# Q13: Subset the first 3 rows
data.loc[:2]
#or


# In[ ]:


data[0:3]


# In[ ]:


# Q14: Subset rows excluding the first 3 rows
data[3:57793]


# In[ ]:


# Q15: Subset the last 2 rows
data[-2:]


# In[ ]:


# Q16: Subset rows where previous_points > 40
data[data.previous_points > 40]


# In[ ]:


# Q17: Subset rows where country_full= France
data[data.country_full == 'France']


# In[ ]:


# Q18: Subset rows where previous_points > 52 and confederation = UEFA
data[(data.previous_points > 52) & (data.confederation == 'UEFA')]


# In[ ]:


# Q19: Subset by columns 1 and 3
sub_col_ix = data.ix[:, [1,3]]
sub_col_ix


# In[ ]:


#or 
sub_col_iloc = data.iloc[:,[1,3]]
sub_col_iloc


# In[ ]:


#or
sub_col_loc = data.loc[:,['country_full','total_points']] 
sub_col_loc


# In[ ]:


# Q20: Subset by columns rank_change and previous_points
col_loc = data.loc[:, ['rank_change', 'previous_points']]
col_loc


# In[ ]:



#or 

col_sub = data[['rank_change', 'previous_points']]
col_sub


# In[ ]:


# Q21: Subset rows where previous_points > 5 and subset columns by confederation and country_full

data[(data.previous_points > 5) & (data.confederation) & (data.country_full)]


# In[ ]:


#or 

df2 = data[(data.previous_points > 5)]
df2.shape
data.shape
df2_loc = data.loc[:,['confedertion', 'country_full']]
df2_loc


# In[ ]:


# Q22: Insert a new column, Foo = previous_points * 10
df = data


# In[ ]:


df.shape


# In[ ]:


df.loc[:,'Foo'] = data.loc[data['previous_points']*10]


# In[ ]:


df.shape


# In[ ]:


data.shape


# In[ ]:


df.loc[:,['previous_points','Foo']]


# In[ ]:


# Q23: Remove column Foo

df.drop(['Foo'], axis = 1, inplace = True)


# In[ ]:


data.columns


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


# Q24: Group By confederation and calculate Aggregate (mean) of previous_points

df.groupby('confederation')['previous_points'].mean()


# In[ ]:





# In[ ]:




