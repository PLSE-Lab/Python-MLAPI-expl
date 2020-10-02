#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Selecting Data**

# In[ ]:


'''
This tutorial will work on the Fifa Data Set. 
We will create a DataFrame names fifa and keep using it throughout the Tutorial
'''
fifa = pd.read_csv('../input/data.csv' , index_col=0)


# In[ ]:


#printing the first 5 rows
fifa.head()


# In[ ]:


#printing first 10 rows
fifa.head(10)


# In[ ]:


#getting information about the DataFrame
fifa.info()


# In[ ]:


#list of columns
fifa.columns


# In[ ]:


#iterating through all columns
for column in fifa.columns:
    print(column)


# In[ ]:


#iterating through rows
for index,row in fifa.iterrows():
    print(f'Player {row["Name"]} belongs to {row["Nationality"]}')
    if index> 5:
        break


# In[ ]:


#Accessing data for a single column
fifa['Name']


# In[ ]:


#alternate way
fifa.Name


# In[ ]:


name = fifa.Name
print(type(name))


# The iloc and the loc operator are used to select data on an index based approach
# The iloc operator takes as arguments the integer indexes of the rows and columns, while the loc operator can work with Labelled indexes as well.

# In[ ]:


fifa.iloc[0:2 ,0:10 ]


# In[ ]:


fifa.loc[0:2,['Name','Age','CB']]


# In[ ]:


#using the iloc operator on an attribute of DataFrame
fifa.Club.iloc[0:10]


# **Boolen indexing / Conditional Selection**

# In[ ]:


fifa.Club =='Juventus'


# In[ ]:


fifa[fifa.Club =='Juventus']


# In[ ]:


fifa[(fifa.Club =='Juventus') & (fifa.Nationality =='Portugal')]


# In[ ]:


fifa.loc[fifa.LF.isnull()]


# In[ ]:


fifa[fifa.LF.isnull()]


# **Creating a new Column**

# In[ ]:


fifa['goodPlayer'] = True


# In[ ]:


fifa.head(2)


# In[ ]:


fifa.goodPlayer = fifa.apply(lambda x: True if x.Potential >90 else False, axis='columns' )


# In[ ]:


fifa.loc[:20,['Potential','goodPlayer']]


# **Data Cleaning**

# Deleting a Column
# 

# In[ ]:


fifa_copy = fifa
#fifa_copy.drop(['CM','CB'], axis=1)


# In[ ]:


#replacing values with Null Values by some other value
fifa.LF.fillna('80+5')


# In[ ]:


#replacing a value using apply function
#apply lambda function


# **Basic Data Calculation and Manipulation**

# Information about a column

# In[ ]:


fifa.Nationality.describe()


# In[ ]:


fifa[fifa.Nationality=='England']


# In[ ]:


fifa.CM.describe()


# In[ ]:


fifa.Potential.describe()


# In[ ]:


fifa.dtypes


# In[ ]:


#Selects Unique Nationalities of players belonging to Juventus
fifa[fifa.Club=='Juventus'].Nationality.unique()


# In[ ]:


#Selects the number of players per Country belonging to Juventus
fifa[fifa.Club=='Juventus'].Nationality.value_counts()


# In[ ]:


#Selects the number of players per Country belonging to Juventus
fifa[fifa.Club=='Juventus'].groupby(fifa.Nationality).Nationality.count()


# In[ ]:


#Lists the maximum Potential per Nationality
fifa.groupby(fifa.Nationality).Potential.max()


# In[ ]:


#apply Lambda Function per Nationality. the Lambda function simply sums up the Potential
fifa.groupby(fifa.Nationality).apply(lambda x:sum(x.Potential))


# **Sorting Values**

# In[ ]:


fifa.sort_values(by ="Nationality", ascending =True)


# **Mini-Exercise 2**

# Use the fifa DataFrame to answer the following questions -
# 1.  Print the Name, Nationality, Clubs and Value of the players from row 10-20 (inclusive)
# 2.  Print the Name, Nationality and Value of players belonging to Chelsea.
# 3.  Print the Name, Nationality and Value of the star players of Chelsea. [ Assume that a player is considered as a star player if Potential is more than 90]
# 4. Find the total value of players belonging to Chelsea.
# 5. Find the total value of star players belonging to Chelsea.
# 6. Find the number of goalkeepers per Nation. Print it in a descending order.
# 7. Drop all the columns except the name, Nationality, Value , club, Potential of the player. Create a new column called as  'goodBuy', which will have True if the Value/Potential < mean of Value/Potential, and False otherwise.

# In[ ]:


#placeholder for q1


# In[ ]:


#placeholder for q2


# In[ ]:


#placeholder for q3


# In[ ]:


#placeholder for q4


# In[ ]:


#placeholder for q5


# In[ ]:


#placeholder for q6


# In[ ]:


#placeholder for q7

