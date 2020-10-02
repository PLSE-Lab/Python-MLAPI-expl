#!/usr/bin/env python
# coding: utf-8

# # PANDAS in Python

# Pandas is a data analysis library that provides a variety of data structures and data manipulation methods that allows to perform complex tasks with simple one-line commands.
# 
# There are two core objects in Python. They are 
# 
# 1. DatFrame
# 2. Series
# 

# In[ ]:


#To find out the version of Pandas library installed

import pandas
pandas.__version__


# In[ ]:


#Importing packages with alias name

import numpy as np
import pandas as pd


# In[ ]:


#Creating series with variable name as data
#Series is a 1 dimensional indexed array

data = pd.Series([0, 1, 2, 3, 4, 5])
data


# In[ ]:


print(data)


# In[ ]:


#print the values of data

data.values


# In[ ]:


#print the index of data

data.index


# In[ ]:


#Accessing Series value

data[4]


# In[ ]:


#Slicing of Series

data[2:5]


# In[ ]:


#Assigning the customized index

data = pd.Series([0, 1, 2, 3, 4, 5], index = ["a" , "b" , "c" , "d" ,"e", "f"])
data


# In[ ]:


#Converting Python Dict to Pandas Series ->Keys are changed as Indexes and values are changed to value of series.

Infant_Details = {"Name": "Akshay Kannan",
                 "Age": 1,
                 "Sex": "M"}
Infant = pd.Series(Infant_Details)
Infant


# In[ ]:


#Accessing the Series

Infant.Name


# In[ ]:


#Accessing the Series

Infant["Name"]


# In[ ]:


#Slicing the Series 

Infant["Name":"Age"]


# In[ ]:


#Creating empty series

e_series = pd.Series()
e_series


# In[ ]:


#Converting Numpy array to Pandas Series

data11 = np.array(['a','b','c','d'])
np_series = pd.Series(data11)
np_series


# In[ ]:


#Create a Series from Scalar

s_series = pd.Series("Aarthi", index = [0,1,2,3,4])
s_series


# In[ ]:


#Reading a CSV file

info = pd.read_csv("D:/New folder/CSV Practise .csv")
info


# In[ ]:


salary = pd.Series(info["Salary"])
salary


# In[ ]:


salary.min()


# In[ ]:


salary.max()


# In[ ]:


#indexing the series using .loc() function -> provides the values from start num mentioned and include the end number also

salary.loc[3:5]


# In[ ]:


#indexing the series using .iloc() function -> provides the values from start num mentioned and exclude the end number

salary.iloc[3:5]


# In[ ]:


d1 = pd.Series([5, 2, 3, 7], index=['a', 'b', 'c', 'd'])
d2 = pd.Series([1, 6, 4, 9], index=['a', 'b', 'd', 'e'])
print(d1, "\n\n", d2)


# In[ ]:


d1.add(d2, fill_value=0)


# In[ ]:


d1.add(d2)


# In[ ]:


#Find out the data type of salary series

salary.dtype


# In[ ]:


#Find out the count of elements in the salary series -> Provide all data points includes the null als0

salary.count()


# In[ ]:


#sort_values() -> sort all the data points in asc order by default

salary.sort_values()


# In[ ]:


#To sort the values in desending order

salary.sort_values(ascending =False)


# In[ ]:


#Sort the series in descending order and view only the first 10 values

salary.sort_values(ascending =False).head(10)


# In[ ]:


#Sort the  series in descending order and  view only last 3 values

salary.sort_values(ascending =False).tail(3)


# In[ ]:


#Find out the unique values in the series

salary.unique()


# In[ ]:


#Find out the count of unique values in the series

salary.nunique()


# In[ ]:


#value_counts() -> to count the number of the times each unique value occurs in a Series

salary.value_counts()

