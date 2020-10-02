#!/usr/bin/env python
# coding: utf-8

# In this kernel i will manipulate the columns of the data in different ways:
# 1. I will change the letters to lowercase
#         1.1. By list comprehension
#         1.2. By for loop
#         1.3. By lambda function in a map function
# 2. I will join the words by "_" in each columns

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

First we call the dataset and see the columns that we have:
# In[ ]:


data = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

data.columns   


# As shown above: there are space in between words such as in "Overal rank" and also letters have lowercase and higher case; i will make them all lowercased and joined with "_".

# **1.1. Lowercasing the letters by list comprehension**
# ![](http://)<lower()> method : lowers the letters of a string "STRING".lower()-->outputs "string"
#     
#     

# In[ ]:


data.columns = [each.lower() for each in data.columns]   # lowercases each column by for loop
print(data.columns)                                      # prints the columns


# In[ ]:


data.columns = [each.upper() for each in data.columns]  # uppercases all columns
print(data.columns)


# **1.2. Lowercase the columns in a different way: By for loop only**
# ![](http://)data.columns[i] will show me the column with "i" index.
# ![](http://)data.columns[0] ---> outputs "overal rank"

# In[ ]:


for each in data.columns:    # Cycyles each column in data
    each.lower()             # lowercased each column
    print(each)              # prints each column


# In[ ]:


print(data.columns[0]) # column index 0 ---> overal rank
print(data.columns[1]) # column index 0 ---> country or region


# I can define a counter (i) and start it with "0" out of loop and, count it in loop, then assign the each count to the data.column in that count: As in below coding:
# 
#         i=0                                  # counter, first count zero
#         for each in data.column:             # cycles each column in data
#             data.columns[i] = each.lower()   # lower cases each column and equalize it to column of index "i".
#             i = i+1                          # increase the count
# 
# This code will end up and error saying that('DataFrame' object has no attribute 'column') or (index does not support mutable operations". Because, eventhough the columns can be called by index such as in <data.columns[0], it is not mutable and can not be used in a loop.
# 
# **To get rid of this problem:**
# 
# ![](http://)I can convert data.columns to list and assign it to list_columns.
# ![](http://)Than i can use list_columns in loop because it is mutable.

# In[ ]:


list_columns = list(data.columns)   # Convert data.columns to list and assign it to list_columns 
i=0
for each in data.columns:           # Cycyles each column in data.columns
    list_columns[i]= each.lower()   # Lowercases each column and assigns them to indexes of list_columns 
    i = i + 1                       
    data.columns = list_columns     # Equalizes data.columns with values of list_columns 
print(data.columns)                 # Prints columns of data


# **1.3. Lowercasing the columns by using lambda function in map function**
# 
# map(func, *iterables) --> map object
# 

# In[ ]:


data.columns = [each.upper() for each in data.columns]   # Uppercases the columns
print(data.columns)


# In[ ]:


list_columns = list(data.columns)    # Converts data.columns to list and assing to list_columns
a = map(lambda x:x.lower(), list_columns)    # a is a map object includes lowercased columns
b = list(a)  # b is a list which is converted from a
data.columns = b   # values of list b assigned to data.coloumns
print(data.columns)

# Below line will also works in place of 3 lines of coding
# data.columns = list(map(lambda x: x.lower(), list(data.columns)))


# **2. Joining the words by "_" in each columns**
# 
# <split()> method: Splits the strings and creates a list: "string1 string2".split() ---> ["split1" , "split2"]
# 
# <len()> method: Outputs the length of a string: len("string") ---> 6 (counts letters)
# 
# <max(list)> method: Shows the maximum value in of a list
# 
First: I want to know that the maximum split word count in any column.
for ex: "overal rank" has "2" split word count / "healthy life expectancy" has "3" split word count etc.

# In[ ]:


lis1=list(data.columns)        # Converting data.column to list of lis_columns
i=0
for each in data.columns:      # Cycles each column in data
  lis1[i]=len(each.split())    # Splits each column and assign the length of each split to list_columns 
  i=i+1 
n = max(lis1)                  # Assignes max split word count in any column in data to "n" 
print(lis1)                    # [2, 3, 1, 3, 2, 3, 5, 1, 3]
print(n)


# In[ ]:


lis2 = list(data.columns)      # Conveting data.columns to list and assigning it to lis2
i=0
for each in data.columns:
  
  if len(each.split()) > n-1:
    lis2[i] = each.split()[0]+"_"+ each.split()[1]+"_"+each.split()[2]+"_"+each.split()[3]+"_"+each.split()[4]
    i=i+1
  elif len(each.split()) > n-2:
    lis2[i] = each.split()[0]+"_"+ each.split()[1]+"_"+each.split()[2]+"_"+each.split()[3]
    i=i+1
  elif len(each.split()) > n-3:
    lis2[i] = each.split()[0]+"_"+ each.split()[1]+"_"+each.split()[2]
    i=i+1
  elif len(each.split()) > n-4:
    lis2[i] = each.split()[0]+"_"+ each.split()[1]
    i=i+1
  else:
    lis2[i] = each
    i=i+1
data.columns = lis2
print(data.columns)

