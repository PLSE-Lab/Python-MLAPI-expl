#!/usr/bin/env python
# coding: utf-8

# ***!!! First of all !!!* **
# Sorry for my English skills
# Thank you for your understanding

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visuallization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/fifa19/data.csv')


# In[ ]:


data.info() # We will learn something about this data set


# In[ ]:


data.corr() # We will make it better after Code Line
# We make with matplotlib


# In[ ]:


f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(), annot = True , linewidths = 1, fmt = '.1f',ax = ax)
plt.title('Correlation Map of Data Set')
plt.show()


# In[ ]:


data.head()
# Default value is 5 but you can change it like this


# In[ ]:


data.head(10)
# ".tail()" method also works in the same logic  


# In[ ]:


data.tail()


# In[ ]:


data.tail(10)


# In[ ]:


data.columns
# We can learn columns' names with this methods


# In[ ]:


# First of all we must fix the column names 
data.columns = [each.split()[0]+'_'+each.split()[1] if(len(each.split()) > 1) else each for each in data.columns]
#This method combines spaces between words
data.columns = [each.lower() for each in data.columns]
#This method writes the titles in lower case
#We will learn for loop later in this kernel


# In[ ]:


data.columns
# This looks better


# In[ ]:


# Line Plot 
data.overall.plot(kind = 'line',color = 'red',label = "Overall",linewidth = 0.75,alpha = 1,grid = True)
data.potential.plot(color = 'green',label = 'Potential',linewidth = 1,grid = True,alpha = 0.15,linestyle = '-.')
plt.legend(loc = 'upper right')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Example of Line Plot')
plt.show()


# In[ ]:


#Scatter Plot
data.plot(kind = 'scatter', x = 'acceleration',y = 'sprintspeed',alpha = 0.25,color = 'red')
plt.xlabel('Acceleration')
plt.ylabel('Sprintspeed')
plt.title('Acceleration and Sprint Speed Scatter Plot')
plt.show()


# In[ ]:


#Histogram Plot
# bins = number of bar in figure
data.jersey_number.plot(kind = 'hist',bins = 99,figsize = (10,10))
plt.xlabel('Players Jersey Number')
plt.ylabel('Frequency')
plt.show()
# bins is 99 because 99 kind jersey number we have


# In[ ]:


#Let's create a dictionary!
dictionary = {'fruit' : 'apple','vegetable' : 'carrot'}
# fruit and vegetable are "KEY"
# apple and carrot are "VALUE"
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


dictionary['fruit'] = 'banana' # update
print(dictionary)
dictionary['fast-food'] = 'hamburger' # new entry
print(dictionary)
del dictionary['fruit'] # remove entry with key 'fruit'
print('fast-food' in dictionary) # cheack include or not
dictionary.clear() #remove all entries in dictionary
print(dictionary)


# Before continue with pandas, we need to learn **logic**, **control** **flow** and **filtering**.
# 
# Comparison operator: ==, <, >, <=
# 
# Boolean operators: and, or ,not

# In[ ]:


# Comparison operator

print(156 > -1)
print(1000000000 != 0)

# Boolean operators
print(True and False)
print(True or False)


# In[ ]:


x = data.gkreflexes > 88
# We have a 7 player who have higher goalkeeper reflexes value than 90
data[x]


# In[ ]:


# Let's use one Boolean operator
data[np.logical_and(data.gkreflexes > 88,data.overall >= 90)]
#You can only use 2 comparison options


# In[ ]:


# You can use Code line too
data[(data.gkreflexes > 88) & (data.overall >= 90)]


# **WHILE** and **FOR LOOPS**

# In[ ]:


n = 0
while n != 5:
    print('n is : ',n)
    n += 1
print(n,' is equal to 5')    


# In[ ]:


array1 = np.arange(1,6,1)# You don't need this method but this is same logic with lis = [1,2,3,4,5] 
lis = list(array1)
for n in lis:
    print('n is : ',n)
print(' ')    


# In[ ]:


# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index," : ",value)
print('')   


# In[ ]:


# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'fruit' : 'apple','vegetable' : 'carrot'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')


# In[ ]:


# For pandas we can achieve index and value
for index,value in data[['overall']][0:1].iterrows():
    print(index, ' : ' ,value)


# In this part , you have to learn:
# 
# 1]How to import csv file,
# 
# 2]Plotting line,Scatter Line and Histogram,
# 
# 3]Basic dictionary features,
# 
# 4]Basic pandas features like filtering that is actually something always used and main for being data scientist,
# 
# 5]While and For loops:
# 

# ***3.CLEANING DATA***

# In[ ]:


data = pd.read_csv("/kaggle/input/fifa19/data.csv")
data.head() # We learned this code line the starting of this kernel


# In[ ]:


data.tail() # This too


# In[ ]:


# shape gives number of rows and columns in a tuble
data.shape


# In[ ]:


# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()


# In[ ]:


data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split()) > 1) else each for each in data.columns]
data.columns = [each.lower() for each in data.columns]
data.columns


# value_counts(): Frequency counts
# 
# outliers: the value that is considerably higher or lower from rest of the data
# 
# Lets say value at 75% is Q3 and value at 25% is Q1.
# 
# Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1). (Q3-Q1) = IQR
# 
# We will use describe() method. Describe method includes:
# 
# count: number of entries
# 
# mean: average of entries
# 
# std: standart deviation
# 
# min: minimum entry
# 
# 25%: first quantile
# 
# 50%: median or second quantile
# 
# 75%: third quantile
# 
# max: maximum entry
# 

# What is quantile?
# 
# 1,4,5,6,8,9,11,12,13,14,15,16,17
# 
# The median is the number that is in middle of the sequence. In this case it would be 11.
# 
# 
# The lower quartile is the median in between the smallest number and the median i.e. in between 1 and 11, which is 6.
# 
# 
# The upper quartile, you find the median between the median and the largest number i.e. between 11 and 17, which will be 14 according to the question above.
# 
# 

# In[ ]:


data_head = data.head(125)
a = data_head.nationality.value_counts(dropna = False)
a


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column = "stamina",by = "work_rate",figsize = (12,12))
plt.show()


# In[ ]:


data_new = data.head(10)
data_new


# In[ ]:


data.columns


# In[ ]:


# Firstly I create new data from players data to explain melt more easily.
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame = data_new,id_vars = "name",value_vars = ["age","work_rate","value","wage"])
melted


# In[ ]:


# Index is name
# I want to make that columns are variable
# Finally values in columns are value

pivoted = melted.pivot(index = "name",columns = "variable",values = "value")
pivoted


# In[ ]:


# We can concatenate two dataframe
# Firstly lets create 2 data frame

data_head = data.head()
data_tail = data.tail()

conc_data_row = pd.concat([data_head,data_tail],axis = 0,ignore_index = True)
conc_data_row


# In[ ]:


data1 = data.overall.head()
data2 = data.potential.head()

data_1_and_2_conc = pd.concat([data1,data2],axis = 1)
# axis = 0 : adds dataframes in row
data_1_and_2_conc


# **DATA TYPES**

# In[ ]:


data.dtypes


# In[ ]:


# lets convert object(str) to categorical and int to float
data.nationality = data.nationality.astype("category")
data.overall = data.overall.astype("float")


# In[ ]:


# As you can see Type 1 is converted from object to categorical
# And Speed ,s converted from int to float
data.dtypes


# ***MISSING DATA and TESTING WITH ASSERT***
# If we encounter with missing data, what we can do:
# 
# 
# 1]leave as is
# 
# 2]drop them with dropna()
# 
# 3]fill missing value with fillna()
# 
# 4]fill missing values with test statistics like mean
# 
# Assert statement: check that you can turn on or turn off when you are done with your testing of the program
# 

# In[ ]:


# Lets look at does player data have nan value
# As you can see there are 18207 entries. However release_clause has 16643 non-null object so it has 1564 null object.
data.info()


# In[ ]:


data.release_clause.value_counts(dropna = False)


# In[ ]:


data.release_clause.dropna(inplace = True) # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?


# In[ ]:


#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true


# In[ ]:


# In order to run all code, we need to make this line comment
# assert 1==2 # return error because it is false


# In[ ]:


assert data.release_clause.notnull().all() # returns nothing because we drop nan values


# In[ ]:


data.release_clause.fillna('empty',inplace = True)


# In[ ]:


assert data.release_clause.notnull().all()# returns nothing because we do not have nan values

