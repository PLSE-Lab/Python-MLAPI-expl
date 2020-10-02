#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# 
# Hello everyone
# In this tutorial we will learn basic python Data Science libraries step by step. Let's begin
# 
# 
# Fist of all we need to import numpy and pandas libraries

# In[ ]:


import numpy as np
import pandas as pd 


# For this tutorial we will use a sample data. It is Global Terrorism Database. I added it. Now we need to call it to our project.

# In[ ]:


data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')


# We can check it with following codes:

# In[ ]:


data.info()


# In[ ]:


data.corr()


#  We need to import  matplotlib.pyplot and seaborn  for plotting  correlation map 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# Let's look first 10 data of our data

# In[ ]:


data.head(10)


# Very nice .
# Than let's look column names of our data

# In[ ]:


data.columns


# PLOTTING SOME VALUES OF DATA
# 
#  *Line plot is better when x axis is time.
# * Scatter is better when there is correlation between two variables
# * Histogram is better when we need to see distribution of numerical data.
# * Customization: Colors,labels,thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle  

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.targtype1.plot(kind = 'line', color = 'g',label = 'targtype1',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.targsubtype1.plot(color = 'r',label = 'targsubtype1',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='targtype1', y='targsubtype1',alpha = 0.5,color = 'red')
plt.xlabel('targtype1')              # label = name of label
plt.ylabel('targsubtype1')
plt.title('targtype1 targsubtype1 Scatter Plot') # title = title of plot
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
data.country.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


# clf() = cleans it up again you can start a fresh
data.country.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()


# <a id="3"></a> <br>
# ### DICTIONARY
# Why we need dictionary?
# * It has 'key' and 'value'
# * Faster than lists
# <br>
# What is key and value. Example:
# * dictionary = {'spain' : 'madrid'}
# * Key is spain.
# * Values is madrid.
# <br>
# <br>**It's that easy.**
# <br>Lets practice some other properties like keys(), values(), update, add, check, remove key, remove all entries and remove dicrionary.

# In[ ]:


#create dictionary and look its keys and values
dictionary = {'turkey' : 'ankara','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['usa'] = "las vegas"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['usa']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)


# In[ ]:


# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted


# <a id="4"></a> <br>
# ### PANDAS
# What we need to know about pandas?
# * CSV: comma - separated values

# In[ ]:


data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')


# In[ ]:


series = data['country']        # data['Defense'] = series
print(type(series))
data_frame = data[['country']]  # data[['Defense']] = data frame
print(type(data_frame))


# <a id="5"></a> <br>
# Before continue with pandas,   we need to learn **logic, control flow** and **filtering.**
# <br>Comparison operator:  ==, <, >, <=
# <br>Boolean operators: and, or ,not
# <br> Filtering pandas

# In[ ]:


# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['country']>200     
data[x]


# In[ ]:


# 2 - Filtering pandas with logical_and

data[np.logical_and(data['country']>200, data['iyear']>2000 )]


# In[ ]:


# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['country']>200) & (data['iyear']>2000)]


# ### WHILE and FOR LOOPS
# We will learn most basic while and for loops

# In[ ]:


# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 50 :
    print('i is: ',i)
    i +=10 
print(i,' is equal to 5')


# In[ ]:


# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['country']][0:1].iterrows():
    print(index," : ",value)


# In this part, you learn:
# * how to import csv file
# * plotting line,scatter and histogram
# * basic dictionary features
# * basic pandas features like filtering that is actually something always used and main for being data scientist
# * While and for loops
# 

# 

# 
