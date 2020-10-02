#!/usr/bin/env python
# coding: utf-8

# **Let's Start:**
# 
# This is my first kernel on Kaggle and I analyzed the avocado prices. 
# 
# Let's start with importing the libraries that we will use:

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

import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output
print (check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv ('../input/avocado.csv')


# Let's have a look at the data at a high level: 

# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


f, ax = plt.subplots (figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax, cmap="PiYG")
plt.show() #to clear the statements appearing by default
#cmap="PiYG":to set the color. Some other options: "YlGnBu" , "Blues", "BuPu", "Greens"
#annot=True :It gives us correlation values inside the boxes
#linewidths= .5 :Thickness of line between boxes
#fmt= '.2f' :It gives how many will be written of correlation values after comma


# In[ ]:


data.head(10)


# In[ ]:


data.columns


#  **MATLAB**
#  
# We will continue with creating 2D graphs. We will use Matlab library for this purpose. There are different kinds of graphs we can create. 
# 
# Here is Line Plot: 
# Line plot is better when x axis is time. 

# In[ ]:


data['AveragePrice'].plot(color = 'm', label = 'avg price',linewidth=2,alpha = 0.4,grid = True, linestyle = '-', figsize=(25,6)) #alpha:opacity
plt.legend(loc='upper right') # legend = puts label into plot
plt.yticks(np.arange(0.1,3.5,0.2))  #to set the range of y axis - (min, max, )
plt.xticks(np.arange(1000,18500,1000)) 
plt.xlabel('Sample Number') 
plt.ylabel('Feature Value of the Sample')
plt.title('Line Plot for Avg Price Values')            
plt.show()


# Scatter Plot:
# Scatter is better when there is correlation between two variables

# In[ ]:


data.plot(kind='scatter', x='Total Bags', y='Total Volume', alpha=0.5, c='r',figsize=(8,8))
plt.xlabel('total bags')
plt.ylabel('total vol')
plt.title('bags - vol scatter')
plt.show()


# Histogram:
# Histogram is better when we need to see distribution of numerical data.

# In[ ]:


data.AveragePrice.plot(kind='hist', bins = 20, figsize =(8,8), color='y', EdgeColor='m')
#bins = number of bar in figure
plt.xlabel('Avg Price')
plt.show()


# **PANDAS**

# 1-Filtering Pandas Data Frame

# In[ ]:


x = data['AveragePrice']>1.0
data[x]


# In[ ]:


data[data['AveragePrice']==data['AveragePrice'].max()]


# 2 - Filtering Pandas with logical_and:

# In[ ]:


data[np.logical_and(data['AveragePrice']<1,data['type']=='organic')]


# Alternative for logical_and:

# In[ ]:


data[(data['AveragePrice']<1) & (data['type']=='organic')]


# **Phyton Data Science Toolbox**
# 
# In this section, we will make some practice to go over the fundamental conceps. 
# 
# Tuple and docstring

# In[ ]:


#tuple is a data type 
def tuple_ex():
    """ This part that is to explain the functions is called docstrings"""
    t = (1,2,3)
    return t
a,b,c = tuple_ex()
print(a,b,c)


# Scope
# 
# * global
# * local
# * built-in

# In[ ]:


x = 2
def f():
    x = 3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope


#   The kernel first checks if there is any local scope. If there is, it is used. If not, then the global scope is used. If none of the two is found, then built in scope is checked.

# In[ ]:


x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x


# How can we learn what is built in scope?
# 
# We don't prefer using the below as the variable name. Because it may unnecesserily confuse the kernel and cause errors.

# In[ ]:


import builtins
dir(builtins)


# Nested functions:
# 
# Function inside of another function.

# In[ ]:


def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())    


# Default arguments:

# In[ ]:


def f(a, b=1)

#If b is not assigned to another value, then b=1.


# Flexible arguments:

# In[ ]:


def f (*args)

#we can define as many as different variables each time.

def f(**kwargs)
#flexible argument for dictinary


# Lambda function: 
# 
# fast way of writing a function.

# In[ ]:


#without lambda:
def f(x):
    return x**2
print (f(5))

#with lambda:
f= lambda x: x**2
print (f(4))


# Anonymous Function:
# 
# Lambda function for lists.
# 
# We use map(func, seq) here. 

# In[ ]:


number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))


# zip(): and unzip(): methods

# In[ ]:


# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z) #we created the zip object here.
z_list = list(z)
print(z_list)
print("")

#unzip example
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))


# List Comprehension

# In[ ]:


# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)


# In[ ]:


# lets classify avocado prices whether they are expensive or affordable. Our threshold is average price.
threshold = sum(data.AveragePrice)/len(data.AveragePrice)
print(threshold)
data["price_level"] = ["expensive" if i > threshold else "affordable" for i in data.AveragePrice]
data.loc[15000:15010,["price_level","AveragePrice"]] # we will learn loc more detailed later


# **CLEANING DATA**
# 
# Box plot:

# We will write a melt() method. We will make data visualisation with seaborn library later on using melt(). 

# In[ ]:


data_new=data.head()
#data_new
melted = pd.melt(frame=data_new, id_vars = 'AveragePrice', value_vars=['Total Volume', 'Total Bags'])
melted


# Reverse of melt():

# In[ ]:


melted.pivot(index = 'AveragePrice', columns = 'variable',values='value')


# Concatenating:
# 
# We will join to dataframes

# In[ ]:


#axis=0 means vertical:    

data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1, data2], axis=0, ignore_index=True) #ignore_index=True means ignore the real index and assign new
conc_data_row #calling the new table


# In[ ]:


#axis=1 means horizaontal

data1 = data['AveragePrice'].head()
data2 = data['Total Volume'].head()
conc_data_col = pd.concat([data1, data2], axis=1) 
conc_data_col 


# Datatypes:

# In[ ]:


data.dtypes


# In[ ]:


#let's change the data tyoe of all the inputs under 4046 from float to category
data['4046']=data['4046'].astype('category')
data.dtypes


# Missing Data and Testing with Assert

# In[ ]:


#How many inputs do we have for each category?
data['region'].value_counts(dropna=False) #dropna=False to show the NaN values, as well.


#   We do not have any NaN values in this dataset. But if we had, we might drop the NaN values.

# In[ ]:


data1=data
data1["region"].dropna(inplace=True)  #to drop NaN values
assert data["region"].notnull().all() #to check if we have NaN valur. As we already droped them, it returns (True), not an Erroe
data["region"].fillna('empty', inplace=True) #to fill NaN values with "empty"

