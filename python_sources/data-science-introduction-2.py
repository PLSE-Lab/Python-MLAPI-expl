#!/usr/bin/env python
# coding: utf-8

# **Basic steps to start Data Science: PART2**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/tmdb_5000_movies.csv')  # Show the path you will use


# In[ ]:


data.info() # Display the content of data


# **User Defined Functions**

# In[ ]:


def tuble_ex():  #define your function
    """Write here anything describes your function"""  #function description with triple quotes
    t=(1,2,3)   #write what you want your function to do
    return t
print(tuble_ex())


# **Scope**

# In[ ]:


# can you guess that what does this code print?
x = 2
def f():
    x = 3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope


# In[ ]:


# What if there is no local scope
x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x

# First local scope searched, then global scope searched, if both of them can not be found lastly built in scope searched.


# In[ ]:


# How can we learn what is built in scope
import builtins
dir(builtins)
# These are defined functions and it is not recommended to use them as a variable name


# **Nested Functions**

# In[ ]:


# A nested function
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


# **Default and Flexible Arguments**

# In[ ]:


# default arguments
def f(a, b = 1, c = 2):
    y = a + b + c
    return y
print(f(5))

# what if we want to change default arguments
print(f(5,4,3))


# In[ ]:


# flexible arguments *args
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)
print("")

# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    """ print key and value of dictionary"""
# If you do not understand this part turn for loops in PART1 and look at dictionary in for loop
    for key, value in kwargs.items():               
        print(key, " ", value)
        
f(country = 'spain', capital = 'madrid', population = 123456)
#f(something)  it gives an error


# **Lambda Function**

# In[ ]:


# Lambda function: makes functions simpler and shorter
square = lambda x: x**2     # where x is name of argument
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))


# **Anonymus Function**

# In[ ]:


# Anonymus function: is just like Lambda but it takes more than one arguments
number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))


# **Iterators**

# In[ ]:


# An iteration example
name = "ronaldo"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration


# **Zipping**

# In[ ]:


# An zipping example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
# It combines lists to save memory (allocation)
print('')
# An unzipping example
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))


# **List Compherension *****
# 
# *One of the most important topic for Data Science.*
# 
# We use list comprehension for data analysis often. 
# 
# List Comprehension: collapses for loops for building lists into a single line .
# 
# Ex: num1 = [1,2,3] and we want to make it num2 = [2,3,4]. This can be done with for loop. However it is unnecessarily long. We can make it one line code that is list comprehension.

# In[ ]:


# An example of a list comprehension
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)


# [i + 1 for i in num1 ]: list of comprehension
# 
# i +1: list comprehension syntax 
# 
# for i in num1: for loop syntax 
# 
# i: iterator 
# 
# num1: iterable object

# In[ ]:


# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 11 else i+5 for i in num1]
print(num2)


# In[ ]:


# lets return movie csv and make one more list comprehension example
# lets classify movies whether they are expensive or cheap. Our threshold is average budget.
threshold = sum(data.budget)/len(data.budget) # calculating average budget
print('Average budget: ', threshold)
data["budget_level"] = ["high" if i > threshold else "low" for i in data.budget]
data.loc[1706:1716,["budget_level","budget"]]   # I will mention "loc" more detailed later


# **Read more here:**
# 
# * [Basic steps to start Data Science: PART1](https://www.kaggle.com/osmanaliyardim/data-science-introduction)
# 
# * [Basic steps to start Data Science: PART3](https://www.kaggle.com/osmanaliyardim/data-science-introduction-3)
# 
# * [Basic steps to start Data Science: PART4](https://www.kaggle.com/osmanaliyardim/data-science-introduction-4)
