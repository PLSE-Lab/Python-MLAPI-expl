#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/fifa19/data.csv')
data.info()


# USER DEFINED FUNCTION

# In[ ]:


def first_func(*args):
    """returns square root of every item in function's parameter"""
    for i in args:
        print(i**(1/2))


# In[ ]:


first_func(4,9,16,25,36)


# SCOPE OF FUNCTION

# In[ ]:


var1 = 48
def test_function():
    var1 = 24
    print('value of var1 (inside the function) : ', var1)
test_function()
print('value of var1 (outside the function) : ', var1)


# In[ ]:


# if there is no local variable function uses global variable
x = 4
def test_function01():
    y = x**x
    print(y)
test_function01()


# NESTED FUNCTION

# In[ ]:


def one():
    """one function returns square of result of function two"""
    def two():
        x = 10
        y = 5
        z = x+y
        return z
    return two()**2
print(one())


# DEFAULT ARGUMENTS

# In[ ]:


def def_arg(first='name'):
    print(first + seco)
def_arg('surname')


# FLEXIBLE ARGUMENTS

# In[ ]:


def flex_arg(*args):
    """returns lowercase of arguments"""
    for i in args:
        print(lower(i))
flex_arg('name','surname')


# ### flexible arguments on dictionaries

# In[ ]:


def new_func(**kwargs):
    for i,j in kwargs.items():
        print(i,j)
new_func(name = 'george best', team = 'liverpool', year = 1976)


# lambda function

# In[ ]:


testo = lambda x : x**(1/2)
print(testo(81))


# ### list comprehension

# In[ ]:


my_list = ['numpy', 'pandas', 'matplotlib']
new_list = [i[::-1] for i in my_list]
print(new_list)


# In[ ]:


data['status'] = ['world class' if i > 92  else  'non' for i in data.Potential]


# In[ ]:


data.loc[::,['Name','status']]


# In[ ]:




