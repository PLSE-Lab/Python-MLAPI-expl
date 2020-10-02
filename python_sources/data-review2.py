#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Content
# Phyton Basics
# * * Matplotlib
# * * Dictionaries
# * * Pandas
# * * Logic, control flow and filtering
# * * Loop data structures
# * 
# * **# # PHYTON DATA SCIENCE TOOLBOX:**
# * 
# * User defined function
# * Scope
# * Nested function
# * Default and flexible arguments
# * Lambda function
# * Anonymous function
# * Iterators
# * List comprehension
#  
# ****** Cleaning Data
# * Diagnose data for cleaning
# * Explotary data analysis(EDA)
# * Visual exploratory data analysis
# * Tidy data(melting)
# * Pivoting data
# * Concatenating data
# * Data types
# * Missing data and testing with assert
# 
# 

#  # Step 1: Loading required libraries
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#  # Step 2: Import and first look data, and columns' names
# 

# In[ ]:


data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.columns
data.head()


# #Step 3: Correlation check between integer variables 
# 

# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# #Step 4: possible plots for data
# 

# In[ ]:


#line plot

data.Amount.plot(kind = 'line', color = 'g', label = 'Amount', linewidth = 1, alpha = 0.9, grid = True, linestyle = ':',figsize=(15,9))
data.V2.plot(kind = 'line', color = 'r', label = 'V2', linewidth = 1, alpha = 0.9, grid = True, linestyle = '-.',figsize=(15,9))
plt.legend(loc = 'upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('line plot')


# In[ ]:


#scatter plot

data.plot(kind='scatter', x='Amount',y='V2',alpha=0.9,color='red',grid = True, figsize=(15,9))
plt.xlabel('Amount')
plt.ylabel('V2')
plt.title('Amount V2 Scatter Plot')


# In[ ]:


#histogram



# example data
mu = data.V2.mean()  # mean of distribution
sigma = data.V2.std()  # standard deviation of distribution
x = mu + sigma * data.V2

num_bins = 50

fig, ax = plt.subplots(figsize=(10,10))

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--',alpha=0.9,color = 'r')
ax.set_xlabel('V2')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of V2')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()


# In[ ]:


#for pandas we can achieve index and value

for index,value in data[['Amount']][0:1].iterrows():
    print(index,":",value)


# 
# PHYTON DATA SCIENCE TOOLBOX: 
# * User defined function
# * Scope
# * Nested function
# * Default and flexible arguments
# * Lambda function
# * Anonymous function
# * Iterators
# * List comprehension

# In[ ]:


# User defined functions
    
    #docstring: documentation for functions


    
def list_ex():
    """return defined + tuble """
    t = list(data.Amount.head())
    return t
a,b,c,d,e = list_ex()
print(a,b,c,d,e)
    


# In[ ]:


#scope
  #global: defined main body in script
  #local: defined in a function
  #built in scope: like print, len

x = int(data.Amount[1])
def f():
   x = 3
   return x
print(x)       
print(f())

#what if there is no local scope

x = int(data.Amount[1])
def f():
   y = 2*x
   return y
print(f())


#first local scope searched, then global scope searched,
# if two of them cannot be found lastly built in scope



# In[ ]:


#how can we learn what is built in scope

import builtins
dir(builtins)


# In[ ]:


#Nested Function
 # function inside function

def square():
    """return square of value"""
    def add():
        """add two local variable"""
        x=data.Amount[1]
        y=data.V2[1]
        z=int(x+y)
        return z
    return add()**2
print(square())


# In[ ]:


#Default and Flexible Arguments
 #Default


    
    
    
def f(a,b=data.V2[1],c=len(data.V2)):
    """b and c are default argument"""
    y = int(a+b+c)
    return y
print(f(5))
print(f(5,4,3))


# In[ ]:


#Flexible
     # *args for list
def f(*args):
    for i in args:
        print(i)
        
f(data.V2[2])
print("")
f(data.V2[2:9])

      # **kwargs for dictionary
def f(**kwargs):
    """print key and value of dictionary"""
    for key, value in kwargs.items():
        print(key,"",value)
        
f(country='spain',capital='madrid',population=123456)


# In[ ]:


#Lambda Function
 #faster way of writing function

square = lambda x:x**2
print(square(data.Amount[5]))

tot = lambda x,y,z : x+y+z
print(tot(1,2,3))


# In[ ]:


#Anonymous Function
#like lambda function but it can take more than one arguments

number_list = data.Amount[0:10]
y = map(lambda x:int(x**2),number_list)
print(list(y))


# In[ ]:


#Iterators
#iterable: an object with an associated iter() method
#example: list, string and dictionary
#iterator: produce next value with next() method

name = "'Venus'Nora"
it = iter(name)
print (next(it,'-1'))
print(*it)

#zip()

list1 = data.Amount[0:10]
list2 = data.V2[0:10]

z =zip(list1,list2)
print(z)
z_list = list(z)
print("ziplist:",z_list)
print("")

# unzip

un_zip = zip(*z_list)
un_list1,un_list2 =list(un_zip)
print("unlist1:",list(un_list1))
print("")
print("unlist2:",list(un_list2)) # it unzip as a tuple so we change the tuple as a list


# In[ ]:


#LIST COMPREHENSION

num1 = data.Amount.head(10)
num2 =[int(i + 1) for i in num1] #list of comprehension
print(num2)





# In[ ]:


#conditionals on iterable

num1 = [5,10,15]
num2 =[i**2 if i == 10 else i-5 if i<7 else i+5 for i in num1]
print(num2)


# In[ ]:


#example with csv data



sum1 = data['Amount'].sum()

threshold = sum1/len(data.Amount)

data["amount_level"] = ["high" if i > threshold else "low" for i in data.Amount]
data.loc[:10,["amount_level","Amount"]] 
data.head()



# In[ ]:


num1 = data.Amount.head(10)

num2 = [int(i**2) if i % 2 == 0 else int(i-i)  for i in num1]

print(num2)

