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


## Dropping Duplicates from List

l=list(input("Enter the elements: ").split(","))
#print(l)
#print(len(l))
#using the list comprehension converting all string into int
l=[int(i) for i in l]
s=[]
for i in l:
    if i not in s:
        s.append(i)
print(s)


# OUTPUT OBTAINED FOR INPUT GIVEN:
# Enter the elements: 1,2,3,2,4,65,1,2,4,6
# [1, 2, 3, 4, 65, 6]

# In[ ]:


## Dropping Duplicates from List

l=[1,2,23,2,1,32]
#print(l)
#print(len(l))
#using the list comprehension converting all string into int
s=[]
for i in l:
    if i not in s:
        s.append(i)
print(s)


# In[ ]:


#ascending order without sort fn.
l=list(input("Enter elements: ").split(","))
l=[int(i) for i in l]
print(l)
for i in range(0,len(l)):
    for j in range(i+1,len(l)):
        if(l[i]>l[j]):
            t=l[j]
            l[j]=l[i]
            l[i]=t
print(l)


# OUTPUT OBTAINED FOR GIVEN INPUT:
# Enter elements: 23,56,78,22,34,1
# [23, 56, 78, 22, 34, 1]
# [1, 22, 23, 34, 56, 78]

# In[ ]:


#ascending order without sort fn.

l=[23,79,1,100,52]
print(l)
for i in range(0,len(l)):
    for j in range(i+1,len(l)):
        if(l[i]>l[j]):
            t=l[j]
            l[j]=l[i]
            l[i]=t
print(l)


# In[ ]:


#check if a string is a palindrome
def palin(str1):
    if(len(str1)==0):
        return True
    elif(len(str1)>0):
        r=str1[::-1]
        if(r==str1):
            return True
        else:
            return False
    

str1=input("Enter the string: ")
n=palin(str1)
print(n)


# OUTPUT OBTAINED FOR GIVEN INPUT:
# Enter the string: MALAYALAM
# True

# In[ ]:


#check if a string is a palindrome
def palin(str1):
    if(len(str1)==0):
        return True
    elif(len(str1)>0):
        r=str1[::-1]
        if(r==str1):
            return True
        else:
            return False
    

str1="MALAYALAM"
n=palin(str1)
print(n)


# 1.4 What do you understand by Data science?
#  
#  Data Science is a intersection of Business,Computer Science/Information Technology,Maths and Statistics fields where Machine Learning algorithms,software development methods etc. are applied on data of many kinds such as unstructured data(gif's),structured data(tables data),semistructured data(csv file data) etc..to predict the future trends across various industries that would be a great aid inorder to make decisions and plan their strategies accordingly.  
#  

# 1.5 What is machine learning?
# 
# Machine Learning can be referred to a process where in the machine(computer system) learns to execute any task based on its previous learnings which are in the form of patterns,inferences without the support of any precise instructions.
# There are various kinds of Machine Learning:
# 1. Supervised Learning--model is trained with output
#     * Classification:
#         output is categorical
#         e.g.: based on the weight of different coins their value can be predicted
#     * Regression
#     
# 2. Unsupervised Learning--model is trained with pattern
#     * Clustering 
#     * Associaton
# 3. Reinforcement Learning
# 4. Semi-supervised Learning
# 

# 1.6 What are different types of distributions in probability?
# 
# The following are the different types of distribution
# 1. Bernouli Distribution
# 2. Bionomial Distribution
# 3. Normal Distribution
# 4. Exponential Distribution
# 5. Poisson Distribution
# 6. Uniform Distribution
# 
# 

# 1.7 What are the difference between skewness and kurtosis?
# Both can be observed only in numeric type of data.
# 
# **SKEWNESS:** It basically means distortion.
#               It is caused due to asymmetry
#             **2 types of skewness**
#           * Positive Skewness(Right Skewness)
#               can be eliminated using:
#               1. Square root transformation--can be applied on only positive values.
#               2. Cube root transformation-- can be applied to -ve and zero values and is weaker than log trans..
#               3. Logarithmic transformation
#           * Negative Skewness(Left Skewness)
#               **can be eliminated using all of the above ones the additional methods:**
#               1. Square transformation
#               2. FindingOutliers--using outliers() fn.(return values at extreme distances from mean)in Outliers pkg
#           
#           
#           
# **KURTOSIS:** It basically means distribution
#               It is caused to due to the tailedness in the curve
#           **3 types of kurtosis**
#           * Leptokurtic(center in the curve is narrow)
#           * Mesokurtic(normal distribution of the curve is witnessed)
#           * Platykurtic(the middle part in the curve is a bit flat)
#           
# **The following methods are used to eliminate both skewness and kurtosis:**
# 1. Standardization--Scaling data should have a mean 0 and a Standard Deviation of 1
# 2. Normalisation--data rescaling is done in a range(min-max,[0,1],etc..)
# 3. Feature Scaling--Similar to Standardization but it is used to adjust all the units of all data available into a uniform (same) measurement for ex:kg,gm into either kg,kg or gm,gm.
# 
# 

# 1.8 What are different data types in python?
#  The following are the different data types:
#  
#  Primitive datatypes--contain pure,simple values of data, building blocks for data manipulation,can't be further divided.
#  
#  Strings-to store text data-(str)
#  Integer-to store numbers-(int)
#  Float-to store float-(float)
#  complex-to store complex(complex)
#  Boolean-True and False are 2 values
#  
#  Non-primitive--don't just store a value,but a collection of values in various formats,can be further divided.
#  
#  Arrays
#  List--Mutuable
#  Dictionaries--Stores data in key-value pairs.
#  Sets(Immutable)-- As there are no indexing elements can be printed in an random order
#  Frozen Sets--the elements cannot be added. 
#  Tuple--Immutable
#  
# 
