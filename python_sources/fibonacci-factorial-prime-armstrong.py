#!/usr/bin/env python
# coding: utf-8

# <h2><strong> Hello Programmers!!! </strong> <h2>
# <h3> This is my first step towards learning python programming and want to share my experience and codes with all programmers who are new to Python. I'll try my best to complete all programs which  are the foundation of any programming language. So, stayed tunned  because i've planned to update this kernel everyday so that we can learn something about python  everyday. 
# If you like this Kernel please <strong>vote up </strong>. <h3>

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


# 
# 

# <strong> Index </strong>
# 1.  Intoduction to python
# 2. Datatypes in python
# 3. Operators in python
# 4. Print Hello World
# 5. Add two numbers
# 6. Find square root of number
# 7. Area of triangle
# 8. Print series of number
#       * Series from 1-10
#       * Even number
#       * Odd Number
#       * Prime Number
#       * Fabonacci Series
# 9. Check whether the number is armstrong or not
# 10. Factorial of any number
#       

# <strong> Data types in Python </strong>
# 
# Python datatypes can be divided into two types primitive and non primitive. Primitive types are those which are not further divided, whereas non-primitive data can further divided or splitted.
# 
# <strong>a) Primitive Data type</strong>
# 1. Number/Integer: (signed integer, Ex- 10, 789,-28)
# 2. Float ( Decimal point real values, Ex- 15.20, -12.3)
# 3. Complex( Imageninory number. Ex- 9.322e-36j, 3.14j)
# 4. String(Characters, Ex- str = "Hello")
# 5. Boolean

# In[ ]:


a = 25
b = 15.20
d =  9.322e-36j
str = "this is a example of string"
print(type(a)) # type() is use to check the type of variable.
print(type(b))
print(type(d))
print(type(str))


# <strong>b) Non-Primitive Datatype</strong>
# 1. List
# List contains items separated by commas and enclosed within square brackets ([]). To some extent, lists are similar to arrays in C. One difference between them is that all the items belonging to a list can be of different data type.
# 
# Ex- lst = ["Abc",1,23,546,"fht"]
# 2. Tuple
# A tuple is another sequence data type that is similar to the list.  However, tuples are enclosed within parentheses. Onr more major difference between list and tuple are list is mutable whereas tuple is unmutable.
# 
# Ex- tuple = (1,2,4,5,"abc")
# 3. Dictionary
# Python's dictionaries are kind of hash table type. They work like associative arrays or hashes found in Perl and consist of key-value pairs.
# 
# Ex- dict = {"a":[1,2,3,4,5], "b": [4,5,6,7,8]}
# 4. Set
# A set is an unordered collection of items. Every element is unique (no duplicates).  Set itself is mutable. We can add or remove items from it.
# 
# Ex- set1 = {1,2,4,["a","b"]}
# 
# Sets can be used to perform mathematical set operations like union, intersection, symmetric difference etc.

# In[ ]:


lst = ["Abc",1,23,546,"fht"]
tpl = (1,2,4,5,"abc")
dict1 = {"a":[1,2,3,4,5], "b": [4,5,6,7,8]}
set1 = {1,2,4,("a","b")}
print(type(lst))
print(type(tpl))
print(type(dict1))
print(type(set1))


# <strong> Arithmetic Operators </strong>

# <strong>The following  lists the arithmetic operators supported by Python::</strong>
# 
# 
# a)   Arithmetic Operators<strong>(+, -, *, /, //, %)</strong>
# 
# b)   Comparison (Relational) Operators<strong>(<,>,<>, ==, <=, >=, !=) </strong>
# 
# c)   Assignment Operators<strong>( =, +=, -=, %=, /=, **=, *=)</strong>
# 
# d)   Logical Operators<strong>(AND, OR, NOR)</strong>
# 
# e)   Bitwise Operators<strong> (&,~,^, <<,>>)</strong>
# 
# f)   Membership Operators<strong>(in, not in)</strong>
# 
# g)   Identity Operators<strong> (is, is not) <strong>

# <strong> Print Hello world </strong>

# In[ ]:


print("Hello World")


# <strong> Add two numbers</strong>

# In[ ]:


def add(a,b):
    return a+b
add(10,3)


# <strong>Find the Squareroot of numbers </strong>

# In[ ]:


def square_root(num):
    return num**0.5
square_root(4)


# <strong>Area of triangle</strong>

# In[ ]:


def area_of_triangle(a,b,c):
    s = (a+b+c)/2 #semiperimeter of triangle
    ar = s*(s-a)*(s-b)*(s-c)
    print("Area of triangle is:",square_root(ar))
area_of_triangle(4,4,5)


# <strong>Print a certain range of number</strong>

# In[ ]:


for i in range(1,10):
    print(i, end = ",")


# <strong>Printing first 10 natural number</strong>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


for i in range(1,11):
    print(i, end = ",")
    i = i+1


# <strong>Print first 100 even number<strong>

# In[ ]:


for i in range(2,100,2):
    print(i, end = ",")


# In[ ]:


for i in range(1,100,2):
    print(i, end = ",")


# <strong>Fabonacci number</strong>

# In[ ]:


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
fib(12)


# <strong>Print prime number between 1 to 100</strong>

# In[ ]:


upper = 100
lower = 2
def print_prime_number(number):
    for number in range(lower, upper +1):
        if number > 1:
            for i in range(2, number):
                if (number % i) == 0:
                    break
            else:
                    print(number, end = ",")
print_prime_number(100)


# <strong> check whether the number is prime or not</strong>

# In[ ]:


def check_prime_number(num):
    if num > 1:
        for i in range(2, num-1):
            if (num%i) == 0:
                print(num, "is not a prime number","it is divisible by", i)
                break
        else:
                print(num,"is a prime number")
        
check_prime_number(33)


# <strong>Armstrong number</strong>
# 153 = (1*1*1 + 5*5*5 + 3*3*3)

# In[ ]:


num = 152 #you can use any number of your choice
def check_num_armstrong(num):
    sum = 0
    temp = num
    while temp > 0:
        digit = temp%10
        sum += digit**3
        temp //= 10
    if num == sum:
       print(num,"is an Armstrong number")
    else:
       print(num,"is not an Armstrong number")
check_num_armstrong(num)


# <strong> Factorial of a number </strong>

# In[ ]:


num = 5 #you can use any number of your choice
def factorial(num):
    factorial = 1
    if num == 0:
        print("factorial of 0 number is 1")
    elif num < 0:
        print("Sorry Factorial of negative number is not available")
    else:
        for i in range(1, num +1):
            factorial = factorial*i
        print("Factorial of", num, "is", factorial)
factorial(num)        
    


# In[ ]:





# In[ ]:




