#!/usr/bin/env python
# coding: utf-8

# # Big-O Notation, Code Complexity and Some Python Basics

# ## Table of Contents
# * [1. Loading Libraries](#1) <br>
# * [2. Algorithm and Code Complexity](#2) <br>
# * [3. Big-O Notattion](#3) <br>
# * [4. Big-O / Omega / Teta](#4) <br>
# * [5. Big-O Examples](#5) <br>
#     * [5.1. Constant Big-O](#6) <br>
#     * [5.2. Linear Big-O](#7) <br>
#     * [5.3. Cubic Big-O](#8) <br>
# * [6. Calculating scale of Big-O](#9) <br>
# * [7. Some Python Basics](#10) <br>
#     * [7.1. What is the difference between deep copy and shallow copy?](#11) <br>
#     * [7.2. What is the difference between list and tuple?](#12) <br>
#     * [7.3. What is the difference between help() and dir() functions](#13) <br>
#     * [7.4. What is negative index?](#14) <br>
#     * [7.5. What is *args and **kwargs?](#15) <br>
#     * [7.6. How to make a list random?](#16) <br>
#     * [7.7. How to use join() and split()](#17) <br>
#     * [7.8. Leading and Trailing Whitespace](#18) <br>
#     * [7.9. How to convert text lowercase and uppercase?](#19) <br>
#     * [7.10. Pass, Break and Continiue](#20) <br>   
# 

# <a id="1"></a>
# ## 1.Loading Libraries

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


# <a id="2"></a>
# ## 2.Algorithm and Code complexity

# * Algorithm : It is kind of a formula aiming to solve a problem.
# * Code complexity : Is a software metric being used to understand the complexity of algorithm
# * If the problem is to sort the numbers in a series, then the solution formula can be named as sorting algorithm.
# * There are several sorting algo --> Buble sort and selection sort algorithms
# * Question is : Which algorithm should i use for a sorting problem?
# * Lets make an example for better understanding:
# * I have a formula that calculates the sum of squares of numbers from 1 to n :

# ![image.png](attachment:image.png)

# * Lets code two different algorithms for this purpose:

# First algorithm: Create using for loop

# In[ ]:


def square_sum1(n):
    """
    Takes an input n and return sum of the squares of numbers from 0 to n
    
    """
    sum_ = 0
    for x in range(0,n+1):
        sum_ = sum_ + x**2
    
    return sum_


# In[ ]:


square_sum1(3)


# Second algorithm : Create using formula

# In[ ]:


def square_sum2(n):
    result = n*(n+1)*(2*n+1)/6
    return result


# In[ ]:


square_sum2(3)


# * So i have created 2 different algorithms, each gives same results
# * Which algoritm should i use then?
# * I can compare the calculation time of each algorithm and pick the faster one.

# In[ ]:


get_ipython().run_line_magic('timeit', 'square_sum1(3)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'square_sum2(3)')


# * As we can see, the calculation time of second algorithm is shorter than the first one.
# * We can quikly say than second one shoul be prefered.
# * However, calculation time may not be qood to get a decision because it is also dependent on hardware of PC.
# * So, in general, Big O notation is being used on comparison of algorithms

# <a id="3"></a>
# ## 3.Big-O Notation

# * Big-O defines a functions complexiy or bigerness.
# * Big-O analysis is also named as asympotic analysis.
# * Some of Big-O function as below:
# * As type of input changes from costant(can be any number) through Exponential, Big-O complexity increases and calculation time increases.
# * So, being in green area in Big-O complexity chart in coding, results most time effective and less time consuming softwares. 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# <a id="4"></a>
# ## 4.Big-O / Omega / Teta

# * Big-O : Testing the code complexity in the worst case scenario
# * Omega : Testing the code complexity in the best case scenario
# * Tetam : Testing the code complexity in the mid case scenario
# * Example : Let's say we have a list containing 2,3,4 numbers in random order.
# * Our aim is to find the Big-O rate of the algoritm that searches and find "2" and gives me the index of it.
# * There are three possibility of order for the list: "4,3,2" or "3,2,4" or "2,3,4".
# * So, algoritm will find 2 quickly in 3rd possibility; in a mid time in 2nd possibility; latest in 1st possibility.
# * Then, we can say that Big-O rate of this algoritm is 2 seconds, becasue it is 2 seconds according to the worst case scenario. 
# 

# ![image.png](attachment:image.png)

# <a id="5"></a>
# ## 5.Big-O Examples

# <a id="6"></a>
# ## 5.1. Constant Big-O / O(1)

# * In this type, time complexity is not dependent on input size.
# * Increasing or decreasing input size does not change output time complexity.
# * In below example, Big-O is contant for two lists because algorithm makes one operation for each.

# In[ ]:


def find_first_index(list1):
    print(list1[0])
    
print("First index of [1,2,3] is :")
find_first_index([1,2,3])

print("First index of [2,3,4,5,6,7,8,9,10] is :")
find_first_index([2,3,4,5,6,7,8,9,10])


# <a id="7"></a>
# ## 5.2. Linear Big-O / O(n)

# * In this type, time complexity increases linearly with input size.
# * In below example: if list size is 2, then values_of_list function will call print function 2 times.
# * If list size is 5, then values_of_list function will call print function 5 times.
# * So we can say that, Big-O of values_of_list algorithm is Linear. 
# 

# In[ ]:


def values_of_list(list1):
    for each in list1:
        print(each)

list1 = [1,2]
list2 = [3,4,5,6,7]

print("Values of list1") 
values_of_list(list1)

print("Values of list2")
values_of_list(list2)


# <a id="8"></a>
# ## 5.3. Cubic Big-O / O(n^3)

# * In this type, If input size is n, then operation time will be n^3. 
# * In below example: cubic_big_o algorithm calls 27 times print function with an input of 3 size list.
# * So time complexity of this algorithm is cubic.

# In[ ]:


def cubic_big_o(list1):
    for item1 in list1:
        for item2 in list1:
            for item3 in list1:
                print(item1, item2, item3)
                
cubic_big_o([1,2,3])
        


# <a id="9"></a>
# ## 6. Calculating Scale of Big-O

# For further reading please refer:
# 
# https://www.integralist.co.uk/posts/calculating-bigo/
# 
# https://www.integralist.co.uk/posts/algorithmic-complexity-in-python/

# In[ ]:


def linear_big_o(list1):
    for each in list1:
        print(each)
        
linear_big_o([1,2]) # Scale of this algorithm is O(n)


# * For above example: 
# * Scale of the algorithm is O(n)
# * For n size of input, gives n times of output.
# * For infinite size of input, algorithm gives infinite output.

# In[ ]:


def linear_big_o_1(list1):
    for each in list1:
        print(each)
    for each in list1:
        print(each)
        
linear_big_o_1([1,2]) # Scale of this algorithm is O(2n)


# * For above example: 
# * Scale of the algorithm is O(2n).
# * For n times of input, gives n times of output 2 times. 
# * For infinite times of output (n=infinite), gives infinite times of output, because 2*infine = infinite
# * In this exampe 2 is insignificant constant and it can be neglegable.
# * Thus , for both first and second example, scale is O(n).

# In[ ]:


def example(list1):
    print(list1[0])

example([1,2])  # scale is O(1)


# In[ ]:


def example(list1):
    print(list1[0])
    for each in list1:
        print(each)

example([1,2])  # scale is O(1+n), here is 1 is insignificant constant


# * For above example() function:
# * Scale of first one is O(1)
# * Scale of second one is O(1+n), if n is infinite then n+1 is infinite; therefore 1 is insignificant constant.
# * In this case, scale of both example() function is O(n)

# ## Examples:

# In[ ]:


n = 2
for x in range(n):
    for y in range(n):
        print("example1")


# * For above example1: Big-O performance (performance = time complexity) is O(n^2)

# In[ ]:


n = 2 
for x in range(n):
    print("example2")


# * For above example2: Big-O performance is O(n).

# In[ ]:


n = 2 
for x in range(n):
    print("1")
for x in range(n):
    print("2")
for x in range(n):
    print("3")


# * For above, Big-O performance is O(3n)
# * 3 is insignificant constant, and performance is O(n).

# <a id="10"></a>
# ## 7. Some Python Basics

# <a id="11"></a>
# ## 7.1. What is the difference between deep copy and shallow copy?

# * DEEP COPY
# * If we change the item that we deep copy, created item does not change. 
# * Let's say i have a list1
# * I created list2 by deep copying list1.
# * In this case, if something changes in list1; list2 is not being effected.
# * Deep copy example:

# In[ ]:


list1 = [[1,2,3],[4,5,6]]

import copy
list2 = copy.deepcopy(list1)
print("list2:",list2)

list1[0][2] = 100
print("list1:", list1)
print("list2 after list1 changed:", list2)


# * SHALLOW COPY
# * If we change the item that we copy, created item also being changed.
# * Let say i have a dictioanary named "family".
# * I created the dictionary named "new_family" by copying "family".
# * Then i updated "family" by adding new key "child" and value "maria".
# * I checked new_family, expecting to have "child":"maria", but i could not although we copy it shallow.
# * So, how this is a shallow copy?
# * If there is some object in dictionary like list or user defined objects instead of primitive datatype, then shallow works.
# * For example: I updated the "pets" by adding new item into its list in "family", then "new_family" also changed. 

# In[ ]:


import copy
family = {"father":"john", "mother":"linda", "pets":["cat"]}
new_family = copy.copy(family)

print("new_family =", new_family)

# adding new item in family
family["child"]="maria" 
print("new_family after adding new item Child in family =", new_family)

# adding new item in list of pets
family["pets"].append("dog")

print("new_family after adding item of pets in family", new_family)


# In[ ]:


import copy
old_list = [[1,2,3,4]]
new_list = copy.copy(old_list)
old_list[0][1]=100

print("old_list:", old_list)
print("new_list:", new_list)


# <a id="12"></a>
# ## 7.2. What is the difference between list and tuple?

# * List is changable, can be add ro remove items.
# * Tuple can not be changed.
# * In an algorithm, if you want the items that you created can not be changed by anyone, than you can use tuple.

# In[ ]:


list1=[1,2,3]
list1[1]=1000
print(list1)
tup =(1,2,3)
# tup[1]=100 # it can not be changed. 


# <a id="13"></a>
# ## 7.3. What is the difference between help() and dir() functions?

# * help() : Shows the explenation of desired function as string
# * dir() : Shows all the features / methods of desired function

# In[ ]:


# If we want to learn what copy() function does?

help(copy.copy)


# * It explains that copy.copy() is used for shallow copy and advising us to see __doc__ for more info.
# * Let us see doc string:

# In[ ]:


copy.copy.__doc__


# In[ ]:


# If we want to learn what are the attributes and methods of copy() function?
dir(copy.copy)


# <a id="14"></a>
# ## 7.4. What is negative index?

# * Index starts from 0 in normal situation.
# * For example, In a list, we can reach the items by indexing zeroth, first, second, third etc element.
# * Negative index is used to reach the items starting from the end of list not from the begining.

# In[ ]:


list1 = [1,2,3,4,5,6]
print("first item of list1:", list1[0])
print("last item of list1:", list1[-1])
print("2nd item starting from end:", list1[-2])


# <a id="15"></a>
# ## 7.5. What is *args and *kwargs?

# * Flexible arguments:
# * While creating a function, if we don't know how many or which arguments we use, we can use *args.

# In[ ]:


def print_items(*args):
    for each in args:
        print(each)

print_items(1,2,3,4,5)


# * kwargs used for dictionaries:

# In[ ]:


def print_items(**kwargs):
    for each in kwargs:
        print(each,"-->", kwargs[each])
        
print_items(a=1, b=2, c=3)


# In[ ]:


def print_items(**kwargs):
    for keys, values in kwargs.items():
        print(keys,"-->", values)
        
print_items(a=1, b=2, c=3)


# <a id="16"></a>
# ## 7.6. How do you make a list random?

# In[ ]:


list1 = [0,1,2,3,4,5,6,7,8,9,10,11,12]

from random import shuffle
shuffle(list1)  # making random
print("shuffled:", list1)
list1.sort()    # turning back to first situation
print("back to initial:", list1)


# <a id="17"></a>
# ## 7.7. How to use join() and split() ?

# In[ ]:


a = "HelloWorld"
joined = ".".join(a)
print("joined",joined)
splitted = joined.split(".")
print("splitted",splitted)
print("".join(splitted))


# <a id="18"></a>
# ## 7.8. Leading and Trailing Whitespace

# * Strings may have spaces beginning of text and end of text.
# * "      text" ---> if it has space beginning named leading whitespace
# * "text      " ---> if it has space end named trailing whitespace

# In[ ]:


string = "    trial    "
print(string.lstrip())  # deleting leading whitespace
print(string.rstrip())  # deleting trailing whitespace
print(string.strip())   # deleting both whitespaces


# <a id="19"></a>
# ## 7.9. How to convert text to lowercase and uppercase?

# In[ ]:


a = "world"
print(a.upper())
print(a.upper().lower())


# <a id="20"></a>
# ## 7.10. Pass, Break and Continue

# * Pass:
# * Let say we define a function and want to complete the definition later.
# * In this case we should close the function with "pass".

# In[ ]:


def trial():
    "Complete the definition later"
    pass


# * Break:
# * It is used in order to finalize or get out from a loop (for loop, while loop etc).**

# In[ ]:


for each in range(10):
    print(each)
    if each == 5:
        break


# * Continue:
# * In a loop, skip the current iteration but continue the next iteration.
# * The continue statement is used to skip the rest of the code inside a loop for the current iteration only. Loop does not terminate but continues on with the next iteration.

# In[ ]:


for each in range(10):
    if each % 2 == 0:
        continue   # skips rest of the code from here
    print(each, "is odd number")
        


# In[ ]:


for each in range(10):
    if each == 2:
        continue    # when each = 2, skips rest of the code, moves on to next iteration, does not print "a"
        print("a")
    print(each)       


# ## **Please upvote if you like my kernel!**
