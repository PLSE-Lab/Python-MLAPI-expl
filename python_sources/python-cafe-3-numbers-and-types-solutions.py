#!/usr/bin/env python
# coding: utf-8

#  # Cafe 3 - Numbers and Type Casting 

# #### Before we start, let's go over one more print function called `format()` that will help you format your output. 

# In[ ]:


# Your code goes here

numApples = "4"

print("I will buy {} apples today".format(numApples))

numOranges = 7

print("I will buy {} apples today and {} oranges tomorrow".format(numApples,numOranges))


# # 1. Numerical Operations
# 
# There are many operations that are built-in to python so that you can perform calculations on python types such as integers and floats. Have a look at this list as a refresher for python arithmetic operations. 
# 
# 
# | Operator     | Name           | Description                                            |
# |--------------|----------------|--------------------------------------------------------|
# | ``a + b``    | Addition       | Sum of ``a`` and ``b``                                 |
# | ``a - b``    | Subtraction    | Difference of ``a`` and ``b``                          |
# | ``a * b``    | Multiplication | Product of ``a`` and ``b``                             |
# | ``a / b``    | True division  | Quotient of ``a`` and ``b``                            |
# | ``a // b``   | Floor division | Quotient of ``a`` and ``b``, removing fractional parts |
# | ``a % b``    | Modulus        | Integer remainder after division of ``a`` by ``b``     |
# | ``a ** b``   | Exponentiation | ``a`` raised to the power of ``b``                     |
# | ``-a``       | Negation       | The negative of ``a``                                  |
# 
# <span style="display:none"></span>
# Let's try some operations! 
# 
# 

# In[ ]:


#Try multiplying Ints
print(5*5)

#Try multiplying strings
s1 = 'Ha'
print(s1*10)

#Try dividing
print(90 / 3)

#Try modulus
print(91 % 3)
print(92 % 3)

#Try an exponentiation
a = 5
b = 2
print(a**b)


# # 2. Checking Types
# 
# We played around with printing and adding different python types together. Now we are going to focus on checking what type a variable is. In some cases you may not know the type of the data you recieve, and this is one way to find out. 
# 
# To check the type of a variable or value in python we use the `type()` function
# 
# Let's try an example!

# In[ ]:


# Your code goes here

iAmString = "I am type string"
iAmAlsoString = "2" 
iAmInt = 2
iAmFloat = 2.0

print("STRING:", iAmString,type(iAmString))
print("STRING:", iAmAlsoString,type(iAmAlsoString))
print("INT:", iAmInt,type(iAmInt))
print("FLOAT:", iAmFloat, type(iAmFloat))


# # 3. Casting Types
# 
# We played around with printing and adding different python types together. Now we are going to focus on changing the types of variables i.e. "casting" a string to be an int and vice versa. 
# 
# To change or "cast" the type of a variable or value in python we will be using the `str()` , `int()` , or `float()` functions.
# 
# Let's try an example! 

# In[ ]:


# Your code goes here

iAmString = "27"

print(type(iAmString))
print(iAmString)

changedToInt = int(iAmString)

print(type(changedToInt))
print(changedToInt)

changedToFloat = float(changedToInt)

print(type(changedToFloat))
print(changedToFloat)


# # Putting it all together
# 
# Candy-sharing friends Alice, Bob and Carol tried to split candies evenly. For the sake of their friendship, any candies left over would be smashed. For example, if they collectively bring home 91 candies, they'll take 30 each and smash 1.
# 
# 1. Create a variable called `numberOfCandies` that holds the total numbers of candies brought home by Alice, Bob and Carol (can be any number).
# 
# 
# 2. Create a variable called `smashedCandies` calculates how many candies they need to smash (what is left over after they divide evenly?)
# 
# HINT: Use one of python's numerical operators. 
# 
# 3. Print the value of `smashedCandies` and the type of `smashedCandies`
# 
# HINT: Use print() and type() function
# 4. Convert `smashedCandies` to a string and print the type
# 
# 
# 
# 

# In[ ]:


####### Your Code Goes Under Here ##########

numberOfCandies = 77

smashedCandies = numberOfCandies % 3

print(smashedCandies)
print(type(smashedCandies))

smashedCandies = str(smashedCandies)

print(type(smashedCandies))

