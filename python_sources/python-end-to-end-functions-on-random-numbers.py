#!/usr/bin/env python
# coding: utf-8

# ***Hello Everyone! Welcome to the world of Data Science :)***
# 
# **If you like this kernel, Please do give an upvote :)**
# 
# This kernel walks through the Random number generation in python which is useful for many operations in data science.
# 
# **Random Module: **
# 
# The Python random module functions depend on a pseudo-random number generator function random(), which generates the float number between 0.0 and 1.0.
# 
# There are different types of functions used in a random module . Lets get into them.

# 1)**random.random():** Generates a random float number between 0.0 to 1.0. The function doesn't need any arguments.

# In[ ]:


import random
print ("A random number between 0 and 1 is : ", end="") 
print (random.random()) 


# This function generates a new random number each time you run it. If you want the same random number to get generated
# each time you the code ,then you can **seed** them.
# 
# **2)random.seed()**:The seed() method is used to initialize the random number generator.
# The random number generator needs a number to start with (a seed value), to be able to generate a random number.If you use the same seed value twice you will get the same random number twice.
# 

# In[ ]:


print("Random number intializing a seed value : ")
random.seed(10)
print(random.random())

random.seed(10)
print(random.random())

print("Random number without intializing a seed value : ")
print(random.random())


# **3)random.randint():**The randint() method returns an integer number selected element from the specified range.This method is an alias for randrange(start, stop+1).
# 

# In[ ]:


rn=random.randint(1,10)
print("The random integer between the specified range is : ",rn)


# **4)random.randrange():** The randrange() method returns a randomly selected element from the specified range.

# In[ ]:


print ("A random number from range is : ",end="") 
print (random.randrange(1,10))


# **5)random.choice():**This method returns a randomly selected element from the specified sequence.The sequence can be a string, a range, a list, a tuple or any other kind of sequence.

# In[ ]:


print ("A random number from list is : ",end="") 
print (random.choice([1, 4, 8, 10, 3])) 
# String manipulation:
x = "WELCOME"
print("Random character from the given string : ",random.choice(x))


# **6) random.sample(): **This method returns a list with a randomly selection of a specified number of items from a sequnce.

# In[ ]:


mylist = ["apple", "banana", "cherry"]
print("The randomly selected items are:")
print(random.sample(mylist, k=2))

string = "PythonProgramming"
print("With string:", random.sample(string, 4))

# Prints list of random items of length 4 from the given tuple. 
tuple1 = ("ankit", "geeks", "computer", "science", 
                   "portal", "scientist", "btech") 
print("With tuple:", random.sample(tuple1, 4)) 
  
#Prints list of random items of length 3 from the given set. 
set1 = {"a", "b", "c", "d", "e"} 
print("With set:", random.sample(set1, 3))


# **7) random.shuffle():** This method takes a sequence (list, string, or tuple) and reorganize the order of the items.

# In[ ]:


li = [1, 4, 5, 10, 2]
print("The list after shuffling :")
random.shuffle(li)
print(li)

mylist = ["apple", "banana", "cherry"]
random.shuffle(mylist)
print("mylist after shuffling : ")
print(mylist)

from random import shuffle
x = [i for i in range(10)]
random.shuffle(x)
print("x :",x)


# 8) **random.uniform(): ** This method returns a random floating number between the two specified numbers (both included).

# In[ ]:


print(random.uniform(20, 60))


# ## Random numbers using numpy
# **1D Array :**

# In[ ]:


import numpy as np 
   
# 1D Array 
array = np.random.randn(5) 
print("1D Array filled with random values : \n", array)


# **2D Array **  
# 
# **np.random.randn**(no. of rows,no. of columns)

# In[ ]:


array = np.random.randn(3, 4) 
print("2D Array filled with random values : \n", array);


# **3D Array  **
# 
# **np.random.randn**(no.of matrix,no. of rows,no. of columns)

# In[ ]:


array = np.random.randn(2, 2 ,2) 
print("3D Array filled with random values : \n", array); 


# # Discrete random variables using Scipy:

# In[ ]:


from scipy.stats import rv_discrete
values = [10, 20, 30]
probabilities = [0.2, 0.5, 0.3]
distrib = rv_discrete(values=(values, probabilities))
distrib.rvs(size=10)

