#!/usr/bin/env python
# coding: utf-8

# # Numpy Basics
# 
# #### Here is the most fun stuff to learn Numpy!
# #### Believe me as I say, its mostly fun
# ###### Documentation:
# https://numpy.org/doc/stable/
# 
# #### If you like the notebook please don't forget to hit the upvote button

# ##### If you are planning to learn ML, first step is Numpy, so grab a coffee we are here together to learn this awesome library
# ###### Lets start with very basic definitions, don't worry I am bored too, just hang with me we'll get through this together. xD
# * Numpy stands for Numerical Python
# * Numpy is most widely used library for numeric computations
# * Numpy arrays are used to store the data of same type 
# * Numpy is highly efficient and fast in computations
# * *ndarray*: n-dimensional array of same type element.
# 

# #### Okay, lets begin the fun
# ### Initialization, dimensions, shape, etc...

# #### Oh God, I am forgetting something! What? What? What?
# #### Warning: Danger ahead!!! Import Numpy library or else be ready to be blown away

# In[ ]:


# Just as you don't like typing too much, I am also lazy af :), "as np" would allow me to use np instead of numpy everytime in rest of the code
# I recommend you to do so... Its so fun believe me xD
import numpy as np


# In[ ]:


# Lets initialize an array, keeping it simple
arr = np.array([1,2,3,4])
print(arr)
print(type(arr))  # type() is an inbuilt function to check type
print("See I don't lie xD")
print("Array initialized, type checked, its numpy's ndarray")


# In[ ]:


arr2d = np.array([[1.1,2.2,3.3,4.4],[5.5,6.6,7.7,8.8]])
print(arr2d)
print(type(arr2d))
print("This one is a 2d array, using float types, saw those extra brackets? Don't forget!!!")


# In[ ]:


# Okay let's find out how fare is python and numpy... I wanna know dimension and data types of arrays
print(arr.ndim)
print(arr2d.ndim)
print(arr.dtype)
print(arr2d.dtype)
print(arr.itemsize)
print(arr2d.itemsize) # Bonus - itemsize - size of each time in bytes


# In[ ]:


# Finally lets check size of arr and arr2d and shapes
print(arr.size)
print(arr2d.size) #count of elements
print(arr.shape)
print(arr2d.shape) #shape of array in form (n,m)


# ##### What is this (n,)? Where the hell is m?
# Well, if m=1, numpy doesn't show it. Maybe I don't like it, Numpy do something please :( . It is called rank-1 array

# In[ ]:


# Lets see something fascinating
# It is called slicing ;p

print(arr[2:3])
print(arr2d[0:1,2:4])
print("Well! Well! Can't exlplain all of it.. Refer to documentations! Sorry! :( ")


# ### Initialization
# ##### What? Again?
# ##### No! Not that boring stuff again!

# In[ ]:


# Initializing all zeros array
arr01d = np.zeros(5) # you can pass (5,1) if you don't want a rank-1 matrix
arr02d = np.zeros((5,2)) # 5-rows, 2-columns
print(arr01d)
print(arr02d)


# In[ ]:


# Initializing all ones array
# So simple, Huh?
arr11d = np.ones(5) # you can pass (5,1) if you don't want a rank-1 matrix
arr12d = np.ones((5,2)) # 5-rows, 2-columns
print(arr11d)
print(arr12d)


# In[ ]:


# Okay enough with 1s and 0s, I want my lucky number 7
arr01d = np.full((5,1),7) #5x1 is fine I guess
print(arr01d)
print("Please tell me you can do for 2d and other dimensions yourself!:)")


# In[ ]:


# Okay here is some fun stuff finally
# I wanna go random
arr_rand = np.random.rand(4,2)
print(arr_rand)
arr_rand = np.random.random_sample(arr12d.shape) # Shape as arr12d
print("Sampled from previous array")
print(arr_rand)
print("So fun!!!")


# In[ ]:


#check It out - Its an identity
np.identity(3)


# In[ ]:


# Okay let's see how much you can grasp yourself, tell me what this function does
np.repeat(arr_rand, 2, axis=1) # axis=1 is for coulumns and axis=0 is for rows


# ###### Danger Zone Ahead
# 

# ##### Me: How do you think an array can be copied to another
# ##### You: Simple ,arr1 = arr2
# ##### *Results into random garbage*

# In[ ]:


# simply puuting arr1 = arr2 makes arr1 point to arr2 which means all changes are reflected in both
# Got it? If yes: congratulations, if no, don;t worry we are heer to help

arr_copy = arr12d.copy()
print("This is indeed simple,right?")
print(arr_copy)
print("And it doesn't have the problem mentioned above")


# ## Broadcasting

# #### Broadcasting is one of the most important concept to be understood
# Pyhton is so cool in the way it provides broadcasting
# Let's see how?

# In[ ]:


# Lets define a new array first
arr_bd = np.array([[1,2,3,4],[5,6,7,8]])
print(arr_bd)


# In[ ]:


# Let's say we want to add 5 to all numbers of array, its way too simple than you would have thought
arr_bd = arr_bd+5
print(arr_bd)
print("Done so quickly with so much ease or as in hindi 'Bade aaram se'!!!xD")


# In[ ]:


# Same is the case for subtraction, multiply and divide
# a**2 is exponentiation in python
arr_bd = np.array([[1,2,1,2],[2,3,2,3]])
arr_bd = arr_bd**2
print(arr_bd)
print("Raised to power 2 so proudly :)")


# I won't do it but if you are maths lover or more spacifically trignometriy love (like meh :D), there are lot of functions waiting for you!!!
# See documentaion (link provided at the beginning)

# ## Miscelleneous
# ###### I hope I spelled it correct xD!!! I am getting bad at spelling, any advice?

# In[ ]:


# Matrix Multiplication

# Danger Zone again - Beware with matrix dimensions

# I'll d it for 2x2 matrices... I am lazy xD
mat1 = np.array([[1,2],[3,4]])
mat2 = np.array([[2,1],[4,3]])
print(np.matmul(mat1,mat2))
print("Yup its correct! I checked it... I am not that lazy too!;)")


# In[ ]:


# lets find determinant
print(np.linalg.det(mat1))


# ##### There is so much more to be checked in Linear Algebra... Not all can be explained in such a small notebook, so check it out in documentation

# In[ ]:


# Mean, median, sum, min ,max
# Small things, small space, just python rule :)
print("Max =", np.max(mat1))
print("Min =", np.min(mat1))
print("Sum =",np.sum(mat1))
print("Mean =",np.mean(mat1))
print("This is just a demo you can do all sorts of stuff...Numpy is so powerful")


# ##### Reshaping
# 

# In[ ]:


# Lets define a new array first
arr_new = np.array([[1,2,3,4],[1,2,3,4]])
# Lets see its dimensions
print(arr_new.shape)


# In[ ]:


# But I want it to be of (4,2)...Numpy says its easy
arr_new.reshape((4,2))


# ##### Bonus = np.hstack(), np.vstack() are some cool functions, check it out in documentation
# 
# ### Conditions of numpy arrays 

# In[ ]:


# Lets define a new array
arr_cond = np.array([[1,2,3,4,5,6,7,8,9,10]])
print(arr_cond)
print(arr_cond.shape)


# In[ ]:


# lets see what we get when we do these things
print(arr_cond>=5)
print("We actually get the boolean mask")


# In[ ]:


# Some more stuff
print(arr_cond[arr_cond>5])
print("Shows elements greater than 5")


# In[ ]:


print(np.any(arr_cond>10,axis=1))
print(np.any(arr_cond>8,axis=0)) 
# If any of the element matches the condition, true is returned


# In[ ]:


print(np.all(arr_cond>5,axis=1)) # still produces false because not all elements in the row are greater than 5
print(np.all(arr_cond>8,axis=0))
# If all of the element matches the condition, true is returned


# #### So I hope it was fun learning the Numpy.
# #### This is just the basic introduction of most important functions needed in daily practice
# #### If anyone wants to suggest anything please comment
# #### If you like it please hit the upvote button
