#!/usr/bin/env python
# coding: utf-8

# ***In this kernel I have tried to give a brief intro for begineers (this is what I learnt) to get a feel of Numpy library. I hope it helps someone***
# 
# **I have learnt from [Pierian Data International by Jose Portilla](http://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview)**
# 
# Just to give a headsup in case you have that question in your mind.
# 
# So some credit goes to them as well, without them I wouldn't be able to create proper guidance.
# 

# In[63]:


#import Numpy library
import numpy as np


# ***Arrays***

# In[13]:


#Creating NumPy Arrays from List
my_list=[1,2,4,8]


# In[14]:


my_list


# In[17]:


#Creat arrays from the matrix
my_matrix = [[1,2,5],[8,9,6],[9,6,3]]
my_matrix


# In[18]:


np.array(my_matrix)


# **There are different Built-in Methods for Numpy Libraries that you can use to generate arrays different ways**

# In[21]:


#arange (You can generate a series from inital to final, intial included only)
np.arange(0,10)


# In[26]:


#Zeros and Ones (You can generate arrays of zeros and ones)
np.zeros(5)


# In[25]:


np.ones(3)


# In[27]:


np.ones((3,3))


# In[29]:


np.zeros ((2,2))


# In[30]:


#Linspace can be used to generate evenly spaced numbers over a particular interval

np.linspace(0,15,4)


# In[31]:


np.linspace(0,15,10)


# In[33]:


np.linspace(0,10,50)


# In[35]:


#Eye used to create an Identity Matrix

np.eye(5)


# In[36]:


#Rand from Random library which generates random number arryas which is uniform
np.random.rand(6)


# In[37]:


#You can either import random libray but for now I will be adding random library in the code
np.random.rand(3,3)


# In[38]:


#Randn returns a sample or samples form the "Standard normal" distribution.
np.random.randn(5)


# In[39]:


#Randint returns random integer from low (inclusive) to high (exclusive)
np.random.randint(1,50)


# In[40]:


#If you want a series of random integer you can add interval in the syntax.
np.random.randint(1,50,20)


# In[41]:


#"Attributes and Methods of Array"
#Gives you sorted array
np.arange(25)


# In[ ]:


#Reshape returns an array cotaining the same data with a new shape


# In[50]:


my_rand=np.random.randint(0,50,9)
my_rand


# In[51]:


my_rand.reshape(3,3)


# In[53]:


my_rand2=np.arange(25)
my_rand2


# In[54]:


my_rand2.reshape(5,5)


# In[ ]:


# "Different aggregates you can use on the arrays for computation purpose or finding the index location"


# In[56]:


#max
my_rand2.max()


# In[57]:


#min
my_rand.min()


# In[60]:


#argmax (It returns the position of the max value in the array)
my_rand2.argmax()


# In[61]:


#argmin (It returns the position of the min value in the array)
my_rand2.argmin()


# In[67]:


my_rand3=np.random.randint(0,15,10)
my_rand3


# In[69]:


#Shape gives out the shape of the array and it is an attribute that an arrays has. It is not a method
my_rand3.shape


# In[73]:


#Now lets reshape the array to something else and then again check its shape
my_rand4=my_rand3.reshape(1,10)
#It has changed to 2D array.
my_rand4


# In[74]:


my_rand4.shape


# In[75]:


#dtype returns the data type of the array
my_rand4.dtype


# **Arithmetic Operation**

# Lets us creat first some arrrays and then we will perform different arithmetic operations on it.

# In[76]:


arr1=np.arange(0,10)
arr2=np.arange(0,10)


# In[77]:


# + to add arrays
arr1+arr2


# In[78]:


# - to subtract arrays
arr1-arr2


# Similarly for * and /

# In[81]:


# Double * means you multiplying it twice. So * can be used to get squares,cubes, roots
arr3=arr1**2
arr3


# In[82]:


#sqrt to get square roots
np.sqrt(arr3)


# There are many more like (exp, sin , cos , log) etc that you can use.

# **Selection and Indexing in NumPy**

# In[83]:


arr1


# In[85]:


# [] use to get a value at particular Index
arr1[5]


# In[86]:


# [:] use to get values in a range where final is exclusive
arr1[1:5]


# In[88]:


# [:]= use to set values for a range of index
arr1[0:4]=20
arr1


# In[90]:


arr2 = np.arange(0,10)
arr2


# In[99]:


#Important thing to note here, lets say
part_of_arr1 = arr1[0:4]
part_of_arr1


# In[100]:


part_of_arr1[:]=2
part_of_arr1


# In[101]:


#Lets check arr1
arr1


# In[102]:


#In NumPy the data is not copied but rather it is the view of the original array. Thta can cause many problems.
#So to avoid it always get a copy of the original array.
arr_copied=arr1.copy()
arr_copied


# In[108]:


#Now lets see 2D indexing
arr_2d = np.array(([11,27,35],[44,59,62],[73,80,99]))
arr_2d


# In[110]:


#The indexing for 2D array has a format that is 
# Either array[row][col] or array[row,col]
arr_2d[1]


# In[111]:


arr_2d[1][2]


# In[116]:


arr_2d[1,2]


# In[118]:


#Slicing a 2d array
arr_2d[:2,1:]


# In[119]:


arr_2d[2,:]


# In[120]:


arr_2d


# In[121]:


#Comparison Operators
# >
arr_2d >45


# *You can use other different operators , go through the documentation and explore.*
# 
# **I hope this helped someone.**
# 
# ***I wish to make more such kernels to help anyone trying to learn. Let me know if you like it, that would encourage me more to make such materials in future.***
# 
# Fell free to comment out the mistakes I will appreciate it. And if you have any doubt feel free to ask, I will try to the best of my knowledge answering them and we also have great Kaggle community so other will help you too.
# 
# ***Happy Learning***
