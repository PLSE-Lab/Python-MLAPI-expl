#!/usr/bin/env python
# coding: utf-8

# # Getting started with Numpy
# ## 1. Create 1-d array with random values in [0, n]

# In[ ]:


import numpy as np
array = np.arange(10)
array


# ## 2. Create 2 dimensional array

# In[ ]:


# create 2 dimensional array with random values = [0,25]
array2 = np.arange(25).reshape(5,5)
array2


# ## 3. Create 3-d array

# In[ ]:


array3d = np.arange(36).reshape(3,3,4)
array3d


# ## 4. Create numpy array with zeros/ones

# In[ ]:


np.zeros((3,4))


# In[ ]:


np.ones((3,4))


# In[ ]:


np.empty((2,3))


# In[ ]:


np.full((2,2), 4)


# ## 5. Create an array with evenly spaced values over a specified interval

# In[ ]:


np.linspace(0, 10, num=5)


# In[ ]:


np.linspace(0, 10, num=6)


# ## 6. Convert to array from list/tuple

# In[ ]:


array = np.array([(1,2,3), (4,5,6)])
array


# In[ ]:


my_num = [0,3,5,6,8]
np.array([my_num, my_num])


# ## 7. Use special library functions

# In[ ]:


np.random.random((2,2))


# In[ ]:


np.random.randint(low=5, high=10, size=8)


# In[ ]:


# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.uniform.html
# create an array with random numbers from a uniform distribution
np.random.uniform(low = 0, high = 1, size=5)


# In[ ]:


# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
# create an array with random numbers from a normal distribution
np.random.normal(loc=0.0, scale=1.0, size=4)


# # How to efficiently deal with large Numpy arrays
# ## 1. Use pre-allocated arrays to boost the process

# ### DON'T
# Don't iterate over each entry and add it to the array one by one like this:

# In[ ]:


import time
start_time = time.time()

entries = range(1000000) # 1 million entries
results = np.array((0,)) # empty array

for entry in entries:
  processed_entry = entry + 5 # do something
  np.append(results, [processed_entry])
    
elapsed_time = time.time() - start_time
elapsed_time


# The problem here is, that python needs to make room in the memory again and again for each append, and this is very time consuming.
# 
# ### DO
# Instead preallocate an array using np.zeros

# In[ ]:


start_time = time.time()

entries = range(1000000) # 1 million entries
results = np.zeros((len(entries),)) # prefilled array

for idx, entry in enumerate(entries):
  processed_entry = entry + 5 # do something
  results[idx] = processed_entry
    
elapsed_time = time.time() - start_time
elapsed_time


# Can you see the significant difference between these two approaches. This finishes in under 0.22 second, because the array is already sitting in the memory in its full size.
# 
# You can even do that if you don't know the final array size beforehand: you can resize the array in chunks with np.resize, which will still be much faster than the other approach.

# ## 2. Use h5py to save your RAM
# ### DON'T
# Sometimes your arrays get so big they wont fit into ram anymore. Execute the following code and your RAM is just gone.

# In[ ]:


# results = np.ones((1000,1000,1000,5))
results = np.ones((500,500,500,5)) # this one eats out 4GB of RAM

# do something...
results[100, 25, 1, 4] = 42


# ### DO
# Obviously that's something to avoid. We need to somehow store these data on our disk, instead of the ram. So h5py is here to rescue

# In[ ]:


import h5py

hdf5_store = h5py.File("./cache.hdf5", "a")
results = hdf5_store.create_dataset("results", (500,500,500,5), compression="gzip")

# do something...
results[100, 25, 1, 4] = 42


# This creates a file cache.hdf5 which will contain the data. create_dataset gets us an object that we can treat just like a numpy array (at least most of the time). Additionally we get a file that contains this array and that we can access from other scripts

# In[ ]:


hdf5_store = h5py.File("./cache.hdf5", "r")

print(hdf5_store["results"][100, 25, 1, 4]) # 42.0


# ## 3. Dont access arrays more often than necessary
# ### DON'T
# This one should be obvious, but I still see it sometimes. You need the value from some entry of an array to loop over something else:

# In[ ]:


start = time.time()

some_array = np.ones((100, 200, 300))

for _ in range(10000000):
    some_array[50, 12, 199] # get some value some_array

runtime = time.time() - start
runtime


# Even though numpy is really fast in accessing even big arrays by index, it still needs some time for it, which gets quiet expensive in big loops.

# ### DO
# By simply moving the array access outside the loop you can gain a significant improvement

# In[ ]:


start = time.time()

some_array = np.ones((100, 200, 300))

the_value_I_need = some_array[50, 12, 199] # access some_array

for _ in range(10000000):
    the_value_I_need
    
runtime = time.time() - start
runtime


# This runs 3 times faster than the previous version. Most of the times it's simple things like this that slow everything down!

# Reference:
# - http://chrisschell.de/2018/02/01/how-to-efficiently-deal-with-huge-Numpy-arrays.html
# - Numpy documentation
