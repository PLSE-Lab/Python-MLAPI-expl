#!/usr/bin/env python
# coding: utf-8

# Question 1: <br>
# First if statement is when a sequence has only one element
# Else, recursive call that finds the max in the sequence and returns

# In[ ]:


def MaxElement(S, start):
    if start == len(s):
        return S[start]
    else:
        max_of_rest = MaxElement(S, Start + 1)
        
        if S[start] > max_of_rest:
            return S[start]
        else:
            return max_of_rest


# **Question 2:**<br>
# 1 power(2,5)<br>
# 2 power(2,4)<br>
# 3 power(2,3)<br>
# 4 power(2,2)<br>
# 5 power(2,1)<br>
# 6 power(2,0)<br>
# 
# 1 return 2 x 16 = 32<br>
# 2 return 2 x 4 = 16<br>
# 3 return 2 x 3 = 8<br>
# 4 return 2 x 2 = 4<br>
# 5 return 2 x 1 = 2<br>
# 6 return 1<br>

# **Question 3:**<br><br>
# power(2, 18)     return 512 x 512 = 262144<br>
# power(2, 9)     return (16 x 16) x 2 = 512<br>
# power(2, 4)    return (4 x 4) = 16<br>
# power(2, 2)   return (2 x 2) = 4<br>
# power(2, 1)  return (1 x 1) x 2 = 2<br>
# power(2, 0) return 1<br>

# **Question 4:**<br><br>
# Contains a recursive case and a base case to handle two positive integers as input<br>
# 

# In[ ]:


def multiply(m,n):
    if n == 1:
        return m
    else:
        return m + Multiply(m, n -1)


# **Question 5:**

# In[ ]:


import sys
from time import time
from dynamic_array import DynamicArray
try:
 maxN = int(sys.argv[1])
except:
 maxN = 10000000
from time import time # import time function from time module
def compute_average(n):
 """Perform n appends to an empty list and return average time elapsed."""
 data = DynamicArray()
 start = time() # record the start time (in seconds)
 for k in range(n):
     data.append(None)
 end = time() # record the end time (in seconds)
 return (end - start) / n # compute average per operation
n = 10
while n <= maxN:
 print('Average of {0:.3f} for n {1}'.format(compute_average(n) * 1000000, n))
 n *= 10


# **Question 6:**

# In[ ]:


class DynamicArrayWithResizeFactor(DynamicArray):
 """ A dynamic array class which allows for a custom resize factor. """
 def __init__(self, resize_factor):
     super().__init__()
     self._resize_factor = resize_factor
 def append(self, obj):
 """ Modified version of append to use the resize factor. """
     if self._n == self._capacity: # not enough room
 # cast to int to make allow for fractional resize_factors, add 1 to make sure
 # capacity always increases
     self._resize(int(self._resize_factor * self._capacity) + 1)
     self._A[self._n]

