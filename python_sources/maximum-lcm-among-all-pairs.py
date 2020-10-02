#!/usr/bin/env python
# coding: utf-8

# This problem is taken from https://www.geeksforgeeks.org/maximum-lcm-among-all-pairs-i-j-from-the-given-array/?ref=leftbar-rightbar. 
# The task is to find the maximum LCM when the elements of the array are taken in pairs.

# In[ ]:


# Explanation
# Input: arr[] = {17, 3, 8, 6}
# Output: 136
# Explanation:
# Respective Pairs with their LCM are:
# {8, 17} has LCM 136,
# {3, 17} has LCM 51,
# {6, 17} has LCM 102,
# {3, 8} has LCM 24,
# {3, 6} has LCM 6, and
# {6, 8} has LCM 24.
# Maximum LCM among these =136.

# Input: array[] = {1, 8, 12, 9}
# Output: 72
# Explanation:
# 72 is the highest LCM among all the pairs of the given array


# Here's my approach towards the problem

# In[ ]:


# This function is to calculate the LCM of two given values at a time
def compute_lcm(x, y):

   # choose the greater number
   if x > y:
       greater = x
   else:
       greater = y

   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break
       greater += 1

   return lcm


# In[ ]:


# This function is to select a pair from the array, one by one and return the maximum LCM among them
def max_lcm(arr):
        arr = list(arr)
        n = len(arr)
        Max = []
        for i in range(n-1):
                for j in range(i+1, n+i):
                        if j<n:
                                a =compute_lcm(arr[i], arr[j])
                                Max.append(a)
        b = tuple(Max)
        return max(b)


# Let's try out out code

# In[ ]:


arr1 = (17,3,8,6)
arr2 = {1,8,12,9}
print(max_lcm(arr1))
print(max_lcm(arr2))


# Thanks for checking this out!
