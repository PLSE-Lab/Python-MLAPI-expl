#!/usr/bin/env python
# coding: utf-8

# Let's practice some recursion.
# We will start from easy exercies and get to hard ones.
# 
# Let's start with
# **Finding the Sum:**

# In[ ]:


import math

def sum(A):
    if not A:
        return 0
    if len(A)==1:
        return A[0]
    if len(A)>1:
        return A[0]+sum(A[1:])

print(sum([1,2,3]))


# **Finding the min:**
# 
# Now let's find the minimum, there's two versions here:
# 
# 1) basic min: O(n)
# 2) optimized version: O(log(n))

# In[ ]:


def basicMin(A, currMin):
    if not A:
        return currMin
    if currMin>A[0]:
        currMin=A[0]
    return basicMin(A[1:],currMin)

# optimized version
def findMin(A, l, r):
    if l==r:
        return A[r]
    mid = math.floor((l+r)/2)
    leftMin = findMin(A,l , mid)
    rightMin = findMin(A, mid+1, r)
    return min(leftMin, rightMin)

print( findMin([3,1,2],0,2))


# **Chceking if a string is a palindrom:**
# 
# A palindrom is a string that can be read the same from left to right or from right to left.
# 
# Examples:
# a,
# aa,
# aba,
# abba
# 

# In[ ]:


# checks if a string is a palindrom
def isPali(text):
    if len(text)== 1:
        return True
    elif len(text)==0 or text[0] != text[len(text)-1]:
        return False
    else:
        temp = text[1:-1]
        return isPali(temp)

print(  isPali('ssws') )


# **Reversing a list:**
# 
# What's new here is the temp list (called rev) that is used to store the new reversed list:

# In[ ]:


def reverseList(A,rev):
    if not A:
        return
    reverseList(A[1:],rev)
    rev.append(A[0])

rev = []
reverseList([3,2,1],rev)
print(rev)


# **Find all subsets of a set**
# 
# This is facebook interview level diffculty. so be patient with this one.
#  
#  There should be 2^n subsets to any given subset. Imagine a tree data structure, with each leave being a different subset. 
#  
#  
#  Examples:
# [1] =>      (2^1=2)    =>    [],[1]
# 
# [1,2] =>      (2^2=4)    =>      [], [1], [2], [1,2]
# 
# [1,2,3] =>     (2^3=8)    =>     [], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3] 
# 

# In[ ]:



# prints a subset without null values
def print_set(subset):
    temp = []
    for x in subset:
        if x!= None:
            temp.append(x)
    print(temp)
    
# allocate an empty subset with the right size. and call helper.
def all_subsets(given_array):
    subset = [None] * len(given_array)
    helper(given_array,subset,0)

#  Either add new item or add null, if we reached the end of the list, print the list.
def helper(given_array, subset, i):
    if i==len(given_array):
        print_set(subset)
    else:
        subset[i] = None
        helper(given_array,subset,i+1)
        subset[i] = given_array[i]
        helper(given_array,subset,i+1)

all_subsets([1,2,3])

