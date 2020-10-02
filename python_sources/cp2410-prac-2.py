#!/usr/bin/env python
# coding: utf-8

# n^10
# 2log(n)
# 3n+100log(n) 
# 4n
# n*log(n)
# 4n*log(n)+2n
# n^2+10n
# n^3
# 2^n

# A: 8n*log(n)
# B: 2n^2
# 
# N0 = (8n*log(n) = 2n^2
# 4n*log(n) = n^2

# In[15]:


def a(n):
    return 8 * n * math.log(n)
    
def b(n):
    return 2 * (2**n)

for i in range(1,100):
    if a(i) < b(i):
        print(i)
        break


# In[ ]:


# O(n)
def example1(S):
 """Return the sum of the elements in sequence S."""
 n = len(S)
 total = 0
 for j in range(n): # loop from 0 to n-1
 total += S[j]
 return total

#O(n)
def example2(S):
 """Return the sum of the elements with even index in sequence S."""
 n = len(S)
 total = 0
 for j in range(0, n, 2): # note the increment of 2
 total += S[j]
 return total

#O(n^2)
def example3(S):
 """Return the sum of the prefix sums of sequence S."""
 n = len(S)
 total = 0
 for j in range(n): # loop from 0 to n-1
 for k in range(1 + j): # loop from 0 to j
 total += S[k]
 return total

#O(n)
def example4(S):
 """Return the sum of the prefix sums of sequence S."""
 n = len(S)
 prefix = 0
 total = 0
 for j in range(n):
 prefix += S[j]
 total += prefix
 return total

#len of A and B are the same therefore can treat the same
#Loop n twice O(2n)
# O(n^3)
def example5(A, B): # assume that A and B have equal length
 """Return the number of elements in B equal to the sum of prefix sums in A."""
 n = len(A)
 count = 0
 for i in range(n): # loop from 0 to n-1
 total = 0
 for j in range(n): # loop from 0 to n-1
     for k in range(1 + j): # loop from 0 to j
 total += A[k]
 if B[i] == total:
 count += 1
 return count

