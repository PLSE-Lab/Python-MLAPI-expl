#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Practical 3.
#Program to find the max element in a sequence of data.
#Question 1:
Def max(S, start):
	If start == len[S]:
		Return start
	Else:
		m = max(S, start + 1)
		If S[start] > m:
			Return S[start]
		Else:
			Return m


# In[ ]:


#Question 2:
#power(2,5)=   2*16 = 32
#    power(2,4)=   2*8 = 16
#        power(2,3)=   2*4 = 8
#            power(2,2)=   2*2 = 4
#                power(2,1)=   2*1 = 2
#                    power(2,0)=   1   
                    
                    


# Question 3:
# power(2,18)=  512*512 = 262144
#    power(2,9)=   (16*16)*2 = 512   
#        power(2,4)=   4*4 = 16
#           power(2,2)=   2*2 = 4
#               power(2,1)=   (1*1)*2 = 2
#                   power(2,0)=   1  
# 

# In[ ]:


#Question 4

def addsub(m, n):
    if n == 1:
        return m
    else:
        return m + addsub(m, n-1)
    
    
print(addsub(5, 3))


# In[ ]:


#Question 5
try:
    maxN = int(sys.argv[1])
except:
    maxN = 10000000

from time import time
from dynamic_array import DynamicArray

def compute_average(n):
  """Perform n appends to an empty list and return average time elapsed."""
  data = DynamicArray()
  start = time()                 # record the start time (in seconds)
  for k in range(n):
    data.append(None)
  end = time()                   # record the end time (in seconds)
  return (end - start) / n       # compute average per operation

n = 10
while n <= maxN:
  print('Average of {0:.3f} for n {1}'.format(compute_average(n)*1000000, n))
  n *= 10


# In[ ]:


#question 6

