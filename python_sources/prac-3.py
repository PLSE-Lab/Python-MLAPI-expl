#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# This algorithm takes a sequence (S) in attempt to find the higest value 
# def maxelement
# if Start == length of S then         base case
#     return S[start]
# else 
#     max=maxelement(S,start+1)   to count upwards
#     if S[start] > max then
#            return S[start]
#     else 
#         return max

# Q2
# 1.  power(2,5)  returns 2*16
# 2. power(2,4)  returns 2*8
# 3. power(2,3)  returns 2*4
# 4. power(2,2)  returns 2*2
# 5. power(2,1)    returns 2*1
# 6. power(2,0)    returns 1

# Q3
# 1. power(2,18)  returns 512*512
# 2. power(2,9)  returns (16*16) *2 =512
# 3. power(2,4)  returns 4*4=16
# 4. power(2,2)  returns 2*2=4
# 5. power(2,1)  returns (1*1) *2 =2
# 6. power(2,0)  returns 1

# Q4. 
# def multiply
# if n==1 then 
#     return m
#  else 
#      return m+multiply(m,n-1)
# 
# 

# In[ ]:


import sys 
from time import time 
from dynamic_array import DynamicArray

try:
    maxN=int(sys.argv[1])
except:
    manN=10000000
    
form time import time

def average(n):
    data=DynamicArray()
    start=time()
    for k in range(n):
        data.append(None)
    enf=time()
    return (end-start)

n=10 
while n<=maxN
    print(average(n), n)
    n*=10


# In[ ]:


class DynamicArrayWResize (DynamicArray):
    def __init__(self, resize factor):
        super.().__init__()
        self._resize_factor=resize_factor
        
    def append (self,obj):
        if self._n==self._capacity
            self.resize(int(self.resize_factor*self_capacity)+1)
        self._A(self._n)=obj
        self._n+=1
        

