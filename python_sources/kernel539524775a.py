#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from positional_list import PositionalList
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Algorithm find_second_last(G):
#     if g.head() = null
#         return none    
#     m = G.head()
#     n = G.head.next()
#     while n.next != none
#         m = n;
#         n = n.next
#     return m;

# In[ ]:


def num_nodes(C):
    if C.current is none:
        return 0
    start = C.current
    count = 1
    next = start.next
    while next != start:
        next = next.next
        count +=1
    return count


# In[ ]:


def is_in_list(C, L, x, y):
    start = x
    while start.next != x:
        if start = y:
            return true
        start = start.next
    return false


# In[ ]:


def list_to_PList(L):
    P = PositionalList()
    n = L.head
    for element in L:
        P.add_last(L)
        n = L.next
    return P


# In[ ]:


def highest_int(P):
    for element in P:
        return max(element)

