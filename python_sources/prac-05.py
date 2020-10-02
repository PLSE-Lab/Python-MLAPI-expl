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


# In[1]:


Algorithm find_second-to_last(L)
    if L.head is None or L.head.next is None:
        return None
    current = L.head
    next = L.head.next
    while next.next is not None:
        current = next
        next = next.next
    return current


# In[3]:


def count_nodes(L):
    if L.current is None:
        return 0
    count = 1
    original = L.current
    current = original
    while current.next != original:
        count += 1
    return count


# In[4]:


Algorithm same_circular_list(x,y):
    current = x 
    while current.next is not y:
        if current.next is x:
            return False
        current = current.next
        return True


# In[5]:


from positional_list import PositionalList

def list_to_positional_list(list_):
    
    pos_list = PositionalList()
    for element in list_:
        pos_list.add_last(element)
    return pos_list


# In[ ]:


def max(self):
    return max(element for element in self)

