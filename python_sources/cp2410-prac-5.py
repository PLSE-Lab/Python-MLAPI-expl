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


# Q1.	 (R-7.1) Give an algorithm for finding the second-to-last node in a singly linked list in which the last node is indicated by a next reference of None.

# In[ ]:


def find_second_tolast(list):
    if list.head is none or list.head.net is none:
        current_el = list.head
        next_el=list.head.next
        while next_el.next is not none:
            current_el = next_el
            next_el = next_el.net
        return current_el


# Q2. (R-7.5) Implement a function that counts the number of nodes in a circularly linked list.

# In[ ]:


def count_nodes(list):
    if list.current is none:
        return 0
    nodes = 1
    origin_node = list.current
    current_node = origin_node
    while current.next != origin_node:
        count += 1
        return count


# Q3. (R-7.6) Suppose that x and y are references to nodes of circularly linked lists,although not necessarily the same list. Describe a fast algorithm for telling if x and y belong to the same list.

# In[1]:


def same_list(list1,list2):
    current = list1
    while current.next is x:
        return false
    current = current.next
    return true 


# Q4. Using PositionalList (ch07/positional_list.py) write a function list_to_positional_list(L) which takes a built-in Python list L and creates a new PositionalList containing the same elements in the same
# order.

# In[2]:


from positional_list import PositionalList
def list_to_positional_list(list_):
    pos_list = PositionalList()
    for element in list_:
        pos_list.add_last(element)
    return pos_list


# Q5. (R-7.11) Implement a function, L.max(), that returns the maximum element from a PositionalList (ch07/positional_list.py) instance L. Assume all values in L are numbers.

# In[3]:


def max( self ):
    return max(element for element in self )

