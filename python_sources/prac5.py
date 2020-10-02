#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:





# def second_to_last(LL):
#     current= LL.head
#     next=LL.Head.next
#     while next.next != None :
#         current=next
#         next=next.next
#     return current
#         

# def count(LL):
#     if L.current==None:
#         return 0
#     count=1
#     previous=LL.current
#     current=previous
#     while current.next !=previous:
#         count=+1
#     return count
# 

# X=current
# while current.next!=y
#     if current.next=X:
#         return false:
#     current=current.next
#     
# reutrn true

# In[2]:


from positional_list import PositionalList

def positonal(list):
    pos_list=Positionallist()
    for i in list:
        pos_list.add_last(i)
    return pos_list


# def max(list):
#     return max(i for i in list)
