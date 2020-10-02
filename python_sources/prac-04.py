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


# QUESTION 1:
# push(5): 5
# push(3): 5,3
# pop(): 5 return 3
# push(2): 5, 2
# push(8): 5,2,8
# pop(): 5,2 return 8 
# pop(): 5 return 2
# push(9): 5,9
# push(1): 5,9,1
# pop():5,9 return 1
# push(7): 5,9,7
# push(6): 5,9,7,6
# pop(): 5,9,7 return 6
# pop(): 5,9 return 7
# push(4): 5,9,4
# pop():5,9 return 4
# pop():5 return 9

# QUESTION 2:
# 10-3 = 7 pops
# therefore:
# 25- 7 = 18

# In[ ]:


#QUESTION 3
def transfer(s,t):
    while not s.is_empty():
        t.push(s.pop)


# QUESTION 4:
# enqueue(5): 5
# enqueue(3): 5, 3
# dequeue(): 3 return 5
# enqueue(2): 3, 2
# enqueue(8): 3,2,8
# dequeue(): 2,8 return 3
# dequeue(): 8 return 2
# enqueue(9): 8,9
# enqueue(1): 8,9,1
# dequeue(): 9,1 return 8
# enqueue(7): 9,1,7
# enqueue(6): 9,1,7,6
# dequeue(): 1,7,6 return 9
# dequeue(): 7,6 return 1
# enqueue(4): 7,6,4
# dequeue(): 6, 4 return 7
# dequeue() 4 return 6

# QUESTION 5:
# 15 (dequeues)-5 (errors) = 10 (dequeues)
# therefore:
# 32-10 = 22
# 

# ![image.png](attachment:image.png)

# 
