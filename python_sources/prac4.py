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


# In[ ]:





# Q1.
# 3
# 8
# 2
# 1
# 6
# 7
# 4
# 9

# Q2
# pop is the only method capbale of removing items, 10 operations, 3 are ingnored therefore the stack S contians 18 values 

# In[ ]:


#Q3
def transfer(S,T):
    while not s.is_empty():
        t.push(S.pop())


# Q4
# 5
# 3
# 2
# 8
# 9
# 1
# 7
# 6

# Q5
# 22

# In[ ]:


#Q6
def rearrangeD(D,Q):
    Q.enqueue(D.delete_first())
    Q.enqueue(D.delete_first())
    Q.enqueue(D.delete_first())
    D.add_last(D.delete_first())
    Q.enqueue(D.delete_first())
    Q.enqueue(D.delete_last())
    Q.enqueue(D.delete_first())
    Q.enqueue(D.delete_first())
    Q.enqueue(D.delete_first())
    


# In[ ]:


#Q7
def rearrangeS(S,Q)
S.push(D.delete_last)
S.push(D.delete_last)
S.push(D.delete_last)
D.add_first(D.delete_last)
S.push(D.delete_last())
S.push(D.delete_first())
S.push(D.delete_last)
S.push(D.delete_last)
S.push(D.delete_last)
for i in range(8):
    D.add_first(S.pop())

