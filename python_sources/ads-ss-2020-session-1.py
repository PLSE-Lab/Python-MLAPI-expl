#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Extra Information...**

# In[ ]:


a = [2,1,5,3,4]
print(a)


# In[ ]:


print(a[0])


# In[ ]:


if (1 % 2 == 0):
    print("something")


# [2  1  5  3  4]
#  j j+1
# 
# index j
# index "j+1"
# 
# comape 2 with 1
# 
# if (j < j+1) ?
# if (2 < 1)
# 
# if (j > j +1)?
# if (2 > 1)? --> SWAP
# 
# helper = 2
# 1, 1, 5, 3, 4
# 
# 1, helper, 5, 3,4
# **1, 2**, 5, 3, 4

# In[ ]:


a = 1
b = 2

print(a)
print(b)
# swap both values

helper = a
a = b
b = helper

# swapped output
print("swapped")
print(a)
print(b)


# In[ ]:





# 1, 2, 5, 3, 4
# if (j > j+1) ? 
# if (2 > 5)
# 
# 1, 2, 5, 3, 4
# 
# if (5 > 3)?
# SWAP!
# 
# 1, 2, 3, 5, 4
# if (5 > 4)?
# SWAP!
# 
# 1, 2, 3, 4, 5
# 
# 

# 

# 

# ORIGIN 2, 1, 5, 3, 4
# 1, 2, 5, 3, 4
# 1, 2, 5, 3, 4
# 1, 2, 3, 5, 4
# 1, 2, 3, 4, 5
# 

# - gives array - 0 - 
# 5, 1, 4, 2
# - 1 - 
# 1, 5, 4, 2
# 1, 4, 5, 2
# 1, 4, 2, 5
# - 2 -
# 1, 4, 2, 5
# 1, 2, 4, 5
# - 3- 
# 1, 2, 4, 5
# - 4 - 
# 1, 2, 4, 5
# 
# ARRAY SORTED ASCENDING
# 
# 
# 

# 13, 2, 9, 4, 18, 45, 37, 63
# 
# 
# after 1st iteration?
# 
# 
# 
# 
# after 2nd iteration?
# 
# 
