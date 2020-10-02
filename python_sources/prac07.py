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


# Question 1:
# a)
# 5
# 5,6
# 3,5,6
# 1,3,5,6
# 1,2,3,5,6
# 1,2,3,5,6,7
# 1,2,3,5,6,7,8
# 1,2,3,5,6,7,8,9
# 
# b)
# 5,6,3,1,2,7,9,8
# 1,6,3,5,2,7,9,8
# 1,2,3,5,6,7,9,8
# 1,2,3,5,6,7,9,8
# 1,2,3,5,6,7,9,8
# 1,2,3,5,6,7,8,9
# 

# The worst case scenario is if all the numbers in the list are in descending order, because each number would then need to be comparedd to the privious numbers.
# 

# question 3.
# https://drive.google.com/open?id=1v2s0gK47zRc8zXqkG7BjzYhmN-uqk9Cp

# Question 4.
# 
# https://drive.google.com/open?id=1BUpz9-DBL3lY6uZ19RO8bY-NuOTctajA

# Question 5.
# third smallest key would be in the 3rd postion being depth 1 or 2.
# 

# the largest key would be stored at the bottom or second from the bottom of the tree
