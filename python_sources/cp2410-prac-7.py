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


# 1. Suppose we have a list [5, 6, 3, 1, 2, 7, 9, 8]. Show the list at each stage when using:
# 
# 

# a. Insertion sort
# - 5
# - 5,6
# - 3,5,6
# - 1,3,5,6
# - 1,2,3,5,6
# - 1,2,3,5,6,7
# - 1,2,3,5,6,7,8,9
# 

# b. Selection sort
# - 1,6,3,5,2,7,9,8
# - 1,2,3,5,6,7,9,8
# - 1,2,3,5,6,7,9,8
# - 1,2,3,5,6,7,8,9

# 2. Give an example of a worst-case sequence with n elements for insertion-sort, and show that insertion-sort runs in O(n^2 ) time on such a sequence.
# 
# 

# In worst case scenario, insertion sort checks every element in the list and compares it to every other element and as such it measures the list n^2 times, as in it checks the length of the elements by the length of the element. 

# 
# 3. Show the state of an initially empty heap at each point as the following keys are added: 5, 1, 4, 7,3, 9, 0, 2, 8.
# ![image.png](attachment:image.png)

# 4. Show all the steps of the algorithm for removing key 2 from the heap of below.
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# 
# 5. (R-9.10) At which positions of a heap might the third smallest key be stored?
# 	- The third smallest key will be at depth 1 or 2
# 
# 
# 

# 6. (R-9.11) At which positions of a heap might the largest key be stored?
# 	- The largest key must be in an external node position. 
