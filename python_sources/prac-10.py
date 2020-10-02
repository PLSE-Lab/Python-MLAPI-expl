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


# Q1. a merge sort takes a set O(n lg n) time because the algorithm uses a middle point to divide this the times depends on the height of the tree. 
# 
# A quick sort however splits on a random value given that there are only 1's or 0's in this sequence it will only take 1 interation hence O(n) time

# Q2. Apply a O(n) merge algoithm to both a an b to produce a third sequence, then traverse the third sequence and remove duplicates. This also takes O(n) time. 

# Q3. use 1 as the pivot in a quick sort, this will place all 0's before the 1's

# Q4. Sort the sequence by Id first, then traverse the sequence storing a max and comparing it to all Id's, replacing max when needed

# Q5. 

# Q6. 1000,80,10,50,70,60,90,20
# 
#    1000,80,10,50
# 1000,80 > 80,1000 | 10,50 > 10, 50 -->  10,50,80,1000                                                      -->10,20,50,60,70,90,1000
# 70,60,90,20
# 70,60 > 60,70 | 90,20 >20,90       --> 20,60,70,90 
# 

# In[ ]:




