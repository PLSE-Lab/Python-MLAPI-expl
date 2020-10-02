#!/usr/bin/env python
# coding: utf-8

# In[11]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
image_list = os.listdir("../input")
print(image_list)
# Any results you write to the current directory are saved as output.


# CP2410 Practical 09 - Search Trees
# 1. Insert, into an empty binary search tree, entries with keys 30, 40, 24, 58, 48, 26, 25 (in this order).
# Draw the tree after each insertion.
# 
# 

# Solution
# Image for question in input/ImagesPrac9/Q1.jpg

# 
# 2. (R-11.3) How many different binary search trees can store the keys {1,2,3}?
# 5, as shown below
# 

# 5 as per the image in the solution. 
# 
# Solution
# Image for question in input/ImagesPrac9/Q2.jpg

# 3. Draw an AVL tree resulting from the insertion of an entry with key 52 into the AVL tree below:
# 
# 

# Solution
# Image for question in input/ImagesPrac9/Q3.jpg

# 4. Consider the set of keys K = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}. Draw a (2, 4) tree
# storing K as its keys using the fewest number of nodes.
# 

# Solution
# Image for question in input/ImagesPrac9/Q4.jpg

# 5. Insert into an empty (2, 4) tree, entries with keys 5, 16, 22, 45, 2, 10, 18, 30, 50, 12, 1 (in this order).
# Draw the tree after each insertion.
# 

# Solution
# Image for question in input/ImagesPrac9/Q5.1.jpg
# 
# and 
# 
# Solution
# Image for question in input/ImagesPrac9/Q5.2jpg
