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


# Q1. (R-8.1) The following questions refer to the tree of Figure 8.3.
# - Which node is the root?
# - What are the internal nodes?
# - How many descendants does node cs016/ have?
# - How many ancestors does node cs016/ have?
# - What are the siblings of node homeworks/?
# - Which nodes are in the subtree rooted at node projects/?
# - What is the depth of node papers/?
# - What is the height of the tree?
# 
# ![image.png](attachment:image.png)

# -	user/rt/courses
# - Cs016, cs252
# - 9
# - 1
# - Programs,grades
# - Buylow,sellhigh
# - 3
# - 5
# 

# Q2 For the following tree, demonstrate (by hand) the output of:
# - Inorder traversal
# - Preorder traversal
# - Postorder traversal
# 

# ![image.png](attachment:image.png)

# 
