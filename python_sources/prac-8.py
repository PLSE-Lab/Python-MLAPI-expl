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





# 0    1     2     3      4     5     6     7     8     9     10
# 
# 13  94                        44               12    16     20
#     39                        88               23    5
#                               11

# 0    1    2    3   4    5   6   7   8   9    10
# 12  94   39   16   5   44  88   11  12  23   20

# 0    1    2    3   4    5   6   7   8   9    10
# 13  94   39   11       44  88  16  12  23    20

# 0    1    2    3   4    5   6   7   8   9    10
# 12  39   23   88   5   44  94  16  12  11    20
# 

# 0    1    2    3   4    5   6   7   8   9    10   11   12   13   
#          12   18  41       36  25      54              38   10
# 
# 14   15    16  17   18   19
#      90    28

# Use a binary search on each individual line. this will be O(nlgn) because it goes through it line once.
