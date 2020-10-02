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





# 1. 
#     a) 5,6,3,1,2,7,9,8
#        1,6,3,5,2,7,9,8
#        1,2,3,5,6,7,9,8
#        1,2,3,5,6,7,8,9
#       b)5,6,3,1,2,7,9,8
#         5
#         5,6
#         3,5,6
#         1,3,5,6
#         1,2,3,5,6
#         1,2,3,5,6,7
#         1,2,3,5,6,7,8
#  2. Worst case senario is that keys are in a decending order, because this would mean checking the entiresequence before every inersertion hance the o(n^2) runtime
#  3.
#        5
# ______________________________
#      5
#   
#   1
#      
# ____________________________
# 
#         1
#      
#     5
#  _____________
#  
#        1
#     5      4
#     
#  _____________
#        1
#     5      4
#  7
#  
#  _________________
#            1
#     5           4
#  7       3
# ______________________
#            1
#        3       4
#     7      5
#     
#     ___________________
#            1
#        3       4
#     7    5    9
#  -------------------------
#            0
#        3        1
#     7     5    9   4
#     
#  _______________________
#      
#            0
#        2        1
#     3     5    9   4
#  7
#  
# _____________________________
#             0
#        2        1
#     3     5    9   4
#  7     8
#  
#  
#  
#  4. 
#  
#             2
#          3        4
#        8    5   7   6
#  __________________________
#  
#              6
#           3        4 
#        8     5    7 
#        
#  _____________________________
#               4
#           3        6 
#        8     5    7 
#      _____________________
#                3
#           4        6 
#        8     5    7 
#        
#        
#   5. Depth 2
#   6, The largest key must be in an external node 
#        
#        
