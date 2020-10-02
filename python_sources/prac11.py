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


# Adjacency List
# V - U,W,X
# U-V,W
# W-U,V,X,Y
# X-Z,V,W,Y
# Y-W,X
# 
# Adjacency Matrix
#   V  U  W  X   Y
# V 0  1  1  1   0
# U 1  0  1  0   0
# W 1  1  0  1   1
# X 1  0  0  0   1
# Y 0  0  1  1   0

# Indicates selected verticies at each step
# 1.Z(0)
# 2.X(1)
# 3.V(2),W(2),Y(2)
# 4.Y(2),W(2),U(3)
# 5.Y(2),U(3)
# 6.U(3)
# 7.

# Indicates selected verticies at each step
# 1/12.U
# 2/11.V
# 3/10.W
# 4/9.X
# 5/6.Y
# 7/8.Z
# 
# 

# 
# a)
# 2---------                -5- 
# |  -      -             -  |  -
# |   - 3----- 4------6--    |   -8          
# |  -      -             _  |  -
# 1---------                -7-
# 
# 
# b)
# Indicates selected verticies at each step
# 1.1(0)
# 2.2(1),3(1),4(1)
# 3.3(1),4(1)
# 4.4(1)
# 5.6(2)
# 6.5(3),7(3)
# 7.7(3),8(4)
# 8.8(4)
# 9.
# 
# c)
# 1/16.1
# 2/15.2
# 3/14.3
# 4/13.4
# 5/12.6
# 6/11.5
# 7/11.7
# 8/9.8

# 
