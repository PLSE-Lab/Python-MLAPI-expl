#!/usr/bin/env python
# coding: utf-8

# Optimization Problem

# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 
# 
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# 
# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# 
# import os
# print(os.listdir("../input"))
# 
# # Any results you write to the current directory are saved as output.

# In[ ]:


c= [-3,-2]
A= [[1,1,],[2,1],[1,0]]
b= [80,100,40]
x0_bounds = (0, None)
x1_bounds = (0, None)
# import library
from scipy.optimize import linprog
#solve 
solution = linprog(c,A,b, bounds=(x0_bounds, x1_bounds), method ='simplex')
print(solution)


# 2 phase
