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


# **Statistical Thinking**
# **Binning** - Square root of the number of data points
# **Binning bias** - The same data maybe interpreted differently with the choice of number of bins. And to get remedy from this we have to use bee swarm plot.
# **ECDF** - Function definition
#     
#     def ecdf(data):
#         """Compute ECDF for a one-dimensional array of measurements."""
#         # Number of data points: n
#         n = len(data)
# 
#         # x-data for the ECDF: x
#         x = np.sort(data)
# 
#         # y-data for the ECDF: y
#         y = np.arange(1, 1+n) / n
# 
#         return x, y
# 
# 

# When the number of data are very large and the bee swarm plots are too cluttered then we should use box and whisker plot.
# Variance and standard deviation
# Covariance and pearson correlation coefficient

# In[ ]:




