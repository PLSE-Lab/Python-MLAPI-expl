#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Environment Sanity Check
# 
# Click the Runtime dropdown at the top of the page, then Change Runtime Type and confirm the instance type is GPU.
# 
# Check the output of !nvidia-smi to make sure you've been allocated a Tesla T4, P4, or P100

# In[ ]:


get_ipython().system('nvidia-smi')


# # Setup:
# 

# ### Python Version

# In[ ]:


# Check Python Version
get_ipython().system('python --version')


# ### Check CUDA Version

# In[ ]:


# Check CUDA/cuDNN Version
get_ipython().system('nvcc -V && which nvcc')


# ### installing Rapids
# 
# #### prerequists
# 
# should the rapids dataset have to be added into input folder before running the installation since
# it contains the libraries that needed to be installed.

# In[ ]:


import sys
get_ipython().system('cp ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# # CUDF tutorials

# In[ ]:


import cudf
from cudf.core import Series
import math


# In[ ]:


get_ipython().run_line_magic('time', 'a = Series([9, 16, 25, 36, 49], dtype=np.float64)')


# ## InliningPython user defined functions (UDFs) into native CUDA kernels:
# * A combination of flexibility and performance
# * A very good example of the power of just-in-time (JIT) compilation
# * Userdoes not need to know CUDA****

# In[ ]:


a


# In[ ]:


get_ipython().run_line_magic('time', 'a.applymap(lambda x : x**2)')


# In[ ]:


get_ipython().run_line_magic('time', 'a.applymap(lambda x: 1 if x < 18 else 2)')


# ## rooling window udf
# ----------------------------
# ![image.png](attachment:image.png)
# 
# (source: http://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21393-combined-pythoncuda-jit-for-flexible-acceleration-in-rapids.pdf)

# In[ ]:


#defining averaging of sqare roots rolling window function

def foo(A):
    sum = 0
    for a in A:
        sum = sum + math.sqrt(a)
    return sum / len(A)


# In[ ]:


#defining averaging rolling window function

def foo2(A):
    sum = 0
    for a in A:
        sum = sum + a
    return sum / len(A)


# **cudf.core.Series.rolling(W, m, center).apply(some_custom_udf)**
#     
#     W - window size
#     m - minimum size of included element in the first window. eg: 1 means the first window will start
#         at the first element itself. But if it is two the first window will start with first 2 elements.
#     center - True or Flase. If true the result will be set at the center of the window. Else on edge of the window
#     

# In[ ]:


get_ipython().run_line_magic('time', 'a.rolling(3, 1, False).apply(foo)')


# In[ ]:


get_ipython().run_line_magic('time', 'a.rolling(3, 1, False).apply(foo2)')


# In[ ]:


get_ipython().run_line_magic('time', 'a.rolling(3, 1, True).apply(foo2)')

