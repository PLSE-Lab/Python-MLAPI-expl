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


# # Kaggle Kernel Diagnostic Info
# 
# The content below has been adapted from [here](https://www.kaggle.com/kardopaska/kernel-hardware-cpu-info). Run the following cell `as-is`.

# In[ ]:


import time

print("Current time : %s"  % time.strftime("%c") )

import platform
print('hello')
print(platform.processor())

print('Version      :', platform.python_version())
print('Version tuple:', platform.python_version_tuple())
print('Compiler     :', platform.python_compiler())
print('Build        :', platform.python_build())
print()
print('uname:', platform.uname())
print()
print('system   :', platform.system())
print('node     :', platform.node())
print('release  :', platform.release())
print('version  :', platform.version())
print('machine  :', platform.machine())
print('processor:', platform.processor())
print('cpu_count:', os.cpu_count())


# # GPU Information
# 
# Run the following cell `as-is`.  
# **Note**: In order to run XgBoost with GPU support refer to [this github thread](https://github.com/dmlc/xgboost/issues/2819) and [this kaggle thread](https://www.kaggle.com/c/home-credit-default-risk/discussion/58787).

# In[ ]:


get_ipython().system('nvidia-smi')


# # Kaggle Environment Information
# 
# <font color='red'> NOTE </font>: We will **REDACT** a few fields before reporting the values in those fields for reasons of account-privacy and safety. The fields that we will REDACT are as follows:  
# ```python
# exclude_keys = ['HOSTNAME', 'KAGGLE_DATA_PROXY_TOKEN', 'KAGGLE_USER_SECRETS_TOKEN', 'KAGGLE_DATA_PROXY_PROJECT']
# ```
# 
# Run the following two cells `as-is`.

# In[ ]:


exclude_keys = ['HOSTNAME', 'KAGGLE_DATA_PROXY_TOKEN', 'KAGGLE_USER_SECRETS_TOKEN', 'KAGGLE_DATA_PROXY_PROJECT']
environment_info = dict(os.environ)
for key in exclude_keys:
    environment_info.update({key:"---- ### REDACTED ### ----"})


# In[ ]:


environment_info


# In[ ]:




