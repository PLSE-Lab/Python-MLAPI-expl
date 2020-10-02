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
# Any results you write to the current directory are saved as output.


# In[ ]:


ROOT = "/kaggle/input/trends-assessment-prediction/"
get_ipython().system('ls {ROOT}')


# In[ ]:


# image and mask directories
data_dir = f'{ROOT}/fMRI_train'


# In[ ]:


#!python
#!/usr/bin/env python
from scipy.io import loadmat
path = data_dir + '/10025.mat'
import h5py
with h5py.File(path, 'r') as file:
    print(list(file.keys()))
    print(file)


# In[ ]:


f = h5py.File(path, 'r')
f['SM_feature']


# In[ ]:


f['SM_feature'][0][0]

