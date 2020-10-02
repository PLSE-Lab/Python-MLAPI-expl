#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


get_ipython().system('wget https://codeload.github.com/aDropInTheOcean/ChineseNRE/zip/master')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import zipfile
with zipfile.ZipFile("master","r") as zip_ref:
    zip_ref.extractall()
get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls')
get_ipython().system('ls ChineseNRE-master/data/people-relation')


# In[ ]:


os.chdir("ChineseNRE-master/data/people-relation/")
get_ipython().system('ls')


# In[ ]:


get_ipython().system('python data_util.py')


# In[ ]:


os.chdir("..")
get_ipython().system('ls')


# In[ ]:


os.chdir("..")
get_ipython().system('ls')


# In[ ]:


get_ipython().system('python train_epoch.py')

