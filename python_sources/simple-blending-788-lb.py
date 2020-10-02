#!/usr/bin/env python
# coding: utf-8

# **Before you begin please upvote the original authors. Its all there effort not mine.**
# **Links to original kernels-->**
# 1.[Lightgbm with simple features by jsaguiar](http://www.kaggle.com/jsaguiar/lightgbm-with-simple-features-0-785-lb)
# 2.[tidy_xgb -all tables by kxx](http://www.kaggle.com/kailex/tidy-xgb-all-tables-0-782/code)

# In[2]:


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


# In[6]:


data1 = pd.read_csv('../input/lightgbm-with-simple-features-0-785-lb/submission_kernel00.csv')
data2 = pd.read_csv('../input/tidy-xgb-all-tables-0-782/tidy_xgb_0.77821.csv')


# In[7]:


data1['TARGET'] = (data1['TARGET']+data2['TARGET'])/2


# In[8]:


data1.to_csv('blend1_.788lb.csv',index = False)


# In[ ]:




