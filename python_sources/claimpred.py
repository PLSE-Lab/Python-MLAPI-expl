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


# # Allstate Claim Prediction (Segmentation)

# In[ ]:


import sklearn
import pandas as pd
from pylab import rcParams #or pl
import numpy as np
import matplotlib.pyplot as plt


# ## Ensure that plots are displayed

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10, 8
pd.set_option('display.max_columns', None)


# ## Load Datasets

# In[ ]:


df_train = pd.read_csv('../input/train_set.csv')


# In[ ]:


df_test = pd.read_csv('../input/test_set.csv')


# In[ ]:


df_train.head()


# 
