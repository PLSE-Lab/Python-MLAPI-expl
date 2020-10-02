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


# In[ ]:


from __future__ import print_function
import sys,tempfile, urllib, os
import pandas as pd


# In[ ]:


train = pd.read_csv('../input/airline-passenger-satisfaction/train.csv')


# In[ ]:


train.drop('Unnamed: 0', axis =1, inplace=True)


# In[ ]:


train.dtypes


# In[ ]:


size = int(0.7*train.shape[0])
train_df = train[:size]
test_df = train[size:]


# In[ ]:


from autoviml.Auto_ViML import Auto_ViML
target='satisfaction'


# In[ ]:


get_ipython().system('pip install autoviml --no-cache-dir --ignore-installed')


# In[ ]:


model, features, trainm, testm = Auto_ViML(train_df, target, test_df,feature_reduction=True,
                                     Boosting_Flag=True,Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False, 
                                    verbose=0)


# In[ ]:





# In[ ]:





# In[ ]:



