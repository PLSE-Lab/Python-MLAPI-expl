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


# In[ ]:


get_ipython().system('pip install -U pycaret')


# In[ ]:


# Importing dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')


# In[ ]:


diabetes['Class variable'].unique()


# In[ ]:


# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')


# In[ ]:


# comparing all models
compare_models()


# In[ ]:


lr = create_model('lr')


# In[ ]:


tuned_lr = tune_model('lr')


# In[ ]:


lr


# In[ ]:


tuned_lr


# In[ ]:


from pycaret.datasets import get_data
diabetes = get_data('diabetes')
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')
# creating decision tree model
dt = create_model('dt')
# ensembling decision tree model (bagging)
dt_bagged = ensemble_model(dt)


# In[ ]:




