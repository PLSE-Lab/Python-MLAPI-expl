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


get_ipython().system('pip install pycaret-nightly==0.20')


# In[ ]:


from pycaret.datasets import get_data
data = get_data('juice')


# In[ ]:


from pycaret.classification import *
clf1 = setup(data, target = 'Purchase', silent=True)


# In[ ]:


t1 = compare_models()


# In[ ]:


rf = create_model('rf')


# In[ ]:


tuned_rf = tune_model(rf)


# In[ ]:


dt = create_model('dt')


# In[ ]:


ensembled_dt = ensemble_model(dt)


# In[ ]:


xgboost = create_model('xgboost', fold=5)


# In[ ]:


dt = create_model(dt, ensemble=True, method = 'Boosting')


# In[ ]:




