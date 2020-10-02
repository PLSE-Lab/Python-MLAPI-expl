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


Installation of Pycaret


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


Read Data


# In[ ]:


data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.drop(['id','Unnamed: 32'],axis=1,inplace=True)


# Import pycaret classification module and setup

# In[ ]:


from pycaret.classification import *
exp_clf101 = setup(data = data, target = 'diagnosis', session_id=123) 


# Compare models

# In[ ]:


compare_models()

create model
# In[ ]:


lda = create_model('lda')


# Tuning of model

# In[ ]:


tuned_lda = tune_model('lda')


# Evaluation of Model

# In[ ]:


plot_model(tuned_lda, plot = 'confusion_matrix')


# In[ ]:


plot_model(tuned_lda, plot = 'pr')


# In[ ]:


plot_model(tuned_lda, plot='feature')


# In[ ]:


plot_model(tuned_lda, plot = 'auc')


# In[ ]:




