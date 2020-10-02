#!/usr/bin/env python
# coding: utf-8

# In[30]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[31]:


import turicreate as tc


# In[44]:


get_ipython().system(' mkdir temp')


# In[45]:


get_ipython().system('ls')


# In[50]:


tc.config.set_runtime_config('TURI_CACHE_FILE_LOCATIONS', 'temp')


# In[51]:


data =  tc.SFrame('../input/BreastCancer2.csv')


# In[52]:


train_data, test_data = data.random_split(0.8)


# In[53]:


model = tc.classifier.create(train_data, target='class', features = ['thickness', 'size','shape','adhesion','single','nuclei','chromatin','nucleoli','mitosis'])


# In[54]:


predictions = model.classify(test_data)


# In[55]:


predictions


# In[56]:


# obtain statistical results for the model by model.evaluate method 
results = model.evaluate(test_data)


# In[57]:


results


# **Model has an accuracy of 95.36% AUC 98.69%**

# In[ ]:




