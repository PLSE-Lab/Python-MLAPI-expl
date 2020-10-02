#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")


# In[ ]:


data.head()


# In[ ]:


data.isna().sum()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.nunique()


# It is a 2 class problem.

# In[ ]:


data.describe().transpose()


# In[ ]:


len(data[data["Class"]==0])


# In[ ]:


len(data[data["Class"]==1])


# It is a an imbalanced dataset.

# In[ ]:


from pycaret.classification import *
classify=setup(data=data,target="Class")


# In[ ]:


compare_models()


# The output is imbalanced and we don't check the accuracy as it leads to metric trap. Instead we go for F1 Score and Kappa score. Kappascore is very much precise when your dataset is imbalanced.

# In[ ]:


# getting the catboost model
catboost=create_model('catboost')


# catboost is trained for 10 cross validations and kappa score is 0.8557 which is good to go.
