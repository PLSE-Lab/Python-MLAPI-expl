#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

# Testing... Hi, Joshua...


# In[ ]:


#print(test_data.describe()) # descriptive summary of the data 
print(test_data.columns) # columns or available features of the dataset


# In[ ]:


# Y = SalesPrice
test_data.SalePrice

