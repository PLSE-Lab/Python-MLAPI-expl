#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


# Read files:
train = pd.read_csv("../input/Train.csv")
test = pd.read_csv("../input/Test.csv")


# In[ ]:


# Combine train and test dataset to perform feature engineering. Specify source columns in new dataset
train['source'] = 'train'
test['source'] = 'test'

