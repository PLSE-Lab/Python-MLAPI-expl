#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


sub1 = pd.read_csv('../input/blends/All_Blends.csv')
sub2 = pd.read_csv('../input/great2/stack_median.csv')
sub3 = pd.read_csv('../input/great3/blend02.csv')
sub4 = pd.read_csv('../input/great2/stack_mean.csv')
sample = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')


# In[ ]:


sample['isFraud'] = (0.40*sub1['isFraud'] + 0.20*sub2['isFraud'] + 0.20*sub3['isFraud'] + 0.20*sub4['isFraud'])
sample.to_csv('ieeeall.csv', index=False)


# In[ ]:


sns.distplot(sample['isFraud']);

