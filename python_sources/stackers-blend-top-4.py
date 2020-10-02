#!/usr/bin/env python
# coding: utf-8

# * Thanks to
# >* https://www.kaggle.com/jazivxt/safe-box
# >* https://www.kaggle.com/vaishvik25/ensemble
# >* https://www.kaggle.com/jolasa/stacking-higher-every-time-0-9427

# # Vote Early And Vote Often

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


sub1 = pd.read_csv('../input/stacking/stack_median.csv')
sub2 = pd.read_csv('../input/stacking-higher-and-higher/stack_median.csv')
sub3 = pd.read_csv('../input/ensemble/submission_p2_1.csv')
sub4 = pd.read_csv('../input/safe-box/blend02.csv')
sample = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')


# # Blending

# In[ ]:


sample['isFraud'] = (0.35*sub1['isFraud'] + 0.30*sub2['isFraud'] + 0.25*sub3['isFraud'] + 0.10*sub4['isFraud'])
sample.to_csv('All_Blends.csv', index=False)


# In[ ]:


sns.distplot(sample['isFraud']);


# ## [UpVote if this was helpful](http://)
