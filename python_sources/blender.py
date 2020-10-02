#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

s1 = pd.read_csv('../input/kernel-01000110-01010101-01001110/submission.csv').rename(columns={'target': 'target1'})
s2 = pd.read_csv('../input/santander-ls/submission.csv').rename(columns={'target': 'target2'})
s3 = pd.read_csv('../input/santander-improved-starter-solution/submission.csv').rename(columns={'target': 'target3'})


# In[ ]:


sub = pd.merge(s1, s2, how='left', on='ID_code')
sub = pd.merge(sub, s3, how='left', on='ID_code')
sub['target'] = ((sub['target1'] * 0.33) + (sub['target2'] * 0.33) + (sub['target3'] * 0.34)).clip(0,1)
sub[['ID_code','target']].to_csv('submission.csv', index=False)

