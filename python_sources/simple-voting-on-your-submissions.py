#!/usr/bin/env python
# coding: utf-8

# # Simple voting on your submissions
# 
# This kernel applies simple voting to your existing submissions. Could help you to improve LB score a little.

# In[ ]:


import pandas as pd
import numpy as np
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Choose several of your best submissions and load .csv:

# In[ ]:


# lb 0.69279
subm1 = pd.read_csv('../input/cdiscount-image-classification-submission-samples/submission_sample_1.csv')
# lb 0.69280
subm2 = pd.read_csv('../input/cdiscount-image-classification-submission-samples/submission_sample_2.csv')
# lb 0.69281
subm3 = pd.read_csv('../input/cdiscount-image-classification-submission-samples/submission_sample_3.csv')
# lb 0.68966
subm4 = pd.read_csv('../input/cdiscount-image-classification-submission-samples/submission_sample_4.csv')


# Merge datasets by '_id' column:

# In[ ]:


subm_all = subm1.merge(subm2,on='_id').merge(subm3,on='_id').merge(subm4,on='_id')
subm_all.head()


# Apply voting by getting most frequent category for each item (__takes several minutes on my laptop__):

# In[ ]:


subm_all_voting = subm_all.mode(axis=1)
subm_all_voting.head()


# Save results to a new .csv file:

# In[ ]:


result = pd.DataFrame()
result['_id'] = subm_all['_id']
result['category_id'] = subm_all_voting[0].astype(int)
result.to_csv('submission_voting.csv', index=False)

