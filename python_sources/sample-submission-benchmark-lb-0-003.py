#!/usr/bin/env python
# coding: utf-8

# # Caution! How to submit for practice only

# ### Check path and file name

# In[ ]:


import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Load sample submission file

# In[ ]:


sample = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
sample


# ### Replace the contents of the submission file with a sample file

# In[ ]:


submit = sample
submit


# ### Export the contents of the submitted file in CSV format

# In[ ]:


submit.to_csv('submission.csv', index=False)

