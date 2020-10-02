#!/usr/bin/env python
# coding: utf-8

# ## Sample submissions
# 
# This is my first submission to the competition! 
# 
# Just tweaked a few things to create random submissions

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/liverpool-ion-switching/train.csv')


# In[ ]:


sample_df = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv', index=False,  float_format='%.4f')


# In[ ]:


sample_df['open_channels']=1


# In[ ]:


sample_df.to_csv('submission_1.csv', index=False,  float_format='%.4f')


# In[ ]:


sample_df['open_channels']=2
sample_df.to_csv("submission_2.csv", index=False,  float_format='%.4f')


# In[ ]:




