#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/submission-2/submission_2.csv')


# In[ ]:


df.to_csv('submission.csv', index=False)


# In[ ]:




