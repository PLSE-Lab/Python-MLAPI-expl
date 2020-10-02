#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# Any results you write to the current directory are saved as output.


# <h2>My own output run in MesoNet <br></h2>
# Code run from https://www.kaggle.com/minhtam/fake-detect-basic

# In[ ]:


submission = pd.read_csv("/kaggle/input/submit-dfdc/submit2.csv",names = ["filename", "label"])
submission_ori = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")


# In[ ]:


submission = submission.sort_values('filename')


# In[ ]:


submission_ori['label'] = 1-submission['label'].astype(float)


# In[ ]:


submission_ori.to_csv('submission.csv', index=False)


# In[ ]:


submission_ori


# In[ ]:




