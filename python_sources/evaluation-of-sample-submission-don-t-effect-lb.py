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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # experiment of sample submission
# In Evaluation tab,
# 
# >Each row contains an id that is a concatenation of an item_id, a store_id, and the prediction interval, which is either validation (corresponding to the Public leaderboard), or evaluation (corresponding to the Private leaderboard). You are predicting 28 forecast days (F1-F28) of items sold for each row. For the validation rows, this corresponds to d_1914 - d_1941, and for the evaluation rows, this corresponds to d_1942 - d_1969. (Note: a month before the competition close, the ground truth for the validation rows will be provided.)
# 
# so, I experiment modify only `evaluation` row, and submit.

# In[ ]:


sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")
sample_submission.head()


# In[ ]:


sample_submission.tail()


# # modify only `evaluation` row

# In[ ]:


sample_submission = sample_submission.set_index("id")
sample_submission[sample_submission.index.map(lambda x: x.split("_")[-1] == "evaluation")] += 1
sample_submission


# In[ ]:


sample_submission.to_csv('submit.csv')


# In[ ]:




