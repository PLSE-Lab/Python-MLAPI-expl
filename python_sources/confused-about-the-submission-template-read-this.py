#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# It took me a while to understand the submission template. I have explained it here using the files given. 

# ## Read the data 

# In[ ]:


#Read the data
train = pd.read_csv("../input/bigquery-geotab-intersection-congestion/train.csv")
test = pd.read_csv("../input/bigquery-geotab-intersection-congestion/test.csv")
sample_submission  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")


# In[ ]:


print (train.columns)
test.head()


# ## Explanation

# In[ ]:


print (test.shape)
print (sample_submission.shape)


# The number of observations in the test file is 1920335 and in the submission file is 11522010. We are asked to predict the following six variables: 
# 
# TotalTimeStopped_p20
# 
# TotalTimeStopped_p50
# 
# TotalTimeStopped_p80
# 
# DistanceToFirstStop_p2
# 
# DistanceToFirstStop_p50 
# 
# DistanceToFirstStop_p80
# 
# Each of the above variables have a metric id as we can see from the submission_metric_map.json file. 
# 

# In[ ]:


submission_metric_map = '../input/bigquery-geotab-intersection-congestion/submission_metric_map.json'
with open(submission_metric_map, 'r') as f:
    data = json.load(f)
data


# In[ ]:


sample_submission.head(5)


# Let's look at the sample_submission file. The first row has a TargetId of 0_0. TargetId 0_0 means RowID of 0 in test and its prediction for TotalTimeStopped_p20. Similarly, TargetID of 0_4 is RowID of 0 in test and its prediction for DistanceToFirstStop_p50. 
# 
# Hope it is clear! 
