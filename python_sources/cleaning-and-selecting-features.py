#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


# In[6]:


#import all the files!
train=pd.read_csv("../input/train.csv")
resources=pd.read_csv("../input/resources.csv")
test=pd.read_csv("../input/test.csv")
sample_submission=pd.read_csv("../input/sample_submission.csv")


# In[8]:


train.columns


# In[10]:


columns = ['teacher_prefix','school_state','project_grade_category','project_subject_categories','teacher_number_of_previously_posted_projects']
target = ['project_is_approved']
train_data =train[columns]
test_data =test[columns]
y_true = train[target]
data_for_submission = test['id']


# In[11]:


print(train_data.shape)
print(y_true.shape)
print(test_data.shape)
print(data_for_submission.shape)


# In[21]:


print(train_data.columns)
y_true.head()

