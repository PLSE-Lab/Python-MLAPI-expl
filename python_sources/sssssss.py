#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


['test.csv', 'train.csv', 'sample_submission.csv']


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


from random import randint
import random
for x in list(train.ITEM_CODE.value_counts().index.values): 
    train.ITEM_CODE = random.randint(1,40000)


# In[ ]:


addiotion_predictions = ' '.join([str(x) for x in list(train.ITEM_CODE.value_counts().head(100).index.values)])


# In[ ]:


predictions_from_sample_sub = ' '.join([str(x) for x in list(sample_submission.head(1).ITEM_CODES.value_counts().head(100).index.values)])
predictions_from_sample_sub


# In[ ]:


sumple_sub_exted = addiotion_predictions + ' ' + predictions_from_sample_sub
sumple_sub_exted


# In[ ]:


sample_submission_extended = sample_submission.copy()
sample_submission_extended.ITEM_CODES = sumple_sub_exted
sample_submission_extended.head()


# In[ ]:


sample_submission_extended.to_csv('sample_submission_extended.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




