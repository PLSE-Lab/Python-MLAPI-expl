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


# In[ ]:


get_ipython().run_line_magic('time', "test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')")
get_ipython().run_line_magic('time', "train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')")
get_ipython().run_line_magic('time', "sample_submission = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')")


# In[ ]:


print(test.head())


# In[ ]:


print(train.head())


# In[ ]:


print(sample_submission.head())


# In[ ]:


test.dtypes


# In[ ]:


train.dtypes


# In[ ]:


test.sample(5)


# In[ ]:


train.sample(5)


# In[ ]:


sample_submission.sample(5)

