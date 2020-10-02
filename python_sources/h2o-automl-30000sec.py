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


# import automl
import h2o
from h2o.automl import H2OAutoML
train = pd.read_csv('../input/cat-in-the-dat/train.csv')
test = pd.read_csv('../input/cat-in-the-dat/test.csv')

h2o.init()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


# train automl
htrain = h2o.H2OFrame(train)


# Identify predictors and response
x = htrain.columns
y = "target"
x.remove(y)
x.remove('id')

# for classification
htrain[y] = htrain[y].asfactor()

aml = H2OAutoML(seed=42, max_runtime_secs=300000)#28800
aml.train(x=x, y=y, training_frame=htrain)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead


# In[ ]:


# predict test
htest = h2o.H2OFrame(test)

htest.drop(['id'])
preds = aml.leader.predict(htest)


# In[ ]:


# make submit
sample_submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')
sample_submission['target'] = preds['p1'].as_data_frame().values
sample_submission.to_csv('H2O_AutoML_3600s.csv', index=False)

