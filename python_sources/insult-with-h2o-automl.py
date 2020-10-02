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


import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='15G')


# In[ ]:


train = h2o.import_file("../input/tf-embedding-files-joiner/train.csv")
test = h2o.import_file("../input/tf-embedding-files-joiner/test.csv")


# In[ ]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[ ]:


x = ['C'+str(i) for i in range(512)]
y = 'insult'


# In[ ]:


# For binary classification, response should be a factor
train[y] = train[y].asfactor()


# In[ ]:


# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=120, seed=1, max_runtime_secs=29000)
aml.train(x=x, y=y, training_frame=train)


# In[ ]:


# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)


# In[ ]:


# The leader model is stored here
aml.leader


# In[ ]:


# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)


# In[ ]:


preds['predict']


# In[ ]:


sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
sample_submission['insult'] = preds.as_data_frame()['p1'].values
sample_submission.to_csv('insult_submission.csv')

