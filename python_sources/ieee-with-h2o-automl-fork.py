#!/usr/bin/env python
# coding: utf-8

# ## TODO: Temporal validation split

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


import h2o
print(h2o.__version__)
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')


# In[ ]:


train = h2o.import_file("../input/standalone-train-and-test-preprocessing/train.csv")
test = h2o.import_file("../input/standalone-train-and-test-preprocessing/test.csv")


# In[ ]:


print(train.head().isna().sum(axis=1))
# train.head().apply(lambda x: x==np.nan).sum()


# In[ ]:


print(train.shape) # 433 columns = includes the static/identity variables
train.head()


# In[ ]:


test.head()


# In[ ]:


object_columns = np.load('../input/standalone-train-and-test-preprocessing/object_columns.npy')


# In[ ]:


object_columns


# In[ ]:


for f in object_columns:
    train[f] = train[f].asfactor()
    test[f] = test[f].asfactor()


# In[ ]:


x = train.columns[2:]
y = 'isFraud'
# For binary classification, response should be a factor
train[y] = train[y].asfactor()


# In[ ]:


aml = H2OAutoML(max_models=18, seed=27, max_runtime_secs=30000, nfolds=4,stopping_rounds=10,stopping_metric="AUC") #exclude_algos=["GLM","DRF"]
aml.train(x=x, y=y, training_frame=train)


# In[ ]:


# View the AutoML Leaderboard
lb = aml.leaderboard
# lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
lb.head()


# In[ ]:


# The leader model is stored here
aml.leader


# In[ ]:


preds = aml.predict(test)
preds['p1'].as_data_frame().values.flatten().shape


# In[ ]:


sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
sample_submission.shape


# In[ ]:


sample_submission['isFraud'] = preds['p1'].as_data_frame().values
sample_submission.to_csv('h2o_automl_submission_random.csv', index=False)

