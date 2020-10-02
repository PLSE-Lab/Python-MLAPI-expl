#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://autogluon.mxnet.io/
get_ipython().system('pip install --upgrade mxnet')
get_ipython().system('pip install autogluon')


# In[ ]:


# https://github.com/awslabs/autogluon
from autogluon import TabularPrediction as task


train_data = task.Dataset(file_path='../input/titanic/train.csv')
test_data = task.Dataset(file_path='../input/titanic/test.csv')
predictor = task.fit(train_data=train_data, label='Survived')
y_pred = predictor.predict(test_data)


# In[ ]:


import pandas as pd


sub = pd.read_csv('../input/titanic/gender_submission.csv')
sub['Survived'] = y_pred


# In[ ]:


sub.to_csv('submission.csv', index=False)
sub.head()

