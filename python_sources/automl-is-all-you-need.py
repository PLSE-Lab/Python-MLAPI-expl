#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U h2o')


# In[ ]:


import pandas as pd

import h2o
from h2o.automl import H2OAutoML

h2o.init()


# Of course, we can analyse data, do model selection, fine-tune hyperparams. But what about something else ...

# In[ ]:


train = h2o.import_file('/kaggle/input/infopulsehackathon/train.csv')
test = h2o.import_file('/kaggle/input/infopulsehackathon/test.csv')

sample_submission = pd.read_csv('/kaggle/input/infopulsehackathon/sample_submission.csv')

y = "Energy_consumption"
x = list(train.columns) 
x.remove(y)


# # Magic

# In[ ]:


aml = H2OAutoML(max_runtime_secs = 18000, sort_metric='mse', stopping_metric='MSE' , stopping_rounds=100)
aml.train(x = x, y = y, training_frame = train)

aml.leaderboard


# # Some more magic

# In[ ]:


sample_submission['Energy_consumption'] = aml.leader.predict(test).as_data_frame()['predict']
sample_submission.loc[sample_submission['Energy_consumption'] < 0, 'Energy_consumption'] = 0


# In[ ]:


sample_submission[['Id','Energy_consumption']].to_csv('submission.csv', index=False)


# # Final thougths

# You can GO DEEPER! or ANALYSE DATA! But you can just blend AutoMLs. According to [this discussion](https://www.kaggle.com/c/infopulsehackathon/discussion/119729#latest-685093) will it dare it? ;)
