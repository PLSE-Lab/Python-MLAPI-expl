#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle.competitions import nflrush
env = nflrush.make_env()

import pandas as pd


# In[ ]:


df_train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
df_test = pd.DataFrame(columns=df_train.drop(columns=['Yards']).columns)

print('Number of Training Examples = {}'.format(df_train.shape[0]))
print('Number of Training Plays = {}'.format(df_train['PlayId'].nunique()))
print('Number of Training Games = {}'.format(df_train['GameId'].nunique()))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))


# > Because this is a time-series code competition that will be evaluated on future data, you will receive data and make predictions with a time-series API. This API provides plays in the time order in which they occurred in a game. Refer to the starter notebook here for an example of how to complete a submission.
# > 
# > To deter cheating by looking ahead in time, the API has been compiled and the test data encrypted on disk. While it may be possible, you should not decompile or attempt to read the test set outside of the API, as the encryption keys will change during the live scoring portion of the competition. During stage one, we ask that you respect the spirit of the competition and do not submit predictions that incorporate future information or the ground truth.
# 
# **It is clearly showing that looking ahead in time and making predictions based on future is forbidden. Test set should be used for data cleaning and EDA.**
# 
# **Fork this kernel and read the test set with `pd.read_csv(path + 'df_test.csv')`.**

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfor (test, sample_pred) in env.iter_test():\n    df_test = pd.concat([df_test, test])\n    env.predict(sample_pred)\n    \nprint('Number of Test Examples = {}'.format(df_test.shape[0]))\nprint('Number of Test Plays = {}'.format(df_test['PlayId'].nunique()))\nprint('Number of Test Games = {}'.format(df_test['GameId'].nunique()))\nprint('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))")


# In[ ]:


df_test.head()


# In[ ]:


# Test set is chronologically sorted
(df_test['PlayId'] == sorted(df_test['PlayId'])).all()


# In[ ]:


df_test.to_csv('df_test.csv', chunksize=50000, index=False)

