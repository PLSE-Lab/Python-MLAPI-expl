#!/usr/bin/env python
# coding: utf-8

# **I am experimenting with simple and fast solutions. This LightGBM model scored above 0.96 on the public leaderboard. It took approx. 2 minutes to train on a cpu. I did not do any parameter tuning.**
# 
# It's nice to discover that it's possible to get such a high accuracy with so little work.

# <hr>

# In[ ]:


import pandas as pd


# In[ ]:


# read the data into a pandas datafrome
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print(df_train.shape)
print(df_test.shape)


# ### Define X and y

# In[ ]:


X = df_train.drop('label', axis=1)
y = df_train['label']

X_test = df_test

print(X.shape)
print(y.shape)
print(X_test.shape)


# ### LightGBM Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom lightgbm import LGBMClassifier\n\nlgbm = LGBMClassifier(objective='multiclass', random_state=5)\n\nlgbm.fit(X, y)\n\ny_pred = lgbm.predict(X_test)")


# In[ ]:


y_pred.shape


# ### Create a submission file

# In[ ]:


# The index should start from 1 instead of 0
df = pd.Series(range(1,28001),name = "ImageId")

ID = df

submission = pd.DataFrame({'ImageId':ID, 
                           'Label':y_pred, 
                          }).set_index('ImageId')

submission.to_csv('mnist_lgbm.csv', columns=['Label']) 


# In[ ]:


submission.head()


# Thank you for reading.

# In[ ]:




