#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Data exploration

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.corr());


# In[ ]:


sns.pairplot(train);


# In[ ]:


train.min()


# In[ ]:


train.max()


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


sns.pairplot(test);


# In[ ]:


test.head()


# In[ ]:


test.min()


# In[ ]:


test.max()


# # Metric

# In[ ]:


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


# # Train

# In[ ]:


X = train.drop(['id', 'target'], axis=1)
y = train.target


# In[ ]:


X['bias'] = 1


# In[ ]:


X_test = test.drop('id', axis=1)
X_test['bias'] = 1


# ## Cross-validation

# In[ ]:


y_pred = []
k = 10
batch = X.shape[0] // k
scores = []
for i in range(k):
    X_train, y_train = pd.concat((X[:i*batch], X[(i+1)*batch:])), pd.concat((y[:i*batch], y[(i+1)*batch:]))
    X_val, y_val = X[i*batch: (i+1)*batch], y[i*batch: (i+1)*batch]
    coef = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    scores.append(rmsle(y_val, X_val.dot(coef)))
    y_pred.append(X_test.dot(coef))
y_pred = np.array(y_pred)
y_pred = y_pred.T.mean(axis=1)
print('RMSLE:', np.mean(scores))


# In[ ]:


pd.DataFrame({'id': test.id, 'target': y_pred}).to_csv('submission.csv', index=None)


# In[ ]:




