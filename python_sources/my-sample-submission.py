#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('../input/train.csv', index_col='id')
df_test = pd.read_csv('../input/test.csv', index_col='id')


# In[ ]:


df_train.head()


# In[ ]:


df_train['vlrLiquido'].hist(log=True, bins=21)


# In[ ]:


n_train = len(df_train)
df_both = pd.concat([
    df_train.drop(['vlrLiquido'], axis=1),
    df_test
])


# In[ ]:


dummy_cols = ['txNomeParlamentar', 'numMes', 'sgUF', 'sgPartido', 'txtDescricao']


# In[ ]:


X = pd.get_dummies(df_both[dummy_cols]).values


# In[ ]:


X_train = X[:n_train]
y_train = np.log1p(df_train['vlrLiquido'])
X_test = X[n_train:]

# Remove nan do treino
X_train = X_train[np.isfinite(y_train)]
y_train = y_train[np.isfinite(y_train)]


# In[ ]:


model = LinearRegression().fit(X_train, y_train)


# In[ ]:


y_pred_train = model.predict(X_train)
plt.figure(figsize=(10, 10))
plt.scatter(y_train, y_pred_train, s=1, alpha=0.5)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')


# In[ ]:


df_test['vlrLiquido'] = np.expm1(model.predict(X_test).clip(0, 12)).round(2)


# In[ ]:


df_test[['vlrLiquido']].to_csv('submission.csv')


# In[ ]:


get_ipython().system('head submission.csv')

