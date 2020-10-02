#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


t = np.linspace(0, 1, 500)
k = 5
growth = 10*t
sine = np.sin(2 * np.pi * k * t)
y = sine + growth
X = np.array([growth, sine]).T


# In[ ]:


t_cut = 300
X_train, X_val = X[:t_cut, :], X[t_cut:, :]
y_train, y_val = y[:t_cut], y[t_cut:]
d_train = xgb.DMatrix(X_train, y_train)
d_all = xgb.DMatrix(X, y)

params = {'objective': 'reg:linear'}
bst = xgb.train(params, d_train)
y_pred = bst.predict(d_all)


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(t, growth, label='growth feature', alpha=0.5)
plt.plot(t, sine, label='sine feature', alpha=0.5)
plt.plot(t, y, label='target')
plt.axvline(t[t_cut], linestyle='--', alpha=0.2, label='training cutoff')
plt.plot(t, y_pred, label='gbrt prediction')
plt.legend(loc='upper left')

