#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


def RMSLE(est, X, y):
    return np.sqrt(mean_squared_log_error(y_true=y, y_pred=est.predict(X)))


# In[ ]:


df_train = pd.read_csv('../input/train.csv', index_col='id')
df_test = pd.read_csv('../input/test.csv', index_col='id')

train = df_train[['x1', 'x2', 'x3']].values
test = df_test.values

y = df_train['target']


# In[ ]:


feat = np.vstack((train, test))
feat.shape


# In[ ]:


pf = PolynomialFeatures(25)
pol_feat = pf.fit_transform(feat)
pol_feat.shape


# In[ ]:


X_train, X_test = pol_feat[:7500, :], pol_feat[7500:, :]


# In[ ]:


cross_val_score(LinearRegression(), X=X_train, y=y, cv=3, scoring=RMSLE)


# In[ ]:


predict = LinearRegression().fit(X_train, y).predict(X_test)


# In[ ]:


pd.DataFrame({
    'id': df_test.index, 
    'target': predict
}).to_csv('submit_pol_feat.csv', index=False)

