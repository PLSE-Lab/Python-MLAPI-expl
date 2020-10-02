#!/usr/bin/env python
# coding: utf-8

# Trying some things out: a bit of xgboost and EDA.

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.SalePrice.hist()


# In[ ]:


np.log(train.SalePrice).hist()


# In[ ]:


train["SalePrice"] = np.log(train["SalePrice"]) + 1


# In[ ]:


train.head(20)


# How many NaN's do we have for each feature?

# In[ ]:


train.isnull().values.sum(axis = 0)


# In[ ]:


pd.DataFrame(train.isnull().values.sum(axis = 0), index = train.columns)


# In[ ]:


A decent amount of NA's.


# In[ ]:


Now let's create dummy variables:


# In[ ]:


X_train = pd.get_dummies(train.loc[:, 'MSSubClass':'SaleCondition'])
y_train = train.SalePrice


# In[ ]:


X_train.shape


# In[ ]:


X_train.head(3)


# In[ ]:


dtrain = xgb.DMatrix(X_train, label = y_train)
param = {'max_depth':1, 'eta':0.3} #booster : "gblinear"}
model_xgb = xgb.cv(param, dtrain, num_boost_round=1000, early_stopping_rounds=10)


# - keep the learning rate (eta) constant at 0.3
# - try max_depth from 1 to 12
# - let the CV decide the optimal number of trees for each tree depth by using `early_stopping_rounds=10`

# In[ ]:


rmse_err = []
for i in range(1,13):
    param = {'max_depth':i}
    cv_error = xgb.cv(param, dtrain, num_boost_round=1000, early_stopping_rounds=10, seed=21)
    rmse_err.append(cv_error["test-rmse-mean"].min())


# In[ ]:


pd.Series(rmse_err, index = range(1,13)).plot()


# So it seems like depth 2 trees are good enough:
# 

# In[ ]:


param = {'max_depth':2, 'eta':0.3} #booster : "gblinear"}
cv_error = xgb.cv(param, dtrain, num_boost_round=1000, early_stopping_rounds=10, seed=21)


# In[ ]:


cv_error["test-rmse-mean"].min() #CV rmse


# In[ ]:


nrounds = cv_error.shape[0]


# In[ ]:


nrounds


# In[ ]:


param = {'max_depth':3, 'eta':0.3}
model = xgb.train(param, dtrain, num_boost_round=nrounds)


# In[ ]:


importance = model.get_fscore()
importance = pd.DataFrame(list(importance.values()), index = importance.keys(), columns = ["f_score"])


# In[ ]:


importance.sort("f_score", ascending = False).head(30)


# In[ ]:


train[["LotFrontage", "SalePrice"]].plot(x = "LotFrontage", y = "SalePrice", kind="scatter")


# In[ ]:


important_feats = importance.sort_values("f_score", ascending = False).head(4).index


# In[ ]:


important_feats


# In[ ]:


imp_train = train[important_feats]


# In[ ]:


imp_train["SalePrices"] = train["SalePrice"]


# In[ ]:


imp_train.head()


# In[ ]:


sns.pairplot(imp_train.dropna())


# Some of the features are skewed, let log-transform them.

# In[ ]:


imp_train.LotArea = np.log(imp_train.LotArea)
imp_train.BsmtUnfSF = np.log(imp_train.LotArea)
imp_train.LotFrontage = np.log(imp_train.LotFrontage)
imp_train.SalePrices = np.log(imp_train.SalePrices)


# In[ ]:


sns.pairplot(imp_train.dropna())


# In[ ]:




