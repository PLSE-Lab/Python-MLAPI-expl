#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
id_test = test.ID
train.sample(3)


# In[ ]:


y_train = train["y"]
x_train = train.drop(["ID", "y"], axis=1)
x_test = test.drop(["ID"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values) + list(x_test[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))


# In[ ]:


xgb_params = {
    'eta': 0.02,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


# In[ ]:


cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


# In[ ]:


num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(8, 64))
xgb.plot_importance(model, height=0.5, ax=ax)


# In[ ]:


y_predict = model.predict(dtest)
output = pd.DataFrame({'ID': id_test, 'y': y_predict})
output.head()


# In[ ]:


output.to_csv('xgbSub.csv', index=False)


# In[ ]:




