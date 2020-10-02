#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test['casual'] = np.nan
test['registered'] = np.nan
test['count'] = np.nan


# In[ ]:


test.head()


# In[ ]:


print(train.info())
print(test.info())


# In[ ]:


train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])


# In[ ]:


print(train.info())


# In[ ]:


train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['DOW'] = train['datetime'].dt.dayofweek
train['hour'] = train['datetime'].dt.hour


# In[ ]:


train.head()


# In[ ]:


test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['DOW'] = test['datetime'].dt.dayofweek
test['hour'] = test['datetime'].dt.hour


# In[ ]:


col = ['workingday','temp','year','month','DOW', 'hour']
x = train[col]
y = train['count']


# In[ ]:


X_test = test[col]
Y_test = test['count']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train , X_valid ,Y_train , Y_valid = train_test_split(x,y,test_size = 0.25, random_state = 201)


# In[ ]:


def RSMLE(predictions , realizations):
    predictions_use = predictions.clip(0)
    rmsle = np.sqrt(np.mean(np.array(np.log(predictions_use+1)-np.log(realizations+1))**2))
    return rmsle


# **DecisionTreeRegressor
# **

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(min_samples_split=25 , random_state=201)
dtr_model = dtr.fit(X_train,Y_train)


# In[ ]:


dtr_pred = dtr_model.predict(X_valid)


# In[ ]:


pd.DataFrame(dtr_model.feature_importances_,index=col)


# In[ ]:


plt.figure(figsize=(7,7))
plt.scatter(dtr_pred,Y_valid, s=0.2)
plt.xlim(-100,1200)
plt.ylim(-100,1200)
plt.plot([-100,1200],[-100,1200],color = 'pink', linestyle = '-', linewidth =7)
plt.suptitle("Predicted / Actual", fontsize = 20 )
plt.xlabel('pred')
plt.ylabel("y_valid")


# In[ ]:


RSMLE(dtr_pred,Y_valid)


#  **RandomForestRegressor
# **

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regress = RandomForestRegressor(n_estimators=500, max_features=4,min_samples_leaf=5, random_state=201)
regress.fit(X_train,Y_train)


# In[ ]:


predict = regress.predict(X_valid)


# In[ ]:


plt.figure(figsize=(7,7))
plt.scatter(predict,Y_valid, s=1.9)
plt.xlim(-100,1200)
plt.ylim(-100,1200)
plt.plot([-100,1200],[-100,1200],color = 'purple', linestyle = '-', linewidth =5)
plt.suptitle("Predicted / Actual", fontsize = 20 )
plt.xlabel('pred')
plt.ylabel("y_valid")


# In[ ]:


RSMLE(predict,Y_valid)


# ***XGBoosted Tree Model
# *******

# In[ ]:


import xgboost as xgb
xgb_train = xgb.DMatrix(X_train,label=Y_train)
xgb_valid = xgb.DMatrix(X_valid)


# In[ ]:


num_round_for_cv = 500
param = { 'max_depth': 6 , 'eta':0.1 , 'seed' : 201 , 'objective' : 'reg:linear'}


# In[ ]:


xgb.cv(param,xgb_train,num_round_for_cv,nfold=5,show_stdv=False,verbose_eval=True,as_pandas=False)


# In[ ]:


num_round = 400
xg_model = xgb.train(param,xgb_train,num_round)
xg_pred = xg_model.predict(xgb_valid)


# In[ ]:


xg_model.get_fscore()


# In[ ]:


xgb.plot_importance(xg_model)


# In[ ]:


plt.figure(figsize=(7,7))
plt.scatter(xg_pred,Y_valid, s=0.6)
plt.xlim(-100,1200)
plt.ylim(-100,1200)
plt.plot([-100,1200],[-100,1200],color = 'orange', linestyle = '-', linewidth =5)
plt.suptitle("Predicted / Actual", fontsize = 20 )
plt.xlabel('xg_pred')
plt.ylabel("y_valid")


# In[ ]:


RSMLE(xg_pred,Y_valid)


# **Making Predictions for testing set
# **

# In[ ]:


test_dt =  dtr.fit(x,y)
predict_dt = test_dt.predict(X_test)
dt_clipped = pd.Series(predict_dt.clip(0))


# In[ ]:


test_rf =  regress.fit(x,y)
predict_rt = test_rf.predict(X_test)
rf_clipped = pd.Series(predict_rt.clip(0))


# In[ ]:


xgbtrain = xgb.DMatrix(x,label=y)
xgbtest = xgb.DMatrix(X_test)
xgmodel = xgb.train(param,xgbtrain,num_round)
xgpred = xgmodel.predict(xgbtest)
xg_clipped = pd.Series(xgpred.clip(0))


# random Forest has less RSMLE  compared to xg Boost and Decison tree

# In[ ]:




