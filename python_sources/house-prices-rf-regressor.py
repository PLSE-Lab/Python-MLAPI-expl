#!/usr/bin/env python
# coding: utf-8

# **Predicting House Prices Using Random Forest**
# 
# https://satya-python.blogspot.com/

# In[ ]:


# Importing Libraries
import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,make_scorer
from sklearn.preprocessing import StandardScaler,RobustScaler


# In[ ]:


# Reading Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
House_IDs = test["Id"]


# **EDA**

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


str_cols= train.loc[:, train.dtypes=='object'].columns.tolist()


# In[ ]:


for col in str_cols:  train[col] = train[col].fillna(train[col].mode()[0])
for col in str_cols:  test[col] = test[col].fillna(test[col].mode()[0])


# In[ ]:


train = train.fillna(train.median())
test = test.fillna(test.median())


# In[ ]:


train = pd.get_dummies(train, columns=str_cols, drop_first=True)
test = pd.get_dummies(test, columns=str_cols, drop_first=True)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


#train._get_numeric_data()


# In[ ]:


X = train.drop(["Id","SalePrice"], axis=1)
Y = train["SalePrice"]


# In[ ]:


X = X.loc[:, test.columns]
X = X.drop(["Id"], axis=1)


# In[ ]:


test = test.drop(["Id"], axis=1)


# In[ ]:


X.info()


# In[ ]:


test.info()


# In[ ]:


scaler = RobustScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)


# In[ ]:


X.head()


# In[ ]:


test.head()


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=2019)


# In[ ]:


x_train.head()


# In[ ]:


y_train.head()


# **Creating ML model**

# In[ ]:


rf = RandomForestRegressor(n_estimators=2000, random_state = 2019)
# rf = RandomForestRegressor(n_jobs=-1, random_state=2019, n_estimators=1000, oob_score=True, max_features=0.5, 
# max_depth=None, min_samples_leaf=2, max_leaf_nodes=2500, min_impurity_decrease=0.00001, min_impurity_split=None)


# In[ ]:


rf.fit(x_train, y_train)


# In[ ]:


y_pred = rf.predict(x_val)
#y_pred


# In[ ]:


rf.score(x_train, y_train)


# In[ ]:


# cv_score = np.sqrt(-cross_val_score(estimator=rf, X=x_train, y=y_train, cv=3, scoring = make_scorer(mean_squared_error, False)))
# cv_score


# In[ ]:


y_pred = rf.predict(test)


# In[ ]:


submission = {}
submission['Id'] = House_IDs
submission['SalePrice'] = y_pred
submission = pd.DataFrame(submission)

submission = submission[['Id', 'SalePrice']]
submission = submission.sort_values(['Id'])
submission.to_csv("submisision.csv", index=False)
submission.head()


# In[ ]:




