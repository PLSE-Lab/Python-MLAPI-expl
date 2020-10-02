#!/usr/bin/env python
# coding: utf-8

# # Problem Statement - Mobility Analytics

# Welcome to Sigma Cab Private Limited - a cab aggregator service. Their customers can download their app on smartphones and book a cab from any where in the cities they operate in. They, in turn search for cabs from various service providers and provide the best option to their client across available options. They have been in operation for little less than a year now. During this period, they have captured surge_pricing_type from the service providers.
# 
# You have been hired by Sigma Cabs as a Data Scientist and have been asked to build a predictive model, which could help them in predicting the surge_pricing_type pro-actively. This would in turn help them in matching the right cabs with the right customers quickly and efficiently.

# Github Link: https://github.com/bilalProgTech/online-data-science-ml-challenges/tree/master/AV-Janata-Hack-Mobility-Analytics

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/mobilityanalytics/train_Wc8LBpr.csv')
test = pd.read_csv('../input/mobilityanalytics/test_VsU9xXK.csv')


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


combine = train.append(test)
combine.shape


# In[ ]:


combine.isnull().sum()


# In[ ]:


combine.dtypes


# In[ ]:


combine.columns


# In[ ]:


combine['Cancellation_Last_1Month'].value_counts()


# In[ ]:


bins= [0, 1, 2, 3, 8]
labels = ['None','Once', 'Twice','More_Than_Thrice']
combine['Cancellation_Last_1Month'] = pd.cut(combine['Cancellation_Last_1Month'], bins=bins, labels=labels, right=False)
combine['Cancellation_Last_1Month'].value_counts()


# In[ ]:


combine['Confidence_Life_Style_Index'].value_counts()


# In[ ]:


combine['Confidence_Life_Style_Index'].fillna('Unknown', inplace=True)
combine['Confidence_Life_Style_Index'].value_counts()


# In[ ]:


combine['Customer_Rating'].describe()


# In[ ]:


combine['Customer_Since_Months'].value_counts()


# In[ ]:


from sklearn.preprocessing import scale
combine['Customer_Since_Months'].fillna(-1, inplace=True)
combine['Customer_Since_Months'] = scale(combine['Customer_Since_Months'])
combine['Customer_Since_Months'].describe()


# In[ ]:


combine['Destination_Type'].value_counts()


# In[ ]:


combine['Gender'].value_counts()


# In[ ]:


combine['Life_Style_Index'].describe()


# In[ ]:


combine['Life_Style_Index'].fillna(combine['Life_Style_Index'].mean(), inplace=True)
combine['Life_Style_Index'].describe()


# In[ ]:


combine['Trip_Distance'].describe()


# In[ ]:


combine['Trip_Distance'] = np.log(combine['Trip_Distance'])
combine['Trip_Distance'].describe()


# In[ ]:


combine['Type_of_Cab'].value_counts()


# In[ ]:


combine['Type_of_Cab'].fillna('Unknown', inplace=True)
combine['Type_of_Cab'].value_counts()


# In[ ]:


combine['Var1'].describe()


# In[ ]:


combine['Var1'].fillna(combine['Var1'].mean(), inplace=True)
combine['Var1'] = np.log(combine['Var1'])
combine['Var1'].describe()


# In[ ]:


combine['Var2'].describe()


# In[ ]:


combine['Var2'] = np.log(combine['Var2'])
combine['Var2'].describe()


# In[ ]:


combine['Var3'].describe()


# In[ ]:


combine['Var3'] = np.log(combine['Var3'])
combine['Var3'].describe()


# In[ ]:


combine.isnull().sum()


# In[ ]:


combine = pd.get_dummies(combine.drop('Trip_ID', axis=1))
combine.shape


# In[ ]:


combine.head()


# In[ ]:


X = combine[combine['Surge_Pricing_Type'].isnull()!=True].drop(['Surge_Pricing_Type'], axis=1)
y = combine[combine['Surge_Pricing_Type'].isnull()!=True]['Surge_Pricing_Type']

X_test = combine[combine['Surge_Pricing_Type'].isnull()==True].drop(['Surge_Pricing_Type'], axis=1)

X.shape, y.shape, X_test.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[ ]:


from lightgbm import LGBMClassifier
model = LGBMClassifier(boosting_type='gbdt',
                       max_depth=5,
                       learning_rate=0.05,
                       n_estimators=5000,
                       min_child_weight=0.01,
                       colsample_bytree=0.5,
                       random_state=1994,
                       objective='multiclass')

model.fit(x_train,y_train,
          eval_set=[(x_train,y_train),(x_val, y_val.values)],
          early_stopping_rounds=100,
          verbose=200)

pred_y = model.predict(x_val)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_val, pred_y))
confusion_matrix(y_val,pred_y)


# In[ ]:


err = []
y_pred_tot_lgm = []

from sklearn.model_selection import StratifiedKFold

fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=2020)
i = 1
for train_index, test_index in fold.split(X, y):
    x_train, x_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y[train_index], y[test_index]
    m = LGBMClassifier(boosting_type='gbdt',
                       max_depth=5,
                       learning_rate=0.05,
                       n_estimators=5000,
                       min_child_weight=0.01,
                       colsample_bytree=0.5,
                       random_state=1994,
                       objective='multiclass')
    m.fit(x_train, y_train,
          eval_set=[(x_train,y_train),(x_val, y_val)],
          early_stopping_rounds=200,
          verbose=200)
    pred_y = m.predict(x_val)
    print(i, " err_lgm: ", accuracy_score(y_val, pred_y))
    err.append(accuracy_score(y_val, pred_y))
    pred_test = m.predict(X_test)
    i = i + 1
    y_pred_tot_lgm.append(pred_test)


# In[ ]:


np.mean(err, 0)


# In[ ]:


err[3]


# In[ ]:


submission = pd.DataFrame()
submission['Trip_ID'] = test['Trip_ID']
submission['Surge_Pricing_Type'] = y_pred_tot_lgm[3]
submission.to_csv('LGBM.csv', index=False, header=True)
submission.shape


# In[ ]:


submission.head()


# In[ ]:


submission['Surge_Pricing_Type'].value_counts()

