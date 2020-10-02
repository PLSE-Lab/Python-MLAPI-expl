#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns = ['Sex']

big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = big_X


# In[ ]:


le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

gbm = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05).fit(train_X,train_y)
predictions = gbm.predict(test_X)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],
                         'Survived':predictions})
submission.to_csv('submission.csv',index=False)

