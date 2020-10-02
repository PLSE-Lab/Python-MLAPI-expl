#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from catboost import CatBoostClassifier, FeaturesData, Pool
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


# In[ ]:


ls ../input


# In[ ]:


train = pd.read_csv("../input/Train.csv")
test = pd.read_csv('../input/Test.csv')
sample = pd.read_csv("../input/SampleSubmission.csv")
info_variables = pd.read_csv('../input/VariableDefinitions.csv')


# In[ ]:


train.bank_account =train.bank_account.map({"Yes":1, "No":0})


# In[ ]:


info_variables.head(10)


# In[ ]:


train.head()


# In[ ]:


cols = ['country', 'year',  'location_type',
       'cellphone_access', 'household_size', 'age_of_respondent',
       'gender_of_respondent', 'relationship_with_head', 'marital_status',
       'education_level', 'job_type']


# In[ ]:


categorical_features = [col for col in cols if train[col].dtypes=="O"]
categorical_features


# In[ ]:


for col in categorical_features:
    le = preprocessing.LabelEncoder()
    le.fit(pd.concat([train[col],test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


n_folds=20
num_boost_round=1000
kf = StratifiedKFold(n_splits = n_folds, random_state = 5168, shuffle = True)
oof_lgb= np.zeros(train.shape[0])
predictions_lgb= np.zeros(test.shape[0])
params = {'objective' : 'binary',
          'boosting_type' : 'gbdt',
          'num_threads': 4,
          'learning_rate': 0.1,
          'max_depth' : 12,
          'tree_learner' : 'serial',
          'feature_fraction': 0.5,
          'bagging_freq' : 5,
          'bagging_fraction':0.9,
          'verbosity': 1,
          "metric":"binary_error",
          'seed' : 44001}
for train_index, val_index in kf.split(train, train.bank_account):
    X_train, X_val = train[cols].iloc[train_index], train[cols].iloc[val_index]
    y_train, y_val = train.bank_account[train_index], train.bank_account[val_index]
    
    dtrain=lgbm.Dataset(data=X_train,label=y_train)
    dval=lgbm.Dataset(data=X_val,label=y_val)
    model = lgbm.train(params=params,train_set=dtrain,num_boost_round=num_boost_round,
                        valid_sets=(dtrain, dval),early_stopping_rounds=50,verbose_eval=50)
    
    best_iteration = model.best_iteration
   
    oof_lgb[val_index]= model.predict(X_val,num_iteration=best_iteration)>0.5
    predictions_lgb= ((model.predict(test[cols],num_iteration=best_iteration)) >0.5)/n_folds
    
print(1-accuracy_score(oof_lgb,train.bank_account))
    


# In[ ]:


train2 = pd.read_csv("../input/Train.csv")
test2 = pd.read_csv('../input/Test.csv')


# In[ ]:


test["unique_id"] = test2["uniqueid"]+" x "+test2["country"]
test["bank_account"]=(predictions_lgb)>0


# In[ ]:


test[["unique_id","bank_account"]].bank_account.astype(int).sum()


# In[ ]:


test[["bank_account"]]=test[["bank_account"]].astype(int)


# In[ ]:


test


# In[ ]:


test[["unique_id","bank_account"]].to_csv("sample.csv",index=False)


# In[ ]:




