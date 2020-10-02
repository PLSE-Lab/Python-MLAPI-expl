#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print((os.listdir('../input/')))


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[ ]:


df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')
df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')


# In[ ]:


test_index=df_test['Unnamed: 0'] #copying test index for later


# In[ ]:


X = df_train.loc[:, 'V1': 'V16']
y = df_train.Class

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)


# In[ ]:


tr = np.array(df_train.values[:,1:-1])
train_label = np.array(df_train.values[:,-1])
train_data = np.array(tr.astype(np.float))
test_data = np.array(df_test.values[:,1:])
trainX, trainY = train_data,train_label
testX = test_data
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from xgboost.sklearn import XGBClassifier
smt = SMOTE()
X_train,y_train = smt.fit_sample(trainX,trainY)


# In[ ]:


def modelfit(alg, dtrain, y,useTrainCV=False, cv_folds=5, early_stopping_round=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_round, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
        print("hwllo")
        alg.fit(dtrain, y,eval_metric='auc')
        
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)
        
    y = np.eye(2)[y]

    if  useTrainCV:
        print ("\nModel Report")
        print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
    return dtrain_predprob  


# In[ ]:



from xgboost import XGBClassifier

xgb1 = XGBClassifier()
modelfit(xgb1, X_train, y_train,True)


# In[ ]:


predic=modelfit(xgb1, testX,None)


# In[ ]:


predic=modelfit(xgb1, testX,None)


# In[ ]:


result.to_csv('output.csv', index=False)

