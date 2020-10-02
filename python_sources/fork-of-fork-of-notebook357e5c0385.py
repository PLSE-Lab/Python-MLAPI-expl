#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

def xgboost_prediction(train,labels,test):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["min_child_weight"] = 100
    params["subsample"] = 0.6
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 9
    
    paramslist = list(params.items())
    
    offset = 4000

    num_rounds = 10000
    xgb_test = xgb.DMatrix(test)
 
    xgb_train = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgb_value = xgb.DMatrix(train[:offset,:], label=labels[:offset])
    
    listforprint = [(xgb_train, 'train'),(xgb_value, 'value')]
    model = xgb.train(paramslist, xgb_train, num_rounds, listforprint, early_stopping_rounds=120)
    prediction1 = model.predict(xgb_test,ntree_limit=model.best_iteration)

    train = train[::-1,:]
    labels = np.log(labels[::-1])

    xgb_train = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgb_value = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    listforprint = [(xgb_train, 'train'),(xgb_value, 'value')]
    model = xgb.train(paramslist, xgb_train, num_rounds, listforprint, early_stopping_rounds=120)
    prediction2 = model.predict(xgb_test,ntree_limit=model.best_iteration)

    prediction = (prediction1)*1.4 + (prediction2)*8.6
    return prediction

train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

train_temp = train
test_temp = test

train_temp.drop('T2_V10', axis=1, inplace=True)
train_temp.drop('T2_V7', axis=1, inplace=True)
train_temp.drop('T1_V13', axis=1, inplace=True)
train_temp.drop('T1_V10', axis=1, inplace=True)

test_temp.drop('T2_V10', axis=1, inplace=True)
test_temp.drop('T2_V7', axis=1, inplace=True)
test_temp.drop('T1_V13', axis=1, inplace=True)
test_temp.drop('T1_V10', axis=1, inplace=True)

columns = train.columns
test_index = test.index

train_temp = np.array(train_temp)
test_temp = np.array(test_temp)

for i in range(train_temp.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(list(train_temp[:,i]) + list(test_temp[:,i]))
    train_temp[:,i] = le.transform(train_temp[:,i])
    test_temp[:,i] = le.transform(test_temp[:,i])

train_temp = train_temp.astype(float)
test_temp = test_temp.astype(float)

prediction1 = xgboost_prediction(train_temp,labels,test_temp)

#model_2 building

train = train.T.to_dict().values()
test = test.T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

prediction2 = xgboost_prediction(train,labels,test)

prediction = prediction1 + prediction2

prediction = pd.DataFrame({"Id": test_index, "Hazard": prediction})
prediction = prediction.set_index('Id')
prediction.to_csv('result.csv')

