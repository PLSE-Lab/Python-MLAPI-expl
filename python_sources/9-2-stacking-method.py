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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import time
import datetime
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")


# In[ ]:


train.dropna(how = "any",axis = 0,inplace =True)


# In[ ]:


combine = pd.concat([train,test],axis=0)
combine.reset_index(inplace = True)
combine = pd.get_dummies(data=combine,columns=["matchType"])


# In[ ]:


train = combine[combine["winPlacePerc"].notna()]
test = combine[combine["winPlacePerc"].isna()]
features = ['DBNOs',
 'heals',
 'killPlace',
 'killStreaks',
 'kills',
 'matchDuration',
 'numGroups',
 'revives',
 'rideDistance',
 'roadKills',
 'swimDistance',
 'teamKills',
 'walkDistance',
 'winPoints',
 'matchType_flarefpp',
 'matchType_flaretpp',
 'matchType_normal-duo',
 'matchType_normal-duo-fpp',
 'matchType_normal-solo',
 'matchType_normal-squad',
 'matchType_solo-fpp',
 'matchType_squad',
 'matchType_squad-fpp']
target = ["winPlacePerc"]


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import  SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold,GridSearchCV
from xgboost import XGBRegressor

def model_selection(X_train,Y_train,model,grid_params):
    gscv = GridSearchCV(estimator=model,param_grid=grid_params,scoring='neg_mean_absolute_error',cv=3)
    gscv.fit(X_train,Y_train)
    best_params = gscv.best_params_
    score = gscv.best_score_
    model.set_params(**best_params)
    print(best_params)
    return model,score
def get_regression_models(X_train,Y_train):
    reg_1 = Lasso()
    paras_1 = {'alpha': [0.001, 0.01, 0.1]}
    
#     reg_2 = SVR()
#     paras_2 = {'C': [0.5], 'gamma': [0.001, 0.01], 'kernel': ['rbf']}
        
    reg_3 = RandomForestRegressor()
    paras_3 = {'n_estimators': [150, 100,200], 'max_depth': [5, 10, 15], }
        
    reg_4 = AdaBoostRegressor()
    paras_4 = {'n_estimators': [150, 100,200], 'learning_rate': [0.03, 0.1, 0.5]}
        
    reg_5 = GradientBoostingRegressor()
    paras_5 = {'learning_rate' :[0.003,0.01,0.1,0.5], 'n_estimators': [150, 100,200]}
    
    models = [reg_1,reg_3,reg_4,reg_5]
    paras = [paras_1,paras_3,paras_4,paras_5]
    kf = KFold(n_splits=len(models))
    index = 0
    for train,test in kf.split(X_train):
        X = X_train[train]
        Y = Y_train[train]
        models[index],error = model_selection(X,Y,models[index],paras[index])
        index += 1
    return models

def stacking_models(models,X_train,X_test,Y_train,kfo):
    kf = KFold(n_splits=kfo)
    blend_train = np.zeros([X_train.shape[0],len(models)],dtype=float)
    blend_test = np.zeros([X_test.shape[0],len(models)],dtype=float)
    index = 0
    for model in models:
        begin = 0
        end = 0
        for train, test in kf.split(X_train):
            model.fit(X_train[train], Y_train[train])
            pre_Y = model.predict(X_train[test])
            temp_blend_test = model.predict(X_test)
            blend_test[:, index] = temp_blend_test
            end = pre_Y.shape[0] + begin
            blend_train[begin:end, index] = pre_Y
            begin = end
        index += 1
        print("one model")
    return blend_train,blend_test


# In[ ]:


rf=RandomForestRegressor(n_estimators=260,max_depth=17,min_samples_split=5,min_samples_leaf=5,max_features=9)
gbdt=GradientBoostingRegressor(n_estimators=360,learning_rate=0.1,max_depth=3,min_samples_split=6,min_samples_leaf=1)
XGB=XGBRegressor(n_estimators = 290,booster='gbtree',learning_rate=0.1,max_depth=5,subsample=0.7,colsample_bytree=0.9)

models = [rf,gbdt,XGB]


# In[ ]:


final_models = []
result = []
for i in range(1):
    temp = train.sample(frac=1.0,random_state = i)
    X_train = temp[features].values
    Y_train = temp[target].values
    X_test = test[features].values
#     models = get_regression_models(X_train,Y_train)
    blend_train,blend_test = stacking_models(models,X_train,X_test,Y_train,3)
    from xgboost import XGBRegressor
    final_model = XGBRegressor()
    paras = {'n_estimators' : [50,100,200,300,400],"learning_rate" : [0.0001,0.003,0.01,0.1],"max_depth" : [2,3,5,10],
        "gamma" : [ 0.003,0.01]}
    final_model,error = model_selection(blend_train,Y_train,final_model,paras)
    final_model.fit(blend_train,Y_train)
    final_models.append(final_model)
    res_Y = final_model.predict(blend_test)
    result.append(res_Y)
    print(error)


# In[ ]:


for index in range(len(result)):
    if index ==0:
        continue
    result[0] = result[0]  + result[index]
res_Y = result[0] / len(result)


# In[ ]:


res_Y = pd.DataFrame(res_Y,columns=["winPlacePerc"])
usid = test["Id"]
usid = pd.DataFrame(usid,columns=["Id"])
usid = usid.reset_index()["Id"]
res_Y = res_Y.reset_index()["winPlacePerc"]

predict_PUBG = pd.concat([usid,res_Y],axis=1,ignore_index="True")
predict_PUBG.columns = ["Id","winPlacePerc"]


# In[ ]:


predict_PUBG.to_csv("submission.csv",index=False,encoding = "utf-8")

