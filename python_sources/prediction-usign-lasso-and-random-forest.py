#!/usr/bin/env python
# coding: utf-8

# prediction usign lasso and random forest

# In[ ]:


import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib as mp
from scipy.stats import skew
import csv
import math
from sklearn import linear_model
import xgboost as xgb

from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,LassoCV,LassoLarsCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor as rfr

data_train=[]
data_test=[]
data_final=[]
price=[]

# def preprocess(lines):
# 	lines["SalePrice"]=np.log1p(lines["SalePrice"])
# 	numeric_feats = lines.dtypes[lines.dtypes != "object"].index



def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, data_train, price, scoring="mean_squared_error", cv = 5))
    return(rmse)
test=pd.read_csv('../input/test.csv')	
train=pd.read_csv('../input/train.csv')

# print train.head()

# preprocess(test);
# preprocess(train);

data_final = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = data_final.dtypes[data_final.dtypes != "object"].index
#categorical_feats= data_final.dtypes[data_final.dtypes == "object"].index

# print numeric_feats
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
# numeric_feats=numeric_feats.index

#print skewed_feats

data_final[skewed_feats] = np.log1p(data_final[skewed_feats])

#print 1,data_final.shape

data_final = pd.get_dummies(data_final)

#filling NA's with the mean of the column:
data_final = data_final.fillna(data_final[:train.shape[0]].mean())

#creating matrices for sklearn:
data_train = data_final[:train.shape[0]]
data_test = data_final[train.shape[0]:]

price = train.SalePrice


rf_model=rfr(n_estimators=100)
rf_model.fit(data_train,price)

rmse1=np.sqrt(-cross_val_score(rf_model,data_train,price,scoring="mean_squared_error",cv=5))

#print "Root Mean Square Error"

#print rmse1.mean()

lasso_model = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(data_train, price)

rmse2=rmse_cv(lasso_model)

#print rmse2.mean()

dtrain = xgb.DMatrix(data_train, label = price)
dtest = xgb.DMatrix(data_test)

params = {"max_depth":2, "eta":0.05}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()



xgb_model = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.05) #the params were tuned using xgb.cv
xgb_model.fit(data_train, price)



rf_preds = np.expm1(rf_model.predict(data_test))
lasso_preds = np.expm1(lasso_model.predict(data_test))
xgb_preds=np.expm1(xgb_model.predict(data_test))

final_result=0.4*lasso_preds+0.3*rf_preds+0.3*xgb_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":final_result})
solution.to_csv("kaggle_sol.csv", index = False)

