#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("/kaggle/input/kobe-bryant-shot-selection/data.csv.zip")
data.head()


# In[ ]:


data.columns


# In[ ]:


data["shot_made_flag"]


# In[ ]:


predict=data[data["shot_made_flag"].isnull()]
predict.shape


# In[ ]:


data.shape


# In[ ]:


deta2=data.dropna(subset=['shot_made_flag'])
deta2.shape


# In[ ]:


deta2.head()


# In[ ]:


del deta2["team_id"]
del deta2["team_name"]
del deta2["matchup"]
del deta2["opponent"]


# In[ ]:


deta2.head()


# In[ ]:


x=deta2[["combined_shot_type","minutes_remaining","period","shot_distance","shot_zone_area"]]
y=deta2[["shot_made_flag"]]


# In[ ]:


y=y.astype(int)


# In[ ]:


x.head()


# In[ ]:


from sklearn import preprocessing
lecombined = preprocessing.LabelEncoder()
combine=lecombined.fit_transform(x["combined_shot_type"])
x["combined_shot_type"]=combine


# In[ ]:


leshotzone = preprocessing.LabelEncoder()
shotzone=leshotzone.fit_transform(x["shot_zone_area"])
x["shot_zone_area"]=shotzone


# In[ ]:


x.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
tpot = TPOTClassifier(verbosity=2,max_time_mins=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1923)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))


# In[ ]:


print(tpot.score(x_test, y_test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train, y_train)
ypred=rfc.predict(x_test)
import sklearn.metrics as metrik
print(metrik.confusion_matrix(ypred,y_test))
print(metrik.accuracy_score(ypred,y_test))


# In[ ]:


hadi=predict[["combined_shot_type","minutes_remaining","period","shot_distance","shot_zone_area"]]

combine=lecombined.transform(hadi["combined_shot_type"])
hadi["combined_shot_type"]=combine

shotzone=leshotzone.transform(hadi["shot_zone_area"])
hadi["shot_zone_area"]=shotzone

tahminler=tpot.predict(hadi)
tahminler


# In[ ]:


#predict["shot_id"]
submisdata=pd.DataFrame({
    "shot_id":predict["shot_id"],
    "shot_made_flag":tahminler
})


# In[ ]:


submisdata.to_csv("submis.csv",index=False)


# In[ ]:


import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=10000)
eval_set  = [(x_train,y_train), (x_test,y_test)]

clf.fit(x_train, y_train, eval_set=eval_set,
        eval_metric="auc", early_stopping_rounds=30)


# In[ ]:


xgbclas=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=10000, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
xgbclas.fit(x,y)


# In[ ]:


tahminler=tpot.predict(hadi)
submisdata=pd.DataFrame({
    "shot_id":predict["shot_id"],
    "shot_made_flag":tahminler
})
submisdata.to_csv("submisxgb.csv",index=False)


# In[ ]:


data.columns


# In[ ]:


set(data["playoffs"])


# In[ ]:


data['opponent']


# In[ ]:


data["matchup"]


# In[ ]:


data[["minutes_remaining","seconds_remaining","period"]]
data["total_time_remaining"]=data["minutes_remaining"]*60+data["seconds_remaining"]+(4-data["period"])*60


# In[ ]:


data["shot_distance"]


# In[ ]:


data33=data[["playoffs","season","opponent","total_time_remaining","shot_zone_area","shot_distance","shot_zone_range","action_type","shot_made_flag"]]


# In[ ]:


data334=data33.dropna(subset=['shot_made_flag'])


# In[ ]:


x=data334.iloc[:,0:8]
y=data334.iloc[:,8:]


# In[ ]:


#opponent, shot_zone_area, shot_zone_range, action_type
from sklearn import preprocessing
leseason = preprocessing.LabelEncoder()
season=leseason.fit_transform(x["season"])
x["season"]=season

leopponent = preprocessing.LabelEncoder()
opponent=leopponent.fit_transform(x["opponent"])
x["opponent"]=opponent

leshot_zone_area = preprocessing.LabelEncoder()
shotarea=leshot_zone_area.fit_transform(x["shot_zone_area"])
x["shot_zone_area"]=shotarea

leshot_zone_range = preprocessing.LabelEncoder()
shotzone=leshot_zone_range.fit_transform(x["shot_zone_range"])
x["shot_zone_range"]=shotzone

leaction_type = preprocessing.LabelEncoder()
actiontype=leaction_type.fit_transform(x["action_type"])
x["action_type"]=actiontype


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1923)


# In[ ]:


import xgboost as xgb

xgbc = xgb.XGBClassifier()
xgbc.fit(x_train,y_train)
ypred=xgbc.predict(x_test)

import sklearn.metrics as metrik
print(metrik.confusion_matrix(ypred,y_test))
print(metrik.accuracy_score(ypred,y_test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train, y_train)
ypred=rfc.predict(x_test)
import sklearn.metrics as metrik
print(metrik.confusion_matrix(ypred,y_test))
print(metrik.accuracy_score(ypred,y_test))


# In[ ]:


parameters={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.85 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10,],
 "min_child_weight" : [ 1, 3, 5, 7,9,11 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4,0.5,0.6,0.7 ],
 "n_estimators"     : [25,50,100,150,200,500,1000,2000],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(xgb.XGBClassifier(), parameters, random_state=0)
search = clf.fit(x_train,y_train)
search.best_params_


# In[ ]:


xgbc = xgb.XGBClassifier(params=search.best_params_)
xgbc.fit(x_train,y_train)
ypred=xgbc.predict(x_test)

import sklearn.metrics as metrik
print(metrik.confusion_matrix(ypred,y_test))
print(metrik.accuracy_score(ypred,y_test))


# In[ ]:


xgbc=xgb.XGBClassifier(n_estimators=25,min_child_weight=3,max_depth=4,learning_rate=0.1,gamma=0.1
                      ,colsample_bytree=0.5
                      )
xgbc.fit(x_train,y_train)
ypred=xgbc.predict(x_test)

import sklearn.metrics as metrik
print(metrik.confusion_matrix(ypred,y_test))
print(metrik.accuracy_score(ypred,y_test))


# In[ ]:


parameters={"learning_rate"    : [0.005,0.01,0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.85 ] ,
 "max_depth"        : [ 2,3, 4, 5, 6,7,8],
 "min_child_weight" : [ 1, 3, 5, 7,9,11 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4,0.5,0.6,0.7,0.8 ],
 "n_estimators"     : [25,50,100,150,200,500,1000,2000,4000],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(xgb.XGBClassifier(), parameters, random_state=1456)
search = clf.fit(x_train,y_train)
search.best_params_


# In[ ]:


xgbc=xgb.XGBClassifier(n_estimators=150,min_child_weight=1,max_depth=2,learning_rate=0.3,gamma=0.5
                      ,colsample_bytree=0.3
                      )
xgbc.fit(x_train,y_train)
ypred=xgbc.predict(x_test)

import sklearn.metrics as metrik
print(metrik.confusion_matrix(ypred,y_test))
print(metrik.accuracy_score(ypred,y_test))


# In[ ]:


hadi=predict[["playoffs","season","opponent","period","minutes_remaining","seconds_remaining","shot_zone_area","shot_distance","shot_zone_range","action_type"]]

hadi["total_time_remaining"]=hadi["minutes_remaining"]*60+hadi["seconds_remaining"]+(4-hadi["period"])*60

season=leseason.transform(hadi["season"])
hadi["season"]=season

opponent=leopponent.transform(hadi["opponent"])
hadi["opponent"]=opponent

shotarea=leshot_zone_area.transform(hadi["shot_zone_area"])
hadi["shot_zone_area"]=shotarea

shotzone=leshot_zone_range.transform(hadi["shot_zone_range"])
hadi["shot_zone_range"]=shotzone

actiontype=leaction_type.fit_transform(hadi["action_type"])
hadi["action_type"]=actiontype

hadi=hadi[['playoffs', 'season', 'opponent', 'total_time_remaining', 'shot_zone_area', 'shot_distance', 'shot_zone_range', 'action_type']]

tahminler=xgbc.predict(hadi)
tahminler


# In[ ]:


x_train


# In[ ]:


submisdata=pd.DataFrame({
    "shot_id":predict["shot_id"],
    "shot_made_flag":tahminler
})
submisdata.to_csv("submisxgb2020.csv",index=False)


# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif
kbest=SelectKBest(f_classif, k=5)
X_train_new = kbest.fit_transform(x_train, y_train)
X_test_new=kbest.transform(x_test)
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train_new,y_train)
ypred=xgbc.predict(X_test_new)

import sklearn.metrics as metrik
print(metrik.confusion_matrix(ypred,y_test))
print(metrik.accuracy_score(ypred,y_test))


# In[ ]:


from sklearn.feature_selection import SelectKBest,mutual_info_classif
kbest=SelectKBest(mutual_info_classif, k=5)
X_train_new = kbest.fit_transform(x_train, y_train)
X_test_new=kbest.transform(x_test)
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train_new,y_train)
ypred=xgbc.predict(X_test_new)

import sklearn.metrics as metrik
print(metrik.confusion_matrix(ypred,y_test))
print(metrik.accuracy_score(ypred,y_test))


# In[ ]:


hadi=predict[["playoffs","season","opponent","period","minutes_remaining","seconds_remaining","shot_zone_area","shot_distance","shot_zone_range","action_type"]]

hadi["total_time_remaining"]=hadi["minutes_remaining"]*60+hadi["seconds_remaining"]+(4-hadi["period"])*60

season=leseason.transform(hadi["season"])
hadi["season"]=season

opponent=leopponent.transform(hadi["opponent"])
hadi["opponent"]=opponent

shotarea=leshot_zone_area.transform(hadi["shot_zone_area"])
hadi["shot_zone_area"]=shotarea

shotzone=leshot_zone_range.transform(hadi["shot_zone_range"])
hadi["shot_zone_range"]=shotzone

actiontype=leaction_type.fit_transform(hadi["action_type"])
hadi["action_type"]=actiontype

hadi=hadi[['playoffs', 'season', 'opponent', 'total_time_remaining', 'shot_zone_area', 'shot_distance', 'shot_zone_range', 'action_type']]

hadibest5=kbest.transform(hadi)

tahminler=xgbc.predict(hadibest5)
tahminler


# In[ ]:


submisdata=pd.DataFrame({
    "shot_id":predict["shot_id"],
    "shot_made_flag":tahminler
})
submisdata.to_csv("submisxgb2021.csv",index=False)


# In[ ]:


xgbr = xgb.XGBRegressor()
xgbr.fit(x_train,y_train)
ypred=xgbr.predict(x_test)

import sklearn.metrics as metrik
print(metrik.mean_absolute_error(ypred,y_test))


# In[ ]:


hadi=predict[["playoffs","season","opponent","period","minutes_remaining","seconds_remaining","shot_zone_area","shot_distance","shot_zone_range","action_type"]]

hadi["total_time_remaining"]=hadi["minutes_remaining"]*60+hadi["seconds_remaining"]+(4-hadi["period"])*60

season=leseason.transform(hadi["season"])
hadi["season"]=season

opponent=leopponent.transform(hadi["opponent"])
hadi["opponent"]=opponent

shotarea=leshot_zone_area.transform(hadi["shot_zone_area"])
hadi["shot_zone_area"]=shotarea

shotzone=leshot_zone_range.transform(hadi["shot_zone_range"])
hadi["shot_zone_range"]=shotzone

actiontype=leaction_type.fit_transform(hadi["action_type"])
hadi["action_type"]=actiontype

hadi=hadi[['playoffs', 'season', 'opponent', 'total_time_remaining', 'shot_zone_area', 'shot_distance', 'shot_zone_range', 'action_type']]

tahminler=xgbr.predict(hadi)
tahminler


# In[ ]:


submisdata=pd.DataFrame({
    "shot_id":predict["shot_id"],
    "shot_made_flag":tahminler
})
submisdata.to_csv("submisxgbreg.csv",index=False)


# In[ ]:


parameters={"learning_rate"    : [0.005,0.01,0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.85 ] ,
 "max_depth"        : [ 2,3, 4, 5, 6,7,8],
 "min_child_weight" : [ 1, 3, 5, 7,9,11 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4,0.5,0.6,0.7,0.8 ],
 "n_estimators"     : [25,50,100,150,200,500,1000,2000,4000,5000],
 "colsample_bytree" : [ 0.1,0.2,0.3, 0.4, 0.5 , 0.7 ] }

from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(xgb.XGBClassifier(), parameters, random_state=1456)
search = clf.fit(x_train,y_train)
search.best_params_


# In[ ]:


xgbr = xgb.XGBRegressor(n_estimators=1000,min_child_weight=5,max_depth=3,learning_rate=0.25,gamma=0.7,colsample_bytree=0.1)
xgbr.fit(x_train,y_train)
ypred=xgbr.predict(x_test)

import sklearn.metrics as metrik
print(metrik.mean_absolute_error(ypred,y_test))


# In[ ]:


tahminler=xgbr.predict(hadi)
tahminler
submisdata=pd.DataFrame({
    "shot_id":predict["shot_id"],
    "shot_made_flag":tahminler
})
submisdata.to_csv("submisxgbreg2.csv",index=False)


# In[ ]:


import lightgbm as lgb
lgbr=lgb.LGBMRegressor()
lgbr.fit(x_train,y_train)
ypred=lgbr.predict(x_test)

import sklearn.metrics as metrik
print(metrik.mean_absolute_error(ypred,y_test))


# In[ ]:


parameters={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.85 ] ,
 "max_depth"        : [ 2,3, 4, 5, 6, 8, 10,100],
 "metric" : ["mae","mse" ],
 "num_boost_round" : [100,200,250,400,500,800,1000,1200],
 "num_leaves"     : [20,25,30,35,40,100,200],
 "min_data_in_leaf" : [ 20,50,100, 150,200 ] }

from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(lgb.LGBMRegressor(), parameters, random_state=0)
search = clf.fit(x_train,y_train)
search.best_params_


# In[ ]:


import lightgbm as lgb
lgbr=search.best_estimator_
lgbr.fit(x_train,y_train)
ypred=lgbr.predict(x_test)

import sklearn.metrics as metrik
print(metrik.mean_absolute_error(ypred,y_test))


# In[ ]:


tahminler=lgbr.predict(hadi)
tahminler
submisdata=pd.DataFrame({
    "shot_id":predict["shot_id"],
    "shot_made_flag":tahminler
})
submisdata.to_csv("submisxgbreglight.csv",index=False)


# In[ ]:




