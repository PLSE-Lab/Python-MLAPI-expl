#!/usr/bin/env python
# coding: utf-8

# 
# ### Explanation
# 
# This kernel used LGBM and treated it as a regression problem. I only did a little feature engineering so far.(just transform some date format features into numeric)
# 
# The ideas is that:
# - if we treated it as a regression problem, it's better to do some smooth operation. See the [kernel](https://www.kaggle.com/hukuda222/nfl-simple-evluation-trick).
# - I used the distribution in [kernel](https://www.kaggle.com/jpmiller/simple-distribution) as my smooth distribution.
# - We can see the simple distribution in [kernel](https://www.kaggle.com/jpmiller/simple-distribution) get the 1436 LB. If we use LGBM to do regression prediction and shift the distribution based on the yards we predicte, we should get a better LB. 

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


###raw mae
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error
import sqlite3
import xgboost as xgb
import datetime
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import gc
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization
from kaggle.competitions import nflrush
env = nflrush.make_env()


# In[ ]:


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv',low_memory=False)


# In[ ]:


train.loc[train.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"
train.loc[train.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"

train.loc[train.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr'] = "BLT"
train.loc[train.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"

train.loc[train.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"
train.loc[train.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"

train.loc[train.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"
train.loc[train.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"


# In[ ]:


train['is_run'] = train.NflId == train.NflIdRusher
train_single = train[train.is_run==True]


# In[ ]:


def transform_time_quarter(str1):
    return int(str1[:2])*60 + int(str1[3:5])
def transform_time_all(str1,quarter):
    if quarter<=4:
        return 15*60 - (int(str1[:2])*60 + int(str1[3:5])) + (quarter-1)*15*60
    if quarter ==5:
        return 10*60 - (int(str1[:2])*60 + int(str1[3:5])) + (quarter-1)*15*60
train_single['time_quarter'] = train_single.GameClock.map(lambda x:transform_time_quarter(x))
train_single['time_end'] = train_single.apply(lambda x:transform_time_all(x.loc['GameClock'],x.loc['Quarter']),axis=1)


# In[ ]:


train_single['TimeHandoff'] = train_single['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train_single['TimeSnap'] = train_single['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train_single['handoff_snap_diff'] = (train_single['TimeHandoff'] - train_single['TimeSnap']).map(lambda x:x.seconds)
###drop timehandoff
### drop timesnap


# In[ ]:


remove_features = ['GameId','PlayId','DisplayName','GameClock','TimeHandoff','TimeSnap']
train_single['date_game'] = train_single.GameId.map(lambda x:pd.to_datetime(str(x)[:8]))
train_single['runner_age'] = (train_single.date_game.map(pd.to_datetime) - train_single.PlayerBirthDate.map(pd.to_datetime)).map(lambda x:x.days)/365
remove_features.append('HomeTeamAbbr')
remove_features.append('VisitorTeamAbbr')
remove_features.append('PlayerBirthDate')
remove_features.append('is_run')
def transform_height(te):
    return (int(te.split('-')[0])*12 + int(te.split('-')[1]))*2.54/100
train_single['runner_height'] = train_single.PlayerHeight.map(transform_height)
remove_features.append('PossessionTeam')
remove_features.append('FieldPosition')
remove_features.append('PlayerHeight')
remove_features.append('NflIdRusher')
remove_features.append('date_game')
train_single['own_field'] = (train_single['FieldPosition'] == train_single['PossessionTeam']).astype(int)
dist_to_end_train = train_single.apply(lambda x:(100 - x.loc['YardLine']) if x.loc['own_field']==1 else x.loc['YardLine'],axis=1)
remove_features.append('own_field')
train_single.drop(remove_features,axis=1,inplace=True)


# In[ ]:


train_single.fillna(-999,inplace=True)


# In[ ]:


y_train = train_single.Yards
X_train = train_single.drop(['Yards'],axis=1)
for f in X_train.columns:
    if X_train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f])+[-999])
        X_train[f] = lbl.transform(list(X_train[f]))


# In[ ]:


def get_cdf_df(yards_array):
    pdf, edges = np.histogram(yards_array, bins=199,
                 range=(-99,100), density=True)
    cdf = pdf.cumsum().clip(0, 1)
    cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T, 
                            columns=['Yards'+str(i) for i in range(-99,100)])
    return cdf_df
cdf = get_cdf_df(y_train).values.reshape(-1,)

def get_score(y_pred,cdf,w,dist_to_end):
    y_pred = int(y_pred)
    if y_pred ==w:
        y_pred_array = cdf.copy()
    elif y_pred - w >0:
        y_pred_array = np.zeros(199)
        y_pred_array[(y_pred-w):] = cdf[:(-(y_pred-w))].copy()
    elif w - y_pred >0:
        y_pred_array = np.ones(199)
        y_pred_array[:(y_pred-w)] = cdf[(w-y_pred):].copy()
    y_pred_array[-1]=1
    y_pred_array[(dist_to_end+99):]=1
    return y_pred_array    

def get_score_pingyi1(y_pred,y_true,cdf,w,dist_to_end):
    y_pred = int(y_pred)
    if y_pred ==w:
        y_pred_array = cdf.copy()
    elif y_pred - w >0:
        y_pred_array = np.zeros(199)
        y_pred_array[(y_pred-w):] = cdf[:(-(y_pred-w))].copy()
    elif w - y_pred >0:
        y_pred_array = np.ones(199)
        y_pred_array[:(y_pred-w)] = cdf[(w-y_pred):].copy()
    y_pred_array[-1]=1
    y_pred_array[(dist_to_end+99):]=1
    y_true_array = np.zeros(199)
    y_true_array[(y_true+99):]=1
    return np.mean((y_pred_array - y_true_array)**2)


def CRPS_pingyi1(y_preds,y_trues,w,cdf,dist_to_ends):
    if len(y_preds) != len(y_trues):
        print('length does not match')
        return None
    n = len(y_preds)
    tmp = []
    for a,b,c in zip(y_preds, y_trues,dist_to_ends):
        tmp.append(get_score_pingyi1(a,b,cdf,w,c))
    return np.mean(tmp)


# In[ ]:


kf=KFold(n_splits = 5)
resu1 = 0
impor1 = 0
resu2_cprs = 0
resu3_mae=0
##y_pred = 0
stack_train = np.zeros([X_train.shape[0],])
models = []
for train_index, test_index in kf.split(X_train, y_train):
    X_train2= X_train.iloc[train_index,:]
    y_train2= y_train.iloc[train_index]
    X_test2= X_train.iloc[test_index,:]
    y_test2= y_train.iloc[test_index]
#     clf = lgb.LGBMRegressor(n_estimators=10000, random_state=47,subsample=0.7,
#                              colsample_bytree=0.7,learning_rate=0.005,importance_type = 'gain',
#                      max_depth = -1, num_leaves = 100,min_child_samples=20,min_split_gain = 0.001,
#                        bagging_freq=1,reg_alpha = 0,reg_lambda = 0,n_jobs = -1)
    clf = lgb.LGBMRegressor(n_estimators=10000, random_state=47,learning_rate=0.005,importance_type = 'gain',
                     n_jobs = -1,metric='mae')
    clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)],early_stopping_rounds=200,verbose=50)
    models.append(clf)
    temp_predict = clf.predict(X_test2)
    stack_train[test_index] = temp_predict
    ##y_pred += clf.predict(X_test)/5
    mse = mean_squared_error(y_test2, temp_predict)
    crps = CRPS_pingyi1(temp_predict,y_test2,4,cdf,dist_to_end_train.iloc[test_index])
    mae = mean_absolute_error(y_test2, temp_predict)
    print(crps)
    
    resu1 += mse/5
    resu2_cprs += crps/5
    resu3_mae += mae/5 
    impor1 += clf.feature_importances_/5
    gc.collect()
print('mean mse:',resu1)
print('oof mse:',mean_squared_error(y_train,stack_train))
print('mean mae:',resu3_mae)
print('oof mae:',mean_absolute_error(y_train,stack_train))
print('mean cprs:',resu2_cprs)
print('oof cprs:',CRPS_pingyi1(stack_train,y_train,4,cdf,dist_to_end_train))


# In[ ]:


def transform_test(test):
    test.loc[test.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"
    test.loc[test.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"

    test.loc[test.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr'] = "BLT"
    test.loc[test.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"

    test.loc[test.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"
    test.loc[test.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"

    test.loc[test.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"
    test.loc[test.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"
    test['is_run'] = test.NflId == test.NflIdRusher
    test_single = test[test.is_run==True]
    test_single['time_quarter'] = test_single.GameClock.map(lambda x:transform_time_quarter(x))
    test_single['time_end'] = test_single.apply(lambda x:transform_time_all(x.loc['GameClock'],x.loc['Quarter']),axis=1)
    test_single['TimeHandoff'] = test_single['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    test_single['TimeSnap'] = test_single['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    test_single['handoff_snap_diff'] = (test_single['TimeHandoff'] - test_single['TimeSnap']).map(lambda x:x.seconds)
    test_single['date_game'] = test_single.GameId.map(lambda x:pd.to_datetime(str(x)[:8]))
    test_single['runner_age'] = (test_single.date_game.map(pd.to_datetime) - test_single.PlayerBirthDate.map(pd.to_datetime)).map(lambda x:x.days)/365
    test_single['runner_height'] = test_single.PlayerHeight.map(transform_height)
    return test_single.drop(remove_features,axis=1)


# In[ ]:


for (test_df, sample_prediction_df) in env.iter_test():
    test_df['own_field'] = (test_df['FieldPosition'] == test_df['PossessionTeam']).astype(int)
    dist_to_end_test = test_df.apply(lambda x:(100 - x.loc['YardLine']) if x.loc['own_field']==1 else x.loc['YardLine'],axis=1)
    X_test = transform_test(test_df)
    X_test.fillna(-999,inplace=True)
    for f in X_test.columns:
        if X_test[f].dtype=='object':
            X_test[f] = X_test[f].map(lambda x:x if x in set(X_train[f]) else -999)
    for f in X_test.columns:
        if X_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X_train[f])+[-999])
            X_test[f] = lbl.transform(list(X_test[f])) 
    pred_value = 0
    for model in models:
        pred_value += model.predict(X_test)[0]/5
    pred_data = list(get_score(pred_value,cdf,4,dist_to_end_test.values[0]))
    pred_data = np.array(pred_data).reshape(1,199)
    pred_target = pd.DataFrame(index = sample_prediction_df.index,                                columns = sample_prediction_df.columns,                                #data = np.array(pred_data))
                               data = pred_data)
    #print(pred_target)
    env.predict(pred_target)
env.write_submission_file()
    


# In[ ]:


# kf=KFold(n_splits = 5)
# resu1 = 0
# impor1 = 0
# ##y_pred = 0
# stack_train = np.zeros([X_train.shape[0],])
# for train_index, test_index in kf.split(X_train, y_train):
#     X_train2= X_train.iloc[train_index,:]
#     y_train2= y_train.iloc[train_index]
#     X_test2= X_train.iloc[test_index,:]
#     y_test2= y_train.iloc[test_index]
#     clf = lgb.LGBMRegressor(n_estimators=10000, random_state=47,subsample=0.7,
#                              colsample_bytree=0.7,learning_rate=0.03,importance_type = 'gain',
#                      max_depth = -1, num_leaves = 256,min_child_samples=20,min_split_gain = 0.001,
#                        bagging_freq=1,reg_alpha = 0,reg_lambda = 0,n_jobs = -1)
#     clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)],early_stopping_rounds=100,verbose=50)
#     temp_predict = clf.predict(X_test2)
#     stack_train[test_index] = temp_predict
#     ##y_pred += clf.predict(X_test)/5
#     mse = mean_squared_error(y_test2, temp_predict)
#     print(mse)
#     resu1 += mse/5
#     impor1 += clf.feature_importances_/5
#     gc.collect()

