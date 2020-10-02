#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:46:07 2018

@author: Jacob
"""

#Importing libraries
import pandas as pd
import os
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import skew
import matplotlib
import matplotlib.pyplot as plt
import warnings
from scipy.special import boxcox1p
warnings.filterwarnings("ignore")

#Getting to directory
#os.chdir('/Users/Jacob/Desktop/Kaggle Competitions/PUBG')

#######################################
#Data Import and Cleaning
#######################################
#Importing data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Memory Optimization
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

#Running optimization
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

#We can get rid of the Id column for both
train_id = train['Id']
train_match = train['matchId']
train_group = train['groupId']
test_id = test['Id']
test_match = test['matchId']
test_group = test['groupId']

#Lets take a look data types
#print(train.dtypes)
#Looks like it is primarily going to be numeric variables

#Lets take a look at our response
#sns.distplot(train['winPlacePerc']);
#Seems like it is a uniform distribution
#A logistic model, or classifier with probabilities would probably be best

#Lets take a look at correlation of our predictors
#%matplotlib qt
#plt.matshow(train.corr())
#plt.xticks(range(len(train.columns)), train.columns)
#plt.yticks(range(len(train.columns)), train.columns)
#plt.colorbar()
#plt.show()

#Lets take a look at correlations only with response
#resp_corr = train.corr().iloc[:,-1]
#resp_corr.reindex(resp_corr.abs().sort_values(ascending = False).index)

#Lets concat our datasets
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.winPlacePerc.values
train['train'] = 1
test['train'] = 0
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['winPlacePerc'], axis=1, inplace=True)

#Lets take a look at nonresponse
#def get_percentage_missing(series):
#    """ Calculates percentage of NaN values in DataFrame
#    :param series: Pandas DataFrame object
#    :return: float
#    """
#    num = series.isnull().sum()
#    den = len(series)
#    percent = round(num/den,4)*100
#    return percent.sort_values(ascending = False)
#print(get_percentage_missing(all_data))
#No missing data, awesome

###############################################################################
#Feature Engineering
###############################################################################

'''
Feature engineering ideas

Match averages
headshot percentage
Cheater on team

'''

#Lets create a variable that denotes team size
team_sizes = all_data.groupby(['matchId','groupId'])['Id'].agg(['count'])
team_sizes.columns = ['teamSize']
all_data = all_data.merge(team_sizes, 
                          left_on = ['matchId','groupId'], 
                          right_index = True, 
                          how = 'left')
all_data['solo'] = np.where(all_data['teamSize']==1, 1, 0)

#Lets create a variable that compares much better the player is than the average KillPoints
avg_kp = all_data.groupby(['matchId'])['killPoints','winPoints'].agg(['mean'])
avg_kp.columns = ['matchAvgKills','matchAvgWins']
all_data = all_data.merge(avg_kp, 
                          left_on = ['matchId'], 
                          right_index = True, 
                          how = 'left')
all_data['killDiff'] = all_data['killPoints'] - all_data['matchAvgKills']
all_data['winDiff'] = all_data['winPoints'] - all_data['matchAvgWins']

#Lets create a variable that measures the group's average external stats
group_avg_kp = all_data.groupby(['matchId','groupId'])['killPoints','winPoints'].agg(['mean'])
group_avg_kp.columns = ['groupAvgKills','groupAvgWins']
all_data = all_data.merge(group_avg_kp, 
                          left_on = ['matchId','groupId'], 
                          right_index = True, 
                          how = 'left')

#Lets create an indicator for when walkDistance is 0
all_data['noWalk'] = np.where(all_data['walkDistance']==0,1,0)

#Lets do the same thing for swimming and driving
all_data['noSwim'] = np.where(all_data['swimDistance']==0,1,0)
all_data['noRide'] = np.where(all_data['rideDistance']==0,1,0)

#Lets create some more variables from a kernel
all_data["distance"] = all_data["rideDistance"]+all_data["walkDistance"]+all_data["swimDistance"]
all_data["healthpack"] = all_data["boosts"] + all_data["heals"]
all_data["skill"] = all_data["headshotKills"]+all_data["roadKills"]

#Lets drop some columns from our training set that are probably not helpful
all_data = all_data.drop(['Id','matchId','groupId'], axis = 1)

#Turning all_data into train and test
X_train = all_data[all_data['train']==1].drop(['train'], axis = 1)
X_test = all_data[all_data['train']==0].drop(['train'], axis = 1)

###############################################################################
#Modeling!
###############################################################################

from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from keras import backend as K
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Dropout, BatchNormalization
K.tensorflow_backend._get_available_gpus()

#Lets separate into train and validation sets
#X_train, X_valid, y_train, y_valid = train_test_split(train, y_train, test_size=0.1, random_state=123)

#Ridge
ridge = Ridge()

#Keras
def create_model(optimizer='RMSprop',
                 kernel_initializer='glorot_uniform', 
                 dropout=0.25):
    model = Sequential()
    model.add(Dense(512,activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(256,activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(64,activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(loss='mae',optimizer=optimizer, metrics = ['mae'])
    return model
keras = Pipeline([
        ('standard',StandardScaler()),
        ('keras', KerasRegressor(build_fn=create_model, 
                                 verbose = 1, 
                                 epochs = 5,
                                 validation_split = 0.1,
                                 batch_size = 1024))])
#Validation
#keras.fit(X = X_train, 
#          y = y_train)
#mean_absolute_error(y_valid, keras.predict(X_valid))

#XGBoost
xgb = XGBRegressor(n_estimators = 1000,
                   tree_method = 'hist',
                   eval_metric = 'mae',
                   verbose = 1,
                   random_state = 123, 
                   n_jobs = 4,
                   colsample_bytree = .8)
                   
#LightGBM
lgb = LGBMRegressor(n_estimators = 1000,
                    learning_rate=0.05,
                    metric = 'mae', 
                    objective = 'regression',
                    random_state = 123, 
                    n_jobs = 4,
                    colsample_bytree = .8,
                    importance_type = 'gain',
                    verbose = 1)

#CatBoost
cat = CatBoostRegressor(iterations = 1000,
                        learning_rate=0.05,
                        depth=12,
                        eval_metric='MAE',
                        random_seed = 123,
                        metric_period = 50,
                        thread_count = 4,
                        colsample_bylevel = 0.8,
                        verbose = True)

#Creating our stacked model
metalgb = LGBMRegressor(n_estimators = 1000,
                    learning_rate=0.05,
                    metric = 'mae', 
                    objective = 'regression',
                    random_state = 123, 
                    n_jobs = 4,
                    colsample_bytree = .8,
                    importance_type = 'gain')
                    
stacked = StackingRegressor(regressors=[ridge, keras, xgb, lgb, cat], 
                           meta_regressor=metalgb,
                           use_features_in_secondary = True)
stacked.fit(X_train.values, y_train)

#Submission of Scores
#del sub
sub = pd.read_csv('../input/sample_submission.csv')
sub['Id'] = test_id
sub['matchId']=test_match
sub['groupId']=test_group
sub['winPlacePerc'] = np.clip(stacked.predict(X_test),0,1)
sub.columns = ['Id','pred','matchId','groupId']
subg = sub.groupby(['matchId','groupId'])['pred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
subg.columns = ['matchId','groupId','winPlacePerc']
sub = sub.merge(subg, how='left', on=['matchId','groupId'])
sub = sub[['Id','matchId','groupId','winPlacePerc']]
sub['winPlacePerc'] = sub.groupby('matchId')['winPlacePerc'].transform(lambda x: minmax_scale(x.values.astype(float)))
sub = sub[['Id','winPlacePerc']]
sub.to_csv('submission.csv', index=False)