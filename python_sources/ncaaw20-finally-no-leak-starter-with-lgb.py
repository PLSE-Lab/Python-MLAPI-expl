#!/usr/bin/env python
# coding: utf-8

# # Overview
# As it has been pointed out in the discussion (e.g. https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/discussion/131028), most of the high scoring public kernels have a leak. That includes mine, apparently. The main cause of this is that they (intentionally or not) ignore the chronological order of the data. In this case you end up with having many cases where the future data are used by a model to predict the past, causing a leak.
# 
# One way to avoid such situations is apparently to use only the past data to predict the future, like using only ~2018 to predict 2019. Here I implemented such a validation strategy. The log_loss in the end should be around 0.5 if you are truely successful, as is always the case with the winners in the past competitions.

# # Import Library & Load Data

# In[ ]:


# Libraries
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
import copy
import datetime
from sklearn.utils import shuffle
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna
from optuna.visualization import plot_optimization_history
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score, log_loss, classification_report, confusion_matrix
import json
import ast
import time
from sklearn import linear_model

# keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Input, Layer, Dense, Concatenate, Reshape, Dropout, merge, Add, BatchNormalization, GaussianNoise
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.layers import Layer
from keras.callbacks import *
import tensorflow as tf
import math

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import gc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

print("Libraries imported!")


# # Model Class
# My modeling pipeline (still in development:D)

# In[ ]:


class BaseModel(object):
    """
    Base Model Class:

    train_df : train pandas dataframe
    test_df : test pandas dataframe
    target : target column name (str)
    features : list of feature names
    categoricals : list of categorical feature names
    n_splits : K in KFold (default is 3)
    cv_method : options are ... KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
    group : group feature name when GroupKFold or StratifiedGroupKFold are used
    task : options are ... regression, multiclass, or binary
    parameter_tuning : bool, only for LGB
    seed : seed (int)
    scaler : options are ... None, MinMax, Standard
    verbose : bool
    """

    def __init__(self, train_df, test_df, target, features, categoricals=[], 
                n_splits=3, cv_method="KFold", group=None, task="regression", 
                parameter_tuning=False, seed=42, scaler=None, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.cv_method = cv_method
        self.group = group
        self.task = task
        self.parameter_tuning = parameter_tuning
        self.seed = seed
        self.scaler = scaler
        self.cv = self.get_cv()
        self.verbose = verbose
        self.params = self.get_params()
        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()

    def train_model(self, train_set, val_set):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    def convert_x(self, x):
        return x

    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest
        if self.task == "multiclass":
            return log_loss(y_true, y_pred)
        elif self.task == "binary":
            return log_loss(y_true, y_pred)
#             return roc_auc_score(y_true, y_pred)
        elif self.task == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def get_cv(self):
        if self.cv_method == "KFold":
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df)
        elif self.cv_method == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target])
        elif self.cv_method == "TimeSeriesSplit":
            cv = TimeSeriesSplit(max_train_size=None, n_splits=self.n_splits)
            return cv.split(self.train_df)

    def fit(self):
        # initialize
        oof_pred = np.zeros((self.train_df.shape[0], ))
        y_vals = np.zeros((self.train_df.shape[0], ))
        y_pred = np.zeros((self.test_df.shape[0], ))
        if self.group is not None:
            if self.group in self.features:
                self.features.remove(self.group)
            if self.group in self.categoricals:
                self.categoricals.remove(self.group)
        fi = np.zeros((self.n_splits, len(self.features)))

        # scaling, if necessary
        if self.scaler is not None:
            # fill NaN
            numerical_features = [f for f in self.features if f not in self.categoricals]
            self.train_df[numerical_features] = self.train_df[numerical_features].fillna(self.train_df[numerical_features].median())
            self.test_df[numerical_features] = self.test_df[numerical_features].fillna(self.test_df[numerical_features].median())
            self.train_df[self.categoricals] = self.train_df[self.categoricals].fillna(self.train_df[self.categoricals].mode().iloc[0])
            self.test_df[self.categoricals] = self.test_df[self.categoricals].fillna(self.test_df[self.categoricals].mode().iloc[0])

            # scaling
            if self.scaler == "MinMax":
                scaler = MinMaxScaler()
            elif self.scaler == "Standard":
                scaler = StandardScaler()
            df = pd.concat([self.train_df[numerical_features], self.test_df[numerical_features]], ignore_index=True)
            scaler.fit(df[numerical_features])
            x_test = self.test_df.copy()
            x_test[numerical_features] = scaler.transform(x_test[numerical_features])
            x_test = [np.absolute(x_test[i]) for i in self.categoricals] + [x_test[numerical_features]]
        else:
            x_test = self.test_df[self.features]

        # fitting with out of fold
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            # train test split
            x_train, x_val = self.train_df.loc[train_idx, self.features], self.train_df.loc[val_idx, self.features]
            y_train, y_val = self.train_df.loc[train_idx, self.target], self.train_df.loc[val_idx, self.target]

            # fitting & get feature importance
            if self.scaler is not None:
                x_train[numerical_features] = scaler.transform(x_train[numerical_features])
                x_val[numerical_features] = scaler.transform(x_val[numerical_features])
                x_train = [np.absolute(x_train[i]) for i in self.categoricals] + [x_train[numerical_features]]
                x_val = [np.absolute(x_val[i]) for i in self.categoricals] + [x_val[numerical_features]]
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model, importance = self.train_model(train_set, val_set)
            fi[fold, :] = importance
            conv_x_val = self.convert_x(x_val)
            y_vals[val_idx] = y_val
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            x_test = self.convert_x(x_test)
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
            print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_val, oof_pred[val_idx])))
        
        # feature importance data frame
        fi_df = pd.DataFrame()
        for n in np.arange(self.n_splits):
            tmp = pd.DataFrame()
            tmp["features"] = self.features
            tmp["importance"] = fi[n, :]
            tmp["fold"] = n
            fi_df = pd.concat([fi_df, tmp], ignore_index=True)
        gfi = fi_df[["features", "importance"]].groupby(["features"]).mean().reset_index()
        fi_df = fi_df.merge(gfi, on="features", how="left", suffixes=('', '_mean'))

        # outputs
        loss_score = self.calc_metric(y_vals, oof_pred)
        if self.verbose:
            print('Our oof loss score is: ', loss_score)
        return y_pred, loss_score, model, oof_pred, y_vals, fi_df

    def plot_feature_importance(self, rank_range=[1, 50]):
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 20))
        sorted_df = self.fi_df.sort_values(by = "importance_mean", ascending=False).reset_index().iloc[self.n_splits * (rank_range[0]-1) : self.n_splits * rank_range[1]]
        sns.barplot(data=sorted_df, x ="importance", y ="features", orient='h')
        ax.set_xlabel("feature importance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return sorted_df


# In[ ]:


class LgbModel(BaseModel):
    """
    LGB wrapper

    """

    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        model = lgb.train(self.params, train_set, num_boost_round = 3000, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        fi = model.feature_importance(importance_type="gain")
        return model, fi

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set

    def get_params(self):
        # fast fit parameters
        params = {
          'num_leaves': 127,
          'objective': self.task,
          'min_data_in_leaf': 50,
          'max_depth': -1,
          'learning_rate': 0.005,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "verbosity": -1,
          'random_state': 42,
         }

        # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        if self.task == "regression":
            params["metric"] = "rmse"
        elif self.task == "binary":
            params["metric"] = "binary_logloss" # auc
        elif self.task == "multiclass":
            params["metric"] = "multi_logloss" # cross_entropy, auc_mu
        
        # Bayesian Optimization by Optuna
        if self.parameter_tuning == True:
            # define objective function
            def objective(trial):
                # train, test split
                train_x, test_x, train_y, test_y = train_test_split(self.train_df[self.features], 
                                                                    self.train_df[self.target],
                                                                    test_size=0.3, random_state=self.seed)
                dtrain = lgb.Dataset(train_x, train_y, categorical_feature=self.categoricals)
                dtest = lgb.Dataset(test_x, test_y, categorical_feature=self.categoricals)

                # parameters to be explored
                hyperparams = {'num_leaves': trial.suggest_int('num_leaves', 24, 1024),
                        'boosting_type': 'gbdt',
                        'objective': params["objective"],
                        'metric': params["metric"],
                        'max_depth': trial.suggest_int('max_depth', 4, 16),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                        'early_stopping_rounds': 100
                        }

                # LGB
                model = lgb.train(hyperparams, dtrain, valid_sets=dtest, verbose_eval=500)
                pred = model.predict(test_x)
                if (self.task == "binary") | (self.task == "multiclass"):
                    return log_loss(test_y, pred)
                elif self.task == "regression":
                    return np.sqrt(mean_squared_error(test_y, pred))

            # run optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)

            print('Number of finished trials: {}'.format(len(study.trials)))
            print('Best trial:')
            trial = study.best_trial
            print('  Value: {}'.format(trial.value))
            print('  Params: ')
            for key, value in trial.params.items():
                print('    {}: {}'.format(key, value))

            params = trial.params

            # lower learning rate for better accuracy
            params["learning_rate"] = 0.001

            # plot history
            plot_optimization_history(study)

        return params


# # Load data

# In[ ]:


data_dict = {}
for i in glob.glob('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/*'):
    name = i.split('/')[-1].split('.')[0]
    if name != 'WTeamSpellings':
        data_dict[name] = pd.read_csv(i)
    else:
        data_dict[name] = pd.read_csv(i, encoding='cp1252')


# # Data Overview

# NCAAW20 has less data than NCAAM20. It may be easier for us to start with NCAAW20.

# In[ ]:


data_dict.keys()


# In[ ]:


fname = 'Cities'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WTeamSpellings'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WSeasons'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WTeams'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WNCAATourneyCompactResults'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WGameCities'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'Conferences'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WNCAATourneySeeds'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


# get int from seed
data_dict['WNCAATourneySeeds']['Seed'] = data_dict['WNCAATourneySeeds']['Seed'].apply(lambda x: int(x[1:3]))
data_dict[fname].head()


# In[ ]:


fname = 'WNCAATourneySlots'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WTeamConferences'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WNCAATourneyDetailedResults'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WRegularSeasonDetailedResults'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


fname = 'WRegularSeasonCompactResults'
print(data_dict[fname].shape)
data_dict[fname].head()


# In[ ]:


# let's also have a look at test
test = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')
print(test.shape)
test.head()


# In[ ]:


# format ID
test = test.drop(['Pred'], axis=1)
test['Season'] = test['ID'].apply(lambda x: int(x.split('_')[0]))
test['WTeamID'] = test['ID'].apply(lambda x: int(x.split('_')[1]))
test['LTeamID'] = test['ID'].apply(lambda x: int(x.split('_')[2]))
test.head()


# # Data processing and feature engineering.
# 
# The main idea is to extract features, which could be useful to understand how much one team is better than another one.

# In[ ]:


# merge tables ============
train = data_dict['WNCAATourneyCompactResults'] # use compact data only for now

# # compact <- detailed (Tourney files)
# train = pd.merge(data_dict['MNCAATourneyCompactResults'], data_dict['MNCAATourneyDetailedResults'], how='left',
#              on=['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT'])
print(train.shape)
train.head()


# In[ ]:


# Train =================================
# merge with Game Cities
gameCities = pd.merge(data_dict['WGameCities'], data_dict['Cities'], how='left', on=['CityID'])
cols_to_use = gameCities.columns.difference(train.columns).tolist() + ["Season", "WTeamID", "LTeamID"]
train = train.merge(gameCities[cols_to_use], how="left", on=["Season", "WTeamID", "LTeamID"])
train.head()

# merge with WSeasons
cols_to_use = data_dict["WSeasons"].columns.difference(train.columns).tolist() + ["Season"]
train = train.merge(data_dict["WSeasons"][cols_to_use], how="left", on=["Season"])
train.head()

# merge with WTeams
cols_to_use = data_dict["WTeams"].columns.difference(train.columns).tolist()
train = train.merge(data_dict["WTeams"][cols_to_use], how="left", left_on=["WTeamID"], right_on=["TeamID"])
train.drop(['TeamID'], axis=1, inplace=True)
train = train.merge(data_dict["WTeams"][cols_to_use], how="left", left_on=["LTeamID"], right_on=["TeamID"], suffixes=('_W', '_L'))
train.drop(['TeamID'], axis=1, inplace=True)
print(train.shape)
train.head()


# In[ ]:


# merge with WNCAATourneySeeds
cols_to_use = data_dict['WNCAATourneySeeds'].columns.difference(train.columns).tolist() + ['Season']
train = train.merge(data_dict['WNCAATourneySeeds'][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),
                    how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
train.drop(['TeamID'], axis=1, inplace=True)
train = train.merge(data_dict['WNCAATourneySeeds'][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),
                    how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))
train.drop(['TeamID'], axis=1, inplace=True)

print(train.shape)
train.head()


# In[ ]:


# test =================================
# merge with Game Cities
cols_to_use = gameCities.columns.difference(test.columns).tolist() + ["Season", "WTeamID", "LTeamID"]
test = test.merge(gameCities[cols_to_use].drop_duplicates(subset=["Season", "WTeamID", "LTeamID"]),
                  how="left", on=["Season", "WTeamID", "LTeamID"])
del gameCities
gc.collect()
test.head()

# merge with WSeasons
cols_to_use = data_dict["WSeasons"].columns.difference(test.columns).tolist() + ["Season"]
test = test.merge(data_dict["WSeasons"][cols_to_use].drop_duplicates(subset=["Season"]),
                  how="left", on=["Season"])
test.head()

# merge with WTeams
cols_to_use = data_dict["WTeams"].columns.difference(test.columns).tolist()
test = test.merge(data_dict["WTeams"][cols_to_use].drop_duplicates(subset=["TeamID"]),
                  how="left", left_on=["WTeamID"], right_on=["TeamID"])
test.drop(['TeamID'], axis=1, inplace=True)
test = test.merge(data_dict["WTeams"][cols_to_use].drop_duplicates(subset=["TeamID"]),
                  how="left", left_on=["LTeamID"], right_on=["TeamID"], suffixes=('_W', '_L'))
test.drop(['TeamID'], axis=1, inplace=True)
test.head()

# merge with WNCAATourneySeeds
cols_to_use = data_dict['WNCAATourneySeeds'].columns.difference(test.columns).tolist() + ['Season']
test = test.merge(data_dict['WNCAATourneySeeds'][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),
                  how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
test.drop(['TeamID'], axis=1, inplace=True)
test = test.merge(data_dict['WNCAATourneySeeds'][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),
                  how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))
test.drop(['TeamID'], axis=1, inplace=True)

print(test.shape)
test.head()


# In[ ]:


not_exist_in_test = [c for c in train.columns.values.tolist() if c not in test.columns.values.tolist()]
print(not_exist_in_test)
train = train.drop(not_exist_in_test, axis=1)
train.head()


# ## Sort by time (just to make sure the ordering...)
# To ensure no leak, we convert "DayZero" column to datetime object such that we explicitly have an order in our data.

# In[ ]:


# to datetime format
train["DayZero"] = pd.to_datetime(train["DayZero"], infer_datetime_format=True)

# sort by date
train = train.sort_values(by=["DayZero", "Season", "DayNum"]).reset_index(drop=True)


# ## CAUTION: dealing with 'WRegularSeasonCompactResults'
# This part is very likely to be the cause of the leak. You cannot merge aggregated data containing future results with your past data.

# In[ ]:


# compact <- detailed (regular season files)
regularSeason = data_dict['WRegularSeasonCompactResults']
# regularSeason = pd.merge(data_dict['WRegularSeasonCompactResults'], data_dict['WRegularSeasonDetailedResults'], how='left',
#              on=['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT'])
print(regularSeason.shape)
regularSeason.head()


# In[ ]:


# merge with regularSeason using only the previous data
def merge_regularSeason(df, regularSeason):
    df_new = pd.DataFrame()
    for i, season in enumerate(df["Season"].unique()):
        print(season)
        if season <= 1998:
            continue
            
        # split winners and losers (make sure not to use the future data!)
        team_win_score = regularSeason.loc[regularSeason["Season"] <= season, :].groupby(['WTeamID']).agg({'WScore':['sum', 'count', 'var']}).reset_index()
        team_win_score.columns = [' '.join(col).strip() for col in team_win_score.columns.values]
        team_loss_score = regularSeason.loc[regularSeason["Season"] <= season, :].groupby(['LTeamID']).agg({'LScore':['sum', 'count', 'var']}).reset_index()
        team_loss_score.columns = [' '.join(col).strip() for col in team_loss_score.columns.values]
        
        # merge with train 
        team_win_score["Season"] = season
        team_loss_score["Season"] = season
        df_fold = df.loc[df["Season"] == season, :].reset_index(drop=True)
        df_fold = df_fold.merge(team_win_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'])
        df_fold = df_fold.merge(team_loss_score, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'LTeamID'])
        df_fold = df_fold.merge(team_loss_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])
        df_fold = df_fold.merge(team_win_score, how='left', left_on=['Season', 'LTeamID_x'], right_on=['Season', 'WTeamID'])
        df_fold.drop(['LTeamID_y', 'WTeamID_y'], axis=1, inplace=True)
        
        # restore
        df_new = pd.concat([df_new, df_fold], ignore_index=True)
    return df_new

train = merge_regularSeason(train, regularSeason)
test = merge_regularSeason(test, regularSeason)


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# In[ ]:


# preprocess
def preprocess(df):
    df['x_score'] = df['WScore sum_x'] + df['LScore sum_y']
    df['y_score'] = df['WScore sum_y'] + df['LScore sum_x']
    df['x_count'] = df['WScore count_x'] + df['LScore count_y']
    df['y_count'] = df['WScore count_y'] + df['WScore count_x']
    df['x_var'] = df['WScore var_x'] + df['LScore var_x']
    df['y_var'] = df['WScore var_y'] + df['LScore var_y']
    return df
train = preprocess(train)
test = preprocess(test)


# In[ ]:


# make winner and loser train
train_win = train.copy()
train_los = train.copy()
train_win = train_win[['DayNum', 'DayZero', 'Season', 'Seed_W', 'Seed_L', 'x_score', 'y_score', 'x_count', 'y_count', 'x_var', 'y_var']]
train_los = train_los[['DayNum', 'DayZero', 'Season', 'Seed_L', 'Seed_W', 'y_score', 'x_score', 'x_count', 'y_count', 'x_var', 'y_var']]
train_win.columns = ['DayNum', 'DayZero', 'Season', 'Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']
train_los.columns = ['DayNum', 'DayZero', 'Season', 'Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']

# same processing for test
test = test[['ID', 'Season', 'Seed_W', 'Seed_L', 'x_score', 'y_score', 'x_count', 'y_count', 'x_var', 'y_var']]
test.columns = ['ID', 'Season', 'Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']


# In[ ]:


# feature enginnering
def feature_engineering(df):
    df['Seed_diff'] = df['Seed_1'] - df['Seed_2']
    df['Seed_ratio'] = df['Seed_1'] / df['Seed_2']
    df['Score_ratio'] = df['Score_1'] / df['Score_2']
    df['Count_ratio'] = df['Count_1'] / df['Count_2']
    df['Var_ratio'] = df['Var_1'] / df['Var_2']
    df['Mean_score1'] = df['Score_1'] / df['Count_1']
    df['Mean_score2'] = df['Score_2'] / df['Count_2']
    df['Mean_score_diff'] = df['Mean_score1'] - df['Mean_score2']
    df['Mean_score_ratio'] = df['Mean_score1'] / df['Mean_score2']
    df = df.drop(['Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2'], axis=1)
    return df
train_win = feature_engineering(train_win)
train_los = feature_engineering(train_los)
test = feature_engineering(test)


# In[ ]:


train_win["result"] = 1
print(train_win.shape)
train_win.head()


# In[ ]:


train_los["result"] = 0
print(train_los.shape)
train_los.head()


# In[ ]:


data = pd.concat([train_win, train_los], ignore_index=True)
data = data.sort_values(by=['DayZero', 'Season', 'DayNum']).reset_index(drop=True)
print(data.shape)
data.head()


# # Predict & Make Submission File

# ## Model training
# We use TimeSeriesSplit to ensure no leakage due to flipping time, like training using ~2018 to predict 2019 and so on.[](http://)

# In[ ]:


target = 'result'
features = data.columns.values.tolist()
dropcols = [target,'DayNum', 'Season', 'DayZero']
features = [f for f in features if f not in dropcols]
print(features)


# ## Fit LGB

# In[ ]:


test["Season"].unique()


# In[ ]:


# predict 2015 by using ~2014, for example
models = {}
test["Pred"] = 0.5
pred = np.zeros(test.shape[0])
for season in test["Season"].unique():
    print(f"Predicting {season}...")
    lgbm = LgbModel(data.loc[data["Season"] < season, :], test.loc[test["Season"] == season, :], target, features, categoricals=[], n_splits=10, 
                    cv_method="TimeSeriesSplit", group=None, task="binary", scaler=None, verbose=True)
    models[season] = lgbm
    test.loc[test["Season"] == season, "Pred"] = lgbm.y_pred


# In[ ]:


# feature importance
fi_df = models[2019].plot_feature_importance()


# ## Submission

# In[ ]:


submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')
submission_df['Pred'] = test['Pred']
submission_df


# In[ ]:


submission_df['Pred'].hist()


# In[ ]:


submission_df.to_csv('submission.csv', index=False)

