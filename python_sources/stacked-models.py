#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import copy

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.precision = 15
from collections import defaultdict

import time
from collections import Counter
import datetime



import warnings
warnings.filterwarnings("ignore")

import eli5
import shap

import json
import altair as alt

get_ipython().run_line_magic('matplotlib', 'inline')
from typing import List

import os
import time
import datetime
import json
import gc
from numba import jit

from tqdm import tqdm_notebook
from sklearn import metrics
from typing import Any
from itertools import product
pd.set_option('max_rows', 500)
import re
from tqdm import tqdm
from joblib import Parallel, delayed


# ## Helper functions and classes

# In[ ]:


"""
Original code by Andrew Lukyanenko https://www.kaggle.com/artgor/quick-and-dirty-regression
"""
def add_datepart(df: pd.DataFrame, field_name: str,
                 prefix: str = None, drop: bool = True, time: bool = True, date: bool = True):
    """
    Helper function that adds columns relevant to a date in the column `field_name` of `df`.
    from fastai: https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py#L55
    """
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']
    if date:
        attr.append('Date')
    if time:
        attr = attr + ['Hour', 'Minute']
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower())
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df


def ifnone(a: Any, b: Any) -> Any:
    """`a` if `a` is not None, otherwise `b`.
    from fastai: https://github.com/fastai/fastai/blob/master/fastai/core.py#L92"""
    return b if a is None else a


# In[ ]:


"""
Original code by Andrew Lukyanenko https://www.kaggle.com/artgor/quick-and-dirty-regression
"""
from sklearn.base import BaseEstimator, TransformerMixin
@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def eval_qwk_lgb(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """

    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'cappa', qwk(y_true, y_pred), True


def eval_qwk_lgb_regr(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """
    y_pred[y_pred <= 1.12232214] = 0
    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1
    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2
    y_pred[y_pred > 2.22506454] = 3

    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return 'cappa', qwk(y_true, y_pred), True


# In[ ]:


"""
Original code by Andrew Lukyanenko https://www.kaggle.com/artgor/quick-and-dirty-regression
and Bruno Aquino https://www.kaggle.com/braquino/890-features
"""
def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code

def get_data(user_sample, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    
    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
                    
            
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())
            
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter
            
        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type 
                        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments

def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals

# read data
train, test, train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
# tranform function to get the train and test set
reduce_train, reduce_test, categoricals = get_train_and_test(train, test)


# In[ ]:


"""
Original code by Andrew Lukyanenko https://www.kaggle.com/artgor/quick-and-dirty-regression
"""
def preprocess(reduce_train, reduce_test):
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        
        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
                                        2040, 4090, 4220, 4095]].sum(axis = 1)
        
        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')
        
    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]
   
    return reduce_train, reduce_test, features
# call feature engineering function
reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)


# In[ ]:


del train, test


# In[ ]:


from sklearn.model_selection import train_test_split

def ensemble_split(X, y):

    X_train, X_meta, y_train, y_meta = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_meta = X_meta.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_meta = y_meta.reset_index(drop=True)
    return X_train, X_meta, y_train, y_meta


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
def lgb_regressor(X_train,y_train,X_meta,final_test,n_splits=4):
    params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'rmse'},
                'subsample': 0.75,
                'subsample_freq': 1,
                'learning_rate': 0.01,
                'max_depth': 15,
                'feature_fraction': 0.75,
                'lambda_l1': 1,  
                'lambda_l2': 1,
                'n_estimators':2000
                }    

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  
    
    final_pred=np.zeros((len(final_test)),dtype=float)    
    
    meta_pred=np.zeros((len(X_meta)),dtype=float)    

    for train_index, val_index in kf.split(X_train, y_train):
        train_X = X_train.iloc[train_index]
        val_X = X_train.iloc[val_index]
        train_y = y_train.iloc[train_index]
        val_y = y_train.iloc[val_index]
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_eval = lgb.Dataset(val_X, val_y)
        lgb_model = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=(lgb_train, lgb_eval),
                    early_stopping_rounds=100,
                    verbose_eval = 250)
        
        final_pred+=lgb_model.predict(final_test)        
        meta_pred += lgb_model.predict(X_meta)
        
    final_pred = final_pred/n_splits  
    meta_pred = meta_pred/n_splits 
    
    return final_pred, meta_pred
        


# In[ ]:


from catboost import CatBoostRegressor

def ctb_regressor(X_train,y_train,X_meta,final_test,n_splits=4):   
   
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)   
    
    # Initialise arrays which will be filled with the predictions
    final_pred=np.zeros((len(final_test)),dtype=float)    
    final_pred=np.zeros((len(final_test)),dtype=float)   
    meta_pred=np.zeros((len(X_meta)),dtype=float)  
    
    # Do 4-fold validation to ensure similarity with the stacked model
    for train_index, val_index in kf.split(X_train, y_train):
        # Create the CatBoost regressor
        ctb = CatBoostRegressor(
                               task_type="CPU",
                               learning_rate=0.05,
                               iterations=1000,
                               od_type="Iter",
                               early_stopping_rounds=40,
                               verbose = 250,
                               loss_function='RMSE'
                              )
        
        # initialise the training and validation data
        train_X = X_train.iloc[train_index]
        val_X = X_train.iloc[val_index]
        train_y = y_train.iloc[train_index]
        val_y = y_train.iloc[val_index]
        
        #fit the model to the data
        ctb.fit(train_X, train_y, eval_set=(val_X,val_y))        
        
        # predict the test data and the data used to train the meta model
        final_pred+=ctb.predict(final_test)        
        meta_pred += ctb.predict(X_meta)
    
    # normalise the predicted data by dividing by the splits
    final_pred = final_pred/n_splits  
    meta_pred = meta_pred/n_splits         
    
    return final_pred, meta_pred


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

def rf_regressor(X_train,y_train,X_meta,final_test,n_splits=4):
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) 
    
    final_pred=np.zeros((len(final_test)),dtype=float)
    
    meta_pred=np.zeros((len(X_meta)),dtype=float) 
    
    for train_index, val_index in kf.split(X_train, y_train):    
        train_X = X_train.iloc[train_index]
        val_X = X_train.iloc[val_index]
        train_y = y_train.iloc[train_index]
        val_y = y_train.iloc[val_index]
        
        regressor = RandomForestRegressor(n_estimators = 100, max_depth = 15, random_state = 42, n_jobs = -1)    
        regressor.fit(train_X, train_y)
    
        final_pred += regressor.predict(final_test)  
        meta_pred += regressor.predict(X_meta)        
               
        
    final_pred = final_pred/n_splits
    meta_pred = meta_pred/n_splits        
    
    return final_pred, meta_pred


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
def linear_regressor(X_train,y_train,X_meta,final_test,n_splits=4):
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) 
    
    final_pred=np.zeros((len(final_test)),dtype=float)
    
    meta_pred=np.zeros((len(X_meta)),dtype=float) 
    
    for train_index, val_index in kf.split(X_train, y_train):    
        train_X = X_train.iloc[train_index]
        val_X = X_train.iloc[val_index]
        train_y = y_train.iloc[train_index]
        val_y = y_train.iloc[val_index]

        regressor = LinearRegression()

        regressor.fit(train_X, train_y)              
        
        final_pred += regressor.predict(final_test)  
        meta_pred += regressor.predict(X_meta)        
               
        
    final_pred = final_pred/n_splits
    meta_pred = meta_pred/n_splits        
    
    return final_pred, meta_pred


# In[ ]:


from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)

def reduce_dim(X, y, m_feats, test = None):
    # Train a random forest regressor on the data
    model = RandomForestRegressor(random_state = 42, n_jobs = -1)
    model.fit(X, y)
    
    # Take the top x number of features
    selector = SelectFromModel(model, prefit = True, threshold=-np.inf, max_features = m_feats)
    X = pd.DataFrame(selector.transform(X))  
    
    # if there is a test set transform the test set as well
    if test is not None:
        test = pd.DataFrame(selector.transform(test))
        return X, test
        
    return X


# In[ ]:


"""
Original code by Andrew Lukyanenko https://www.kaggle.com/artgor/quick-and-dirty-regression
"""
from functools import partial
import scipy as sp
class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


# In[ ]:


def meta_model(stacked_train, y_stacked, stacked_test, n_splits=2 ):
    params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'rmse'},
                'subsample': 0.75,
                'subsample_freq': 1,
                'learning_rate': 0.05,
                'max_depth': 8,
                'feature_fraction': 0.5,
                'lambda_l1': 1,  
                'lambda_l2': 1,
                'n_estimators':2000
                }   

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)   
    
    final_pred=np.zeros((len(stacked_test)),dtype=float)
    train_pred=np.zeros((len(stacked_train)),dtype=float)   

    for train_index, val_index in kf.split(stacked_train, y_stacked):
        train_X = stacked_train.iloc[train_index]
        val_X = stacked_train.iloc[val_index]
        train_y = y_stacked.iloc[train_index]
        val_y = y_stacked.iloc[val_index]
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_eval = lgb.Dataset(val_X, val_y)
        lgb_model = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=(lgb_train, lgb_eval),
                    early_stopping_rounds=100,
                    verbose_eval = 100)
        
        final_pred += lgb_model.predict(stacked_test)
        train_pred += lgb_model.predict(stacked_train)
    
    final_pred /= n_splits
    train_pred /= n_splits
    
    optR = OptimizedRounder()
    optR.fit(train_pred.reshape(-1,), y_stacked)
    coefficients = optR.coefficients()
    
    pre = optR.predict(train_pred.reshape(-1, ), coefficients)
    print(qwk(y_stacked, train_pred))
    
    final_pred = optR.predict(final_pred.reshape(-1, ), coefficients)
    
    return final_pred, coefficients


# In[ ]:


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scikitplot as skplt

def holdout_val(X, y):
    X = reduce_dim(X, y, 200)
    
    X_train, X_holdout, y_train, y_holdout = train_test_split( X, y, test_size=0.1, random_state=42, stratify = y) 
    
    X_train, X_stacked, y_train, y_stacked = ensemble_split(X_train, y_train)      
    
    print('dimenstion reduction complete')  
    
    stacked_train = pd.DataFrame() 
    stacked_test = pd.DataFrame()
    
    final_pred, meta_pred = lgb_regressor(X_train, y_train, X_stacked,X_holdout)
    stacked_train['lgb_regressor'] = meta_pred
    stacked_test['lgb_regressor'] = final_pred
    print('lgb done')
    
    final_pred, meta_pred = rf_regressor(X_train, y_train,X_stacked,X_holdout)
    stacked_train['rf_regressor'] = meta_pred
    stacked_test['rf_regressor'] = final_pred
    print('rf done')

    final_pred, meta_pred = ctb_regressor(X_train, y_train,X_stacked,X_holdout)
    stacked_train['ctb_regressor'] = meta_pred
    stacked_test['ctb_regressor'] = final_pred
    print('ctb done')
    
    final_pred, coefficients = meta_model(stacked_train, y_stacked, stacked_test)
    
    print('Stacked QWK', qwk(y_holdout, final_pred))
    skplt.metrics.plot_confusion_matrix(y_holdout, final_pred, figsize=(12,12))
    


# In[ ]:


from sklearn.model_selection import train_test_split

def KF_val(X, y, n_splits = 10):    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    # select the top 200 features to be used for training
    X = reduce_dim(X, y, 200)
    
    # Do 10-fold cross validation
    for train_index, val_index in kf.split(X, y):
        X_train = X.iloc[train_index]
        X_holdout = X.iloc[val_index]
        y_train = y.iloc[train_index]
        y_holdout = y.iloc[val_index]    
        
        # split the data into two parts
        X_train, X_stacked, y_train, y_stacked = ensemble_split(X_train, y_train)  
        
        print('dimenstion reduction complete')  
        
        #initialise the stacked training and testinf dataframes
        stacked_train = pd.DataFrame() 
        stacked_test = pd.DataFrame()
        
        # let all models predict on the test data and the training data for training the meta model
        final_pred, meta_pred = lgb_regressor(X_train, y_train, X_stacked,X_holdout)
        stacked_train['lgb_regressor'] = meta_pred
        stacked_test['lgb_regressor'] = final_pred
        print('lgb done')

        final_pred, meta_pred = rf_regressor(X_train, y_train,X_stacked,X_holdout)
        stacked_train['rf_regressor'] = meta_pred
        stacked_test['rf_regressor'] = final_pred
        print('rf done')

        final_pred, meta_pred = ctb_regressor(X_train, y_train,X_stacked,X_holdout)
        stacked_train['ctb_regressor'] = meta_pred
        stacked_test['ctb_regressor'] = final_pred
        print('ctb done')
        
        # Train the meta model on the stacked predictions of the base model and predict the test set
        final_pred, coefficients = meta_model(stacked_train, y_stacked, stacked_test)
        score = qwk(y_holdout, final_pred)
        scores.append(score)
        print('Stacked QWK', score)
    return scores


# In[ ]:


X_train = reduce_train.drop(['accuracy_group', 'installation_id'], axis=1)
y_train = reduce_train['accuracy_group']


# In[ ]:


#holdout_val(X_train, y_train)


# In[ ]:


#scores = KF_val(X_train, y_train)


# In[ ]:


# plt.tick_params(
#     axis='x',          
#     which='both',      
#     bottom=False,      
#     top=False,         
#     labelbottom=False)
# plt.scatter(np.zeros_like(scores), scores)
# plt.axhline(y=np.array(scores).mean(), c ='r')
# plt.title('Stacked 10-fold performance')


# In[ ]:


# np.array(scores).mean()


# In[ ]:


def competition(X_train, y_train, reduce_test):    
    
    X_train, X_test = reduce_dim(X_train, y_train, 200, test = reduce_test.drop(['installation_id', 'accuracy_group'], axis=1))    
    
    X_train, X_stacked, y_train, y_stacked = ensemble_split(X_train, y_train) 
    
    stacked_train = pd.DataFrame() 
    stacked_test = pd.DataFrame()
    
    final_pred, meta_pred = lgb_regressor(X_train, y_train, X_stacked,X_test)
    stacked_train['lgb_regressor'] = meta_pred
    stacked_test['lgb_regressor'] = final_pred
    print('lgb done')
    
    final_pred, meta_pred = rf_regressor(X_train, y_train,X_stacked,X_test)
    stacked_train['rf_regressor'] = meta_pred
    stacked_test['rf_regressor'] = final_pred
    print('rf done')

    final_pred, meta_pred = ctb_regressor(X_train, y_train,X_stacked,X_test)
    stacked_train['ctb_regressor'] = meta_pred
    stacked_test['ctb_regressor'] = final_pred
    print('ctb done')
    
    final_pred, coefficients = meta_model(stacked_train, y_stacked, stacked_test)    
    
    return final_pred


# In[ ]:


final_pred = competition(X_train, y_train, reduce_test)


# In[ ]:


sample_submission['accuracy_group'] = final_pred.astype(int)
sample_submission.to_csv('submission.csv', index=False)

