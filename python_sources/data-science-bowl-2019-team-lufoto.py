#!/usr/bin/env python
# coding: utf-8

# Model 1 by [Jayasooryan K V](https://www.kaggle.com/jayasoo)  
# Model 2 by [Aditya kumar](https://www.kaggle.com/negi009)    
# Model 3 [public kernal](https://www.kaggle.com/fatsaltyfish/convert-to-regression-feature-test)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm,tqdm_notebook
import gc
import random

from collections import Counter
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import normalize
from functools import partial
import scipy as sp
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils import shuffle

import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from catboost import CatBoostRegressor
from matplotlib import pyplot
import shap
from time import time
from scipy import stats
import json
import xgboost as xgb
import lightgbm as lgb


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gc
pd.set_option('max_rows',600)
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


np.random.seed(42)


# In[ ]:


train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')


# In[ ]:


train_df = train_df.loc[train_df['installation_id'].isin(set(train_labels_df['installation_id']))]


# In[ ]:


train_df['title'] = train_df['title'].str.replace(',','')
test_df['title'] = test_df['title'].str.replace(',','')


# In[ ]:


train_df['title_eventcode'] = train_df['title'] + '_' + train_df['event_code'].apply(str)
test_df['title_eventcode'] = test_df['title'] + '_' + test_df['event_code'].apply(str)


# In[ ]:


train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])


# In[ ]:


train_df.head()


# In[ ]:


len(list(set(train_df['title_eventcode'].values)))


# In[ ]:


title_eventcodes = list(set(train_df['title_eventcode'].values))
titles = train_df['title'].unique()
assessments = [title for title in titles if 'Assessment' in title]
event_codes = train_df['event_code'].unique()
games = train_df.loc[train_df['type']=='Game']['title'].unique()
activities = train_df.loc[train_df['type']=='Activity']['title'].unique()


# In[ ]:


def calculate_accuracy_group(true_attempts, false_attempts):
    accuracy = true_attempts/(true_attempts+false_attempts) if true_attempts + false_attempts > 0 else 0   
    if accuracy == 0:
        accuracy_group = 0
    elif accuracy == 1:
        accuracy_group = 3
    elif accuracy == 0.5:
        accuracy_group = 2
    else:
        accuracy_group = 1
    return accuracy_group


# In[ ]:


MAX_TIME_VAL = 9999999


# In[ ]:


def feature_extractor(user_data, test=False):
    features_list = []
    title_eventcounts = {titlecode:0 for titlecode in title_eventcodes}
    title_counts = {title:0 for title in titles}
    assessment_times = {assessment + '_pass_total_time':0 for assessment in assessments}
    assessment_times.update({assessment + '_fail_total_time':0 for assessment in assessments})
    assessment_counts = {assessment + '_pass_count':0 for assessment in assessments}
    assessment_counts.update({assessment + '_fail_count':0 for assessment in assessments})
    assessment_false_steps_count = {assessment + '_false_steps_count':0 for assessment in assessments}
    assessment_last_false_steps_count = {assessment + '_last_false_steps_count':0 for assessment in assessments}
    assessment_min_times = {assessment + '_pass_min_time':MAX_TIME_VAL for assessment in assessments}
    assessment_min_times.update({assessment + '_fail_min_time':MAX_TIME_VAL for assessment in assessments})
    assessment_max_times = {assessment + '_pass_max_time':0 for assessment in assessments}
    assessment_max_times.update({assessment + '_fail_max_time':0 for assessment in assessments})
    assessment_accuracy_group_counts = {'accuracy_group_count_'+str(i):0 for i in range(4)}
    assessment_last_accuracy = {assessment + '_last_accuracy_group': -1 for assessment in assessments}
    assessment_accuracy_group_last = -1
    assessment_correct_count = 0
    assessment_false_count = 0
    event_codes_count = {str(code): 0 for code in event_codes}
    games_counter = {game + '_played_count':0 for game in games}
    game_times = {game + '_total_time':0 for game in games}
    game_min_times = {game + '_min_time':MAX_TIME_VAL for game in games}
    game_max_times = {game + '_max_time':0 for game in games}
    activities_counter = {game + '_played_count':0 for game in activities}
    activity_times = {activity + '_total_time':0 for activity in activities}
    activity_min_times = {activity + '_min_time':MAX_TIME_VAL for activity in activities}
    activity_max_times = {activity + '_max_time':0 for activity in activities}
    first_session_is_welcome = None
    session_count = 0
    durations = []
    game_false_steps_count = {game + '_false_steps_count':0 for game in games}
    game_round_max = {game + '_round_max':0 for game in games}
    game_round_total = {game + '_round_total':0 for game in games}
    game_false_steps_avg_per_round = {game + '_false_steps_avg_per_round':0 for game in games}
    game_time_avg_per_round = {game + '_time_avg_per_round':0 for game in games}
    game_last_false_steps_count = {game + '_last_false_steps_count':0 for game in games}
    game_last_rounds = {game + '_last_rounds':0 for game in games}
    game_last_false_steps_avg_per_round = {game + '_last_false_steps_avg_per_round':0 for game in games}
    
    installation_id = user_data.iloc[0]['installation_id']
    
    for game_session, session in user_data.groupby('game_session', sort=False):
        session_title = session.iloc[0]['title']
        session_type = session.iloc[0]['type']
        session_time = session.iloc[-1]['game_time']
        session_world = session.iloc[0]['world']
        
        if first_session_is_welcome is None:
                if session_title == 'Welcome to Lost Lagoon!':
                    first_session_is_welcome = True
                else:
                    first_session_is_welcome = False

        if session.iloc[0]['type']=='Assessment':
            features = {}
            assessment_avg_times = {assessment + '_pass_avg_time':0 for assessment in assessments}
            assessment_avg_times.update({assessment + '_fail_avg_time':0 for assessment in assessments})
            features['installation_id'] = installation_id
            features['game_session'] = game_session
            features['assessment_title'] = session_title
            features['assessment_world'] = session_world
            features.update(title_eventcounts.copy())
            features.update(title_counts.copy())
            features['first_session_is_welcome'] = first_session_is_welcome        
            features['session_count_so_far'] = session_count
            
            for assessment in assessments:
                pass_count = assessment_counts[session_title+'_pass_count']
                fail_count = assessment_counts[session_title+'_fail_count']
                pass_total_time = assessment_times[session_title+'_pass_total_time']
                fail_total_time = assessment_times[session_title+'_fail_total_time']
                
                pass_avg_time = pass_total_time / pass_count if pass_count > 0 else 0
                fail_avg_time = fail_total_time / fail_count if fail_count > 0 else 0
                assessment_avg_times[session_title+'_pass_avg_time'] = pass_avg_time
                assessment_avg_times[session_title+'_fail_avg_time'] = fail_avg_time
            features.update(assessment_avg_times.copy())   
            features.update(assessment_min_times.copy())
            features.update(assessment_max_times.copy())
            features.update(assessment_accuracy_group_counts.copy())
            features['accuracy_group_last'] = assessment_accuracy_group_last
            features['assessment_true_attempts_count'] = assessment_correct_count
            features['assessment_false_attempts_count'] = assessment_false_count
            features['assessment_count_total'] = assessment_correct_count + assessment_false_count
            features['accumulated_accuracy_group'] = calculate_accuracy_group(assessment_correct_count, assessment_false_count)
            features.update(event_codes_count.copy())
            features.update(assessment_last_accuracy.copy())
            features.update(assessment_false_steps_count.copy())
            features.update(assessment_last_false_steps_count.copy())
            features.update(game_false_steps_count.copy())
            features.update(game_round_max.copy())
            features.update(game_round_total.copy())
            
            
            if len(durations) == 0:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            
            complete_code = 4100
            if session.iloc[0]['title']=='Bird Measurer (Assessment)':
                complete_code = 4110
            all_attempts = session.loc[session['event_code']==complete_code]['event_data']
            true_attempts = all_attempts.str.contains('true').sum()
            false_attempts = all_attempts.str.contains('false').sum()
            
            accuracy_group = calculate_accuracy_group(true_attempts, false_attempts)
            features['accuracy_group'] = accuracy_group
            
            all_attempts = session.loc[session['event_code']==complete_code, ['event_data', 'game_time']]
            all_attempts['time_taken'] = all_attempts['game_time'].diff()
            all_attempts.fillna(0,inplace=True)
            true_attempts = all_attempts[all_attempts['event_data'].str.contains('true')]
            false_attempts = all_attempts[all_attempts['event_data'].str.contains('false')]
            assessment_times[session_title+'_pass_total_time'] += true_attempts['time_taken'].sum()
            assessment_times[session_title+'_fail_total_time'] += false_attempts['time_taken'].sum()
            assessment_counts[session_title+'_pass_count'] += true_attempts.shape[0]
            assessment_counts[session_title+'_fail_count'] += false_attempts.shape[0]
            
            min_pass_time = true_attempts['time_taken'].min() if true_attempts.shape[0] > 0 else MAX_TIME_VAL
            max_pass_time = true_attempts['time_taken'].max() if true_attempts.shape[0] > 0 else 0
            min_fail_time = false_attempts['time_taken'].min() if false_attempts.shape[0] > 0 else MAX_TIME_VAL
            max_fail_time = false_attempts['time_taken'].max() if false_attempts.shape[0] > 0 else 0
            
            assessment_min_times[session_title+'_pass_min_time'] = min(assessment_min_times[session_title+'_pass_min_time'], min_pass_time)
            assessment_min_times[session_title+'_fail_min_time'] = min(assessment_min_times[session_title+'_fail_min_time'], min_fail_time)
            assessment_max_times[session_title+'_pass_max_time'] = max(assessment_max_times[session_title+'_pass_max_time'], max_pass_time)
            assessment_max_times[session_title+'_fail_max_time'] = max(assessment_max_times[session_title+'_fail_max_time'], max_fail_time)
        
            
            assessment_accuracy_group_counts['accuracy_group_count_'+str(accuracy_group)] += 1
            assessment_accuracy_group_last = accuracy_group
            assessment_correct_count += true_attempts.shape[0]
            assessment_false_count += false_attempts.shape[0]
            
            assessment_last_accuracy[session_title+'_last_accuracy_group'] = accuracy_group
            
            
            session_end = session.loc[session['event_code']==complete_code]
            if session_end.shape[0]>0:
                session_end_event_count = session_end.iloc[-1]['event_count']
            else:
                session_end_event_count = 9999999
            false_steps = session.loc[(session['event_count']<session_end_event_count) & (session['event_code']!=complete_code) & (session['event_data'].str.contains('"correct":false'))].shape[0]
            assessment_false_steps_count[session_title + '_false_steps_count'] += false_steps
            assessment_last_false_steps_count[session_title + '_last_false_steps_count'] = false_steps
            
            game_avg_times = {game + '_avg_time':0 for game in games}
            for game in games:
                total_time = game_times[game+'_total_time']
                game_played = games_counter[game+'_played_count']
                game_avg_times[game+'_avg_time'] = total_time/game_played if game_played > 0 else 0
                total_rounds = game_round_total[game+'_round_total']
                game_false_steps_avg_per_round[game+'_false_steps_avg_per_round'] = game_false_steps_count[game+'_false_steps_count'] / total_rounds if total_rounds > 0 else 0
                game_time_avg_per_round[game+'_time_avg_per_round'] = total_time / total_rounds if total_rounds > 0 else 0
                game_last_false_steps_avg_per_round[game+'_last_false_steps_avg_per_round'] = game_last_false_steps_count[game+'_last_false_steps_count'] / game_last_rounds[game+'_last_rounds'] if game_last_rounds[game+'_last_rounds'] > 0 else 0
            
            features.update(game_min_times.copy())
            features.update(game_max_times.copy())
            features.update(game_avg_times.copy())
            features.update(game_false_steps_avg_per_round.copy())
            features.update(game_time_avg_per_round.copy())
            features.update(game_last_false_steps_count.copy())
            features.update(game_last_rounds.copy())
            features.update(game_last_false_steps_avg_per_round.copy())
            
            
            activity_avg_times = {activity + '_avg_time':0 for activity in activities}
            for activity in activities:
                total_time = activity_times[activity+'_total_time']
                activity_played = activities_counter[activity+'_played_count']
                activity_avg_times[activity+'_avg_time'] = total_time/activity_played if activity_played > 0 else 0
            features.update(activity_min_times.copy())
            features.update(activity_max_times.copy())
            features.update(activity_avg_times.copy())
            
            durations.append((session.iloc[-1]['timestamp'] - session.iloc[0]['timestamp']).seconds)
            
            if test:
                features_list.append(features)
            elif true_attempts.shape[0] + false_attempts.shape[0] > 0:
                features_list.append(features)

        
        session_title_event_count = Counter(session['title_eventcode'])
        for key in session_title_event_count.keys():
            if key in title_eventcounts.keys():
                title_eventcounts[key] += session_title_event_count[key]
        
        session_event_code_count = Counter(session['event_code'])
        for key in session_event_code_count.keys():
            if key in event_codes:
                event_codes_count[str(key)] += session_event_code_count[key]
                
        title_counts[session_title] += 1
        
        if session.iloc[0]['type']=='Game':
            games_counter[session_title + '_played_count'] += 1
            game_times[session_title + '_total_time'] += session.iloc[-1]['game_time']
            game_min_times[session_title + '_min_time'] = min(session.iloc[-1]['game_time'], game_min_times[session_title + '_min_time'])
            game_max_times[session_title + '_max_time'] = max(session.iloc[-1]['game_time'], game_max_times[session_title + '_max_time'])
            
            false_steps = session.loc[(session['event_data'].str.contains('"correct":false'))].shape[0]
            game_false_steps_count[session_title + '_false_steps_count'] += false_steps
            rounds = session['event_data'].map(json.loads).map(lambda x: x['round']).max()
            game_round_max[session_title+'_round_max'] = max(rounds, game_round_max[session_title+'_round_max'])
            game_round_total[session_title+'_round_total'] += rounds
            game_last_false_steps_count[session_title+'_last_false_steps_count'] = false_steps
            game_last_rounds[session_title+'_last_rounds'] = rounds
            
        if session.iloc[0]['type']=='Activity':
            activities_counter[session_title + '_played_count'] += 1
            activity_times[session_title + '_total_time'] += session.iloc[-1]['game_time']
            activity_min_times[session_title + '_min_time'] = min(session.iloc[-1]['game_time'], activity_min_times[session_title + '_min_time'])
            activity_max_times[session_title + '_max_time'] = max(session.iloc[-1]['game_time'], activity_max_times[session_title + '_max_time'])
        
        session_count += 1
        
    if test:
        return features_list[-1]
    return features_list


# In[ ]:


def get_feature_df(df, test=False):
    features_list = []
    for i, (inst_id, user_data) in tqdm(enumerate(df.groupby(['installation_id'], sort=False))):
        user_features_list = feature_extractor(user_data, test)
        if test:
            features_list.append(user_features_list)
        else:
            features_list.extend(user_features_list)
    return pd.DataFrame(features_list)


# In[ ]:


reduced_test_df = get_feature_df(test_df, True)


# In[ ]:


reduced_test_df.head()


# In[ ]:


reduced_train_df = pd.read_csv("../input/dsb2019-features-1/reduced_train.csv")


# In[ ]:


reduced_train_df.shape


# In[ ]:


reduced_train_df.drop([col for col in reduced_train_df.columns if 'quit' in col], axis=1, inplace=True)
reduced_test_df.drop([col for col in reduced_test_df.columns if 'quit' in col], axis=1, inplace=True)


# In[ ]:


reduced_train_df.shape


# In[ ]:


columns_sorted = sorted(reduced_train_df.columns)
random.Random(42).shuffle(columns_sorted)
reduced_train_df = reduced_train_df.reindex(columns_sorted, axis=1)
reduced_test_df = reduced_test_df.reindex(columns_sorted, axis=1)


# In[ ]:


cols = list(reduced_train_df.columns)
cols_to_drop = ['accuracy_group', 'installation_id', 'game_session']
features = list(set(cols) - set(cols_to_drop))
categorical_features = ['assessment_title', 'assessment_world', 'first_session_is_welcome']


# In[ ]:


for category in categorical_features:
    reduced_train_df[category] = reduced_train_df[category].astype('category')
    reduced_test_df[category] = reduced_test_df[category].astype('category')


# In[ ]:


min_max_features = [col for col in reduced_train_df.columns if 'min_time' in col or 'max_time' in col]


# In[ ]:


class OptimizedRounder(object):
    
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])


    def coefficients(self):
        return self.coef_['x']


# In[ ]:


dist = Counter(reduced_train_df['accuracy_group'])
for k in dist:
    dist[k] /= len(reduced_train_df)

def classify(x, bound=[0.5,1.5,2.5]):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3
        
def post_process_pred(pred):
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(pred, acum * 100)
    pred = np.array(list(map(lambda x: classify(x, bound), pred)))
    return pred


# In[ ]:


len(features)


# In[ ]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'seed': 42,
    'num_threads': 4,
    'learning_rate': 0.04,
    'num_leaves': 61,
    'max_depth': 22,
    'lambda_l1': 5,
    'lambda_l2': 1,
}


# In[ ]:


reduced_train_df = shuffle(reduced_train_df, random_state=42)


# In[ ]:


folds=5
group_kfold = GroupKFold(n_splits=folds)
groups = reduced_train_df['installation_id']
X = reduced_train_df.drop(cols_to_drop, axis=1)
y = reduced_train_df['accuracy_group']

qwk_scores = []
mse_values = []
models = []
predictions_1 = np.empty((folds,reduced_test_df.shape[0]))
predictions_1_raw = np.empty((folds,reduced_test_df.shape[0]))
fold_coefficients_1 = np.empty((folds,3))
i = 0

for train_index, test_index in group_kfold.split(X, y, groups):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    
    fold_indices = pd.read_csv("../input/dsb2019cv-1/train_fold"+str(i+1)+"_indices.csv")
    X_train_sampled = X_train.loc[fold_indices['0'].values]
    y_train_sampled = y_train.loc[fold_indices['0'].values]
    
    fold_indices = pd.read_csv("../input/dsb2019cv-1/cv_fold"+str(i+1)+"_indices.csv")
    X_test_sampled = X_test.loc[fold_indices['0'].values]
    y_test_sampled = y_test.loc[fold_indices['0'].values]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test_sampled, y_test_sampled)
    gbm = lgb.train(params,
                lgb_train,
                valid_sets=(lgb_train, lgb_eval),
                verbose_eval=20,
                num_boost_round=1000,
                early_stopping_rounds=100)
    models.append(gbm)
    
    pred = gbm.predict(X_train)
    optR = OptimizedRounder()
    optR.fit(pred, y_train)
    coefficients = optR.coefficients()
    fold_coefficients_1[i,:] = coefficients
    #y_pred = post_process_pred(pred)
    y_pred = optR.predict(pred, coefficients)
    
    train_qwk_score = cohen_kappa_score(y_train, y_pred, weights="quadratic")
    train_mse_score = mean_squared_error(y_train, pred)
    
  
    
    pred = gbm.predict(X_test_sampled)
    #y_pred = post_process_pred(pred)
    y_pred = optR.predict(pred, coefficients)
    
    val_qwk_score = cohen_kappa_score(y_test_sampled, y_pred, weights="quadratic")
    val_mse_score = mean_squared_error(y_test_sampled, pred)
    
    qwk_scores.append((train_qwk_score, val_qwk_score))
    mse_values.append((train_mse_score, val_mse_score))
    
    print(confusion_matrix(y_test_sampled, y_pred))
    
    pred = gbm.predict(reduced_test_df.drop(cols_to_drop, axis=1))
    y_pred = optR.predict(pred, coefficients)
    
    predictions_1[i,:] = y_pred
    predictions_1_raw[i,:] = pred
    i += 1


# In[ ]:


qwk_scores, mse_values


# In[ ]:


np.asarray(qwk_scores).mean(axis=0), np.asarray(mse_values).mean(axis=0)


# In[ ]:


del reduced_train_df
del reduced_test_df
del specs_df
del train_labels_df
del X
del y
del X_train
del y_train
del X_test
del y_test

gc.collect()


# In[ ]:


del lgb_train
del lgb_eval
gc.collect()


# In[ ]:


del X_train_sampled
del y_train_sampled
del X_test_sampled
del y_test_sampled
gc.collect()


# In[ ]:


del groups
del gbm
del optR
del models
gc.collect()


# In[ ]:


for i in range(5):
    sns.distplot(predictions_1[i,:])


# In[ ]:


predictions = predictions_1 #np.concatenate([predictions_1, predictions_2], axis=0)


# In[ ]:


y_pred = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, predictions)


# In[ ]:


sns.countplot(y_pred)


# In[ ]:


Counter(y_pred)


# Model 2
# > 1 XGB MODEL with 4  folds and 2 Subfold with (diff seed trucated validation)

# In[ ]:


train = train_df 
keep_id=train[train.type=='Assessment'][['installation_id']].drop_duplicates()
train=train.merge(keep_id,on='installation_id',how='inner')
train.timestamp=pd.to_datetime(train.timestamp)


# In[ ]:


del keep_id
gc.collect()


# In[ ]:


test = test_df
test.timestamp=pd.to_datetime(test.timestamp)


# In[ ]:


np.random.seed(11)


# In[ ]:


user_activity={'Clip':0,'Activity':0,'Game':0,'Assessment':0}
# title count 
titles=set(set(train.title.unique()).union(set(test.title.unique())))

# add user activit title count
# words
worlds=train.world.value_counts().index
world_count =dict(zip(worlds,0*np.arange(len(worlds))))
user_activity.update(world_count)

list_of_event_code= list(set(set(train.event_code.unique()).union(set(test.event_code.unique()))))
list_of_event_id= list(set(set(train.event_id.unique()).union(set(test.event_id.unique()))))
Assessment_titles=list(train[train.type=='Assessment'].title.unique())



train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))


list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
list_of_user_activities

activities_labels = dict(zip(np.arange(len(list_of_user_activities)), 
                              list_of_user_activities))
all_session_type=list(train.type.unique())


# In[ ]:


session_title=list(set(train.title.unique()).union(set(test.title.unique())))
session_title_encode=dict(zip(session_title,np.arange(len(session_title))))
worlds=set(test.world.unique())
worlds=dict(zip(worlds,np.arange(len(worlds))))


# In[ ]:


activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
train['title'] = train['title'].map(activities_map)
test['title'] = test['title'].map(activities_map)


# [QWK](https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133)

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
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


# In[ ]:


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
        initial_coef = [1.12, 1.74, 2.225]
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


def update_counters(counter: dict,session, col: str):
            num_of_session_count = Counter(session[col])
            for k in num_of_session_count.keys():
                x = k
                if col == 'title':
                    x = activities_labels[k]
                counter[x] += num_of_session_count[k]
            return counter


# In[ ]:


def get_features(user_sample,test_set=False):
    # intialized all user activity 0
    user_activity={_type:0 for _type in all_session_type}
    
    
    #duration in clip Activity Game Assessment
    durations={_type+'_duration':[] for _type in all_session_type }
    
    #  user avg time in 
    user_avg_time={_type+'_avg':0 for _type in all_session_type }
    user_activity.update(user_avg_time)
    
    #  user avg time in 
    user_std_time={_type+'_std':0 for _type in all_session_type }
    user_activity.update(user_std_time)
    
    avg_accuracy_group={'avg_'+title:[] for title in Assessment_titles}
    
    
    # 
    last_accuracy_title = {'acc_' + title: -1 for title in Assessment_titles}
    last_game_time_title = {'lgt_' + title: 0 for title in Assessment_titles}
    ac_game_time_title = {'agt_' + title: 0 for title in Assessment_titles}
    avg_group_title ={'avg_'+title:-1 for title in Assessment_titles}
    
    std_group_title ={'std_'+title:-1 for title in Assessment_titles}
    #
    ac_true_attempts_title = {'ata_' + title: 0 for title in Assessment_titles}
    ac_false_attempts_title = {'afa_' + title: 0 for title in Assessment_titles}
       
    all_true_attempts_title = {'all_true_' + title: 0 for title in Assessment_titles}
    all_false_attempts_title = {'all_false_' + title: 0 for title in Assessment_titles}
   
    # counts 
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    #event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
        
    #Assessment_titles actions
    Assessment_titles_actions = {'Assessment_' + title+'actions': 0 for title in Assessment_titles}
    # no of action in each session_type
    activity_session_actions= {'session_size_' + activity +'actions':0 for activity in ['Clip','Activity','Game','Assessment']}

    #game_max_round={'game_level_max_'+game_type:0 for game_type in session_title_game}
    varity_features={'event_code_count':0,'title_count':0,'title_event_code_count':0}
    
    total_correct=0 # by single user
    total_incorrect=0 # by single user 
    total_accuracy=0 # total acuracy till session ith
    counter=0      # no of session
    accuracy=0    # accuracy in session ith 
    total_accuracy_group=0 # total accuracy before current Assessment
    total_actions=0 # no of action 
    all_assesment =[] # add detail abount and  befor assesment
    last_activity=0 # last session type 
    last_activity_title=0 # last session title
    last_world_name=0
    all_true=0
    all_false=0
    
    
    
    
    accuracy_groups = {0:0,1:0,2:0,3:0} # user accuracy groups 
    i=0
   
    
    for s_id,session in ((user_sample.groupby('game_session',sort=False))):
        
        session_type=session.type.iloc[0]
        session_title=activities_labels[session.title.iloc[0]]
        world=session.world.iloc[0]
        
        if test_set is True:
            flag=True
        else:
            if len(session)==1:
                flag=False
            else:
                flag=True 

                
        user_activity['session_count']=i
        i=i+1
        
        if session_type=='Assessment' and flag:

            # check all attempts  in session 
            attempts=session[(session['event_code']==4110) | (session['event_code']==4100)]
            # all true in one session 
            attempts_true=attempts['event_data'].str.contains("true").sum()
            # all false in one session 
            attempts_false=attempts['event_data'].str.contains("false").sum()
            
            # feature genration 
            features=user_activity.copy()
            features.update(last_accuracy_title.copy())
            features.update(ac_true_attempts_title.copy())
            features.update(ac_false_attempts_title.copy())
            features.update(last_game_time_title.copy())
            features.update(ac_game_time_title.copy())

            
            
            features.update(title_event_code_count.copy())
            features.update(event_code_count.copy())
           # features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(Assessment_titles_actions.copy())
            features.update(activity_session_actions.copy())
           #features.update(varity_features.copy())
            features.update(all_true_attempts_title.copy())
            features.update(all_false_attempts_title.copy())
            features.update(avg_group_title.copy())
            features.update(std_group_title.copy())
              # game 
            #features.update(game_type_all_true.copy())
            #features.update(game_type_all_false.copy())
            #features.update(game_type_all_true_false_ratio.copy())
            #features.update(game_max_round.copy())
            
              
            variety_features = [('var_event_code', event_code_count),
                               ('var_title', title_count),
                               ('var_title_event_code', title_event_code_count)]
            for name, dict_counts in variety_features:
                arr = np.array(list(dict_counts.values()))
                features[name] = np.count_nonzero(arr)
            
            
            # session title  exmple Chest Sorter
            features['session_title']=session_title
            # total correct true 
            features['total_correct']=total_correct
            # total correct falsefeatures.update(last_accuracy_title.copy())
           
            features['total_incorrect']=total_incorrect
            # total_correct is prev all correct + new correct in assesment 
            total_correct+=attempts_true
            total_incorrect+=attempts_false
            
            #total accuracy avg accuray till session i for unique installation_id
            features['total_accuracy']=total_accuracy/counter if counter>0 else 0
            # add new attempts in total counter 
            ac_true_attempts_title['ata_' + session_title] += attempts_true
            ac_false_attempts_title['afa_' + session_title] += attempts_false
            
            all_true=session['event_data'].str.contains("true").sum()
            all_false=session['event_data'].str.contains("false").sum()
            
            all_true_attempts_title['all_true_'+session_title]+=all_true
            all_false_attempts_title['all_false_'+session_title]+=all_false
            
            last_game_time_title['lgt_' + session_title] = session['game_time'].iloc[-1]
            ac_game_time_title['agt_' + session_title] += session['game_time'].iloc[-1]

            
            accuracy=attempts_true/(attempts_true+attempts_false) if                         (attempts_false+attempts_true)!=0 else 0
            
            last_accuracy_title['acc_' + session_title] = accuracy
            
            
            # Assessment_titles actions 
            Assessment_titles_actions['Assessment_'+session_title+'actions']+=len(session)
            features['total_actions'] = total_actions
            
            
            # summ all accuracy 
            total_accuracy+=accuracy
            # accuracy group for current session 
            if accuracy==0:
                features['accuracy_group']=0
            elif accuracy==1:
                features['accuracy_group']=3
            elif accuracy==0.5:
                features['accuracy_group']=2
            else:
                features['accuracy_group']=1     
                
            features.update(accuracy_groups)
            # avg accuracy for all assesemt with this assesment
            features['total_accuracy_group']=total_accuracy_group/counter if counter>0 else 0

            
            total_accuracy_group+=features['accuracy_group']
            accuracy_groups[features['accuracy_group']]+=1
            
            features['world']=world
            
            avg_accuracy_group['avg_'+session_title].append(features['accuracy_group'])
            for title in Assessment_titles:
                if len(avg_accuracy_group['avg_'+title]) is 0:
                    continue
                avg_group_title['avg_'+title]=np.mean(avg_accuracy_group['avg_'+title]) 
                std_group_title['std_'+title]=np.std(avg_accuracy_group['avg_'+title])
            
            
            if test_set:
                # if tese add assesment
                all_assesment.append(features)
            else:
                # if  assesment check if try or not 
                if (attempts_true+attempts_false)>0:
                    all_assesment.append(features)
            
    
            
            # counter for assesment no 
            
            counter+=1
        
          # add user avg time 
        
        
        for _type in all_session_type:
            if len(durations[_type+'_duration']) is 0:
                continue
            user_activity[_type+'_avg']=np.mean(durations[_type+'_duration']) 
            user_activity[_type+'_std']=np.std(durations[_type+'_duration']) 
        durations[session_type+'_duration'].append((session.iloc[-1]['timestamp']-session.iloc[0]['timestamp']).total_seconds())      
           
 
            
        #even count, event id count event title code combo count in one session 
        event_code_count = update_counters(event_code_count, session,"event_code")
       #event_id_count = update_counters(event_id_count,session, "event_id")
        title_count = update_counters(title_count,session, 'title')
        title_event_code_count = update_counters(title_event_code_count,session,'title_event_code')
        
        
        
        activity_session_actions['session_size_'+session_type+'actions']+=len(session)
        # total action in till previous session + this session 
        total_actions+=len(session)
        
        if last_activity != session_type:
            # add session activity  type 
            user_activity[session_type]+=1
            last_activity=session_type    
    if test_set:
        return all_assesment[-1]
    else:
        return all_assesment


# In[ ]:


compile_data =[]
user_id =[]
for i , (ins_id, user_sample) in tqdm_notebook(enumerate(train.groupby('installation_id', sort=False))):
    sample=get_features(user_sample)
    compile_data+=sample
    #assement session  sample user id 
    user_id+=[i]*len(sample)


# In[ ]:


j=i
for i, (ins_id, user_sample) in tqdm(enumerate(test.groupby('installation_id', sort=False))):
    sample=get_features(user_sample)[:-1]
    compile_data+=sample
    #assement session  sample user i'],d 
    user_id+=[i+j]*len(sample)


# In[ ]:


train_df =pd.DataFrame(compile_data)
train_df.drop(['Clip_avg'],axis=1,inplace=True)
train_df.drop(['Clip_std'],axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df.columns=["".join (c if c.isalnum() else "_" for c in str(x)) for x in train_df.columns]
train_df=train_df[sorted(["".join (c if c.isalnum() else "_" for c in str(x)) for x in train_df.columns])]
zero_col=(train_df.sum()==0).values
train_df.columns[zero_col]


# In[ ]:


train_df.drop(train_df.columns[zero_col],axis=1,inplace=True)
train_df=train_df[sorted(train_df.columns)]
session_title_encode=dict(zip(session_title,np.arange(len(session_title))))
train_df.session_title = train_df.session_title.map(session_title_encode)
train_df.world=train_df.world.map(worlds)


# In[ ]:


train_df['user_id']=user_id


# In[ ]:


a0 = 1.12
a1 = 1.74
a2 = 2.225


# In[ ]:


def Cappa(y_pred, train_data,eps=1e-5):
    labels=train_data.get_label()
    
    y_pred[y_pred <= a0] = 0
    y_pred[np.where(np.logical_and(y_pred > a0, y_pred <= a1))] = 1
    y_pred[np.where(np.logical_and(y_pred > a1, y_pred <= a2))] = 2
    y_pred[y_pred > a2] = 3
    return 'cappa',1-cohen_kappa_score(labels, y_pred,weights='quadratic'), False


# In[ ]:


np.random.seed(10)


# In[ ]:


params = {
        'colsample_bytree':0.9839376346182411,
        'feature_fraction': 0.8637232338991787,
        'gamma': 0.024799499154065652,
        'in_child_weight': 1.882771434364163,
        'learning_rate': 0.02,
        'max_depth': 6,
        'reg_alpha': 1.99440604517944,
        'reg_lambda': 1.9731522286720549,
        'subsample': 0.5538619922076616,
        'random_state' :7,
         'nthread ':-1,
        'objective':'reg:squarederror',
        'n_estimators':500
    }


# In[ ]:


folds = 4
coefficients=[]
g_kf = GroupKFold(n_splits=folds)
outputs =[]
fold_coefficients_2 = np.empty((folds*2,3))

models = []
j=0
for train_index, val_index in g_kf.split(train_df, train_df['accuracy_group'],train_df['user_id']): 
    
    train_X = train_df.iloc[train_index]
    train_X,train_y =  train_X.drop(['user_id','accuracy_group'],axis=1),train_X['accuracy_group']
    
    
    for i in range(2):
        val_X = train_df.iloc[val_index]
        val_X=val_X.reset_index(drop=['index'])
        np.random.seed(10+i*10)
        val_X=val_X.loc[np.random.permutation(len(val_X))]
        val_X=val_X.groupby('user_id', group_keys=False).apply(lambda df: df.sample(1,replace=True,random_state=10+i*30))
        
        val_X,val_y = val_X.drop(['user_id','accuracy_group'],axis=1),val_X['accuracy_group']
        print(val_y.value_counts())
        
        train_set = xgb.DMatrix(train_X, train_y)
        val_set = xgb.DMatrix(val_X, val_y)
        model_xgb=xgb.train(params, train_set, 
                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=30, early_stopping_rounds=100)

        pred=model_xgb.predict(train_set)
        optR = OptimizedRounder()
        optR.fit(pred.reshape(-1,), train_y)
        coefficients.append(optR.coefficients())
        fold_coefficients_2[j,:] = optR.coefficients()

        a0=coefficients[j][0]
        a1=coefficients[j][1]
        a2=coefficients[j][2]

        y_pred=model_xgb.predict(val_set)
        y_pred[y_pred <= a0] = 0
        y_pred[np.where(np.logical_and(y_pred > a0, y_pred <= a1))] = 1
        y_pred[np.where(np.logical_and(y_pred > a1, y_pred <= a2))] = 2
        y_pred[y_pred > a2] = 3
        expected_y=val_y
        #a0,a1,a2=0.5,1.5,2.5
        print('---------------------------cappa-value------------------------')
        value=cohen_kappa_score(expected_y,y_pred,weights='quadratic')
        outputs.append(value)
        print(value)
        print('----------------------classification_report-------------------')
        print(metrics.classification_report(expected_y, y_pred))
        print('--------------------------confusion_matrix--------------------')
        print(metrics.confusion_matrix(expected_y, y_pred))
        print('--------------------------------------------------------------')
        models.append(model_xgb)
        j=j+1


# In[ ]:


print(outputs)
print(sum(outputs)/folds/2)


# In[ ]:


fold_coefficients_2


# In[ ]:



preds = []
for i,model in enumerate(models):
    a0,a1,a2 =coefficients[i][0],coefficients[i][1],coefficients[i][2]
    train_df['session_title']=train_df['session_title'].astype('int')
    train_set=xgb.DMatrix(train_df.drop(['user_id','accuracy_group'],axis=1))
    pred =model.predict(train_set)
    pred=np.array(pred)
    pred[pred <= a0] = 0
    pred[np.where(np.logical_and(pred > a0, pred <= a1))] = 1
    pred[np.where(np.logical_and(pred > a1, pred <= a2))] = 2
    pred[pred > a2] = 3
    preds.append(pred)
    
result=np.array(preds).T
pred=[]
for i in range(result.shape[0]):
    test_list =list(result[i,:]) 
    res = max(set(test_list), key = test_list.count) 
    pred.append(res)
cohen_kappa_score(train_df['accuracy_group'],pred,weights='quadratic')


# In[ ]:


train_df['accuracy_group'].value_counts().plot.bar()


# In[ ]:


print(pred.count(3))
print(pred.count(0))
print(pred.count(1))
print(pred.count(2))


# In[ ]:


counts=[(i,pred.count(i)) for i in set(pred)]
labels, ys = zip(*counts)
xs = np.arange(len(labels)) 
width = 0.4
fig = plt.figure()                                                               
ax = fig.gca()  #get current axes
ax.bar(xs, ys, width, align='center')


# In[ ]:


ax=xgb.plot_importance(model_xgb,max_num_features=60,importance_type='gain')
fig=ax.figure
fig.set_size_inches(8, 10)


# In[ ]:


ax=xgb.plot_importance(model_xgb,max_num_features=60,importance_type='total_gain')
fig=ax.figure
fig.set_size_inches(8, 10)


# In[ ]:


ax=xgb.plot_importance(model_xgb,max_num_features=60,importance_type='weight')
fig=ax.figure
fig.set_size_inches(8, 10)


# In[ ]:


del model_xgb


# In[ ]:


new_test = []
for ins_id, user_sample in tqdm_notebook(test.groupby('installation_id', sort=False)):
    a = get_features(user_sample, test_set=True)
    new_test.append(a)


# In[ ]:


X_test = pd.DataFrame(new_test)

X_test.columns=["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]

X_test.session_title = X_test.session_title.map(session_title_encode)
X_test.world=X_test.world.map(worlds)


# In[ ]:


X_test=X_test[train_df.drop(['user_id','accuracy_group'],axis=1).columns]


# In[ ]:


preds = []
predictions_2_raw = np.empty((8,X_test.shape[0]))
for i,model in enumerate(models):
    a0,a1,a2= coefficients[i][0],coefficients[i][1],coefficients[i][2]
    X_test['session_title']=X_test['session_title'].astype('int')
    test_set=xgb.DMatrix(X_test)
    pred =model.predict(test_set)
    predictions_2_raw[i,:] = pred
    pred=np.array(pred)
    pred[pred <= a0] = 0
    pred[np.where(np.logical_and(pred > a0, pred <= a1))] = 1
    pred[np.where(np.logical_and(pred > a1, pred <= a2))] = 2
    pred[pred > a2] = 3
    preds.append(pred)
    
result=np.array(preds).T
pred=[]
for i in range(result.shape[0]):
    test_list =list(result[i,:]) 
    res = max(set(test_list), key = test_list.count) 
    pred.append(res)
pred[:10]


# In[ ]:


result.T.shape


# In[ ]:


predictions_1_raw.shape, predictions_2_raw.shape


# In[ ]:


luffy_predictions = np.asarray(preds).copy()
luffy_predictions


# In[ ]:


naruto_predictions = predictions.copy()
naruto_predictions


# In[ ]:


predictions_combined = np.concatenate([ naruto_predictions,luffy_predictions], axis=0)


# In[ ]:



del X_test
del train_df
del models
del train_X
gc.collect()


# 3 Model public 
# Lightgbm Model with 5 folds used

# In[ ]:


def read_data():

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train_labels, specs, sample_submission
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
    # hour
    train['hour'] = train['timestamp'].dt.hour
    test['hour'] = test['timestamp'].dt.hour
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
clip_time = {'Welcome to Lost Lagoon!':19,'Tree Top City - Level 1':17,'Ordering Spheres':61, 'Costume Box':61,
        '12 Monkeys':109,'Tree Top City - Level 2':25, 'Pirate\'s Tale':80, 'Treasure Map':156,'Tree Top City - Level 3':26,
        'Rulers':126, 'Magma Peak - Level 1':20, 'Slop Problem':60, 'Magma Peak - Level 2':22, 'Crystal Caves - Level 1':18,
        'Balancing Act':72, 'Lifting Heavy Things':118,'Crystal Caves - Level 2':24, 'Honey Cake':142, 'Crystal Caves - Level 3':19,
        'Heavy Heavier Heaviest':61}

def cnt_miss(df):
    cnt = 0
    for e in range(len(df)):
        x = df['event_data'].iloc[e]
        y = json.loads(x)['misses']
        cnt += y
    return cnt


# In[ ]:


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
    game_time_dict = {'Clip_gametime':0, 'Game_gametime':0, 'Activity_gametime':0, 'Assessment_gametime':0}
    Assessment_mean_event_count = 0
    Game_mean_event_count = 0
    Activity_mean_event_count = 0
    mean_game_round = 0
    mean_game_duration = 0 
    mean_game_level = 0
    accumulated_game_miss = 0
    
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
    clip_durations = []
    Activity_durations = []
    Game_durations = []
    
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
        
    # last features
    sessions_count = 0
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
                    
        if session_type == 'Clip':
            clip_durations.append((clip_time[activities_labels[session_title]]))
        
        if session_type == 'Activity':
            Activity_durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1])/2.0
        
        if session_type == 'Game':
            Game_durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            Game_mean_event_count = (Game_mean_event_count + session['event_count'].iloc[-1])/2.0
            
            game_s = session[session.event_code == 2030]   
            misses_cnt = cnt_miss(game_s)
            accumulated_game_miss += misses_cnt
            
            try:
                game_round = json.loads(session['event_data'].iloc[-1])["round"]
                mean_game_round =  (mean_game_round + game_round)/2.0
            except:
                pass

            try:
                game_duration = json.loads(session['event_data'].iloc[-1])["duration"]
                mean_game_duration = (mean_game_duration + game_duration) /2.0
            except:
                pass
            
            try:
                game_level = json.loads(session['event_data'].iloc[-1])["level"]
                mean_game_level = (mean_game_level + game_level) /2.0
            except:
                pass
            
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
            # features.update(game_time_dict.copy())
            
            features['installation_session_count'] = sessions_count
            features['hour'] = session['hour'].iloc[-1]
            features['Assessment_mean_event_count'] = Assessment_mean_event_count
            features['Game_mean_event_count'] = Game_mean_event_count
            features['Activity_mean_event_count'] = Activity_mean_event_count
            features['mean_game_round'] = mean_game_round
            features['mean_game_duration'] = mean_game_duration
            features['mean_game_level'] = mean_game_level
            features['accumulated_game_miss'] = accumulated_game_miss
            
            variety_features = [('var_event_code', event_code_count),
                              ('var_event_id', event_id_count),
                               ('var_title', title_count),
                               ('var_title_event_code', title_event_code_count)]
            
            for name, dict_counts in variety_features:
                arr = np.array(list(dict_counts.values()))
                features[name] = np.count_nonzero(arr)
                 
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
                features['duration_std'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
                features['duration_std'] = np.std(durations)
            if clip_durations == []:
                features['Clip_duration_mean'] = 0
                features['Clip_duration_std'] = 0
            else:
                features['Clip_duration_mean'] = np.mean(clip_durations)
                features['Clip_duration_std'] = np.std(clip_durations)
                
            if Activity_durations == []:
                features['Activity_duration_mean'] = 0
                features['Activity_duration_std'] = 0
            else:
                features['Activity_duration_mean'] = np.mean(Activity_durations)
                features['Activity_duration_std'] = np.std(Activity_durations)
                
            if Game_durations == []:
                features['Game_duration_mean'] = 0
                features['Game_duration_std'] = 0
            else:
                features['Game_duration_mean'] = np.mean(Game_durations)
                features['Game_duration_std'] = np.std(Game_durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1])/2.0
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
        
        sessions_count += 1
        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter
            
        game_time_dict[session_type+'_gametime'] = (game_time_dict[session_type+'_gametime'] + (session['game_time'].iloc[-1]/1000.0))/2.0
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


# In[ ]:


def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample)
    
    # add extenal data 
    for i, (ins_id, user_sample) in tqdm(enumerate(test.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample)[:-1]
    
    
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
        
        
        
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals


# In[ ]:


y_pred_j=[]
class Base_Model(object):
    
    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = 'accuracy_group'
        self.cv = self.get_cv()
        self.verbose = verbose
        self.params = self.get_params()
        self.y_pred, self.score, self.model = self.fit()
        
    def train_model(self, train_set, val_set):
        raise NotImplementedError
        
    def get_cv(self):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return cv.split(self.train_df, self.train_df[self.target])
    
    def get_params(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
        
    def fit(self):
        oof_pred = np.zeros((len(reduce_train), ))
        y_pred = np.zeros((len(reduce_test), ))
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model = self.train_model(train_set, val_set)
            conv_x_val = self.convert_x(x_val)
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            x_test = self.convert_x(self.test_df[self.features])
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
            y_pred_j.append(model.predict(x_test).reshape(y_pred.shape))
            print('Partial score of fold {} is: {}'.format(fold, eval_qwk_lgb_regr(y_val, oof_pred[val_idx])[1]))
        _, loss_score, _ = eval_qwk_lgb_regr(self.train_df[self.target], oof_pred)
        if self.verbose:
            print('Our oof cohen kappa score is: ', loss_score)
        return y_pred, loss_score, model


# In[ ]:


class Lgb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set
        
    def get_params(self):
        params = {'n_estimators':5000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,  
                    'lambda_l2': 1,
                    'early_stopping_rounds': 100
                    }
        return params
class Xgb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return xgb.train(self.params, train_set, 
                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=verbosity, early_stopping_rounds=100)
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = xgb.DMatrix(x_train, y_train)
        val_set = xgb.DMatrix(x_val, y_val)
        return train_set, val_set
    
    def convert_x(self, x):
        return xgb.DMatrix(x)
        
    def get_params(self):
        params = {'colsample_bytree': 0.8,                 
            'learning_rate': 0.01,
            'max_depth': 10,
            'subsample': 1,
            'objective':'reg:squarederror',
            #'eval_metric':'rmse',
            'min_child_weight':3,
            'gamma':0.25,
            'n_estimators':5000}

        return params


# In[ ]:


train['title'] = train['title'].map(dict(zip(activities_map.values(),activities_map.keys())))
test['title'] = test['title'].map(dict(zip(activities_map.values(),activities_map.keys())))


# In[ ]:


train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
# tranform function to get the train and test set
reduce_train, reduce_test, categoricals = get_train_and_test(train, test)
reduce_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]
reduce_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_test.columns]


# In[ ]:


del train,test


# In[ ]:


# call feature engineering function
features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
features = [x for x in features if x not in ['accuracy_group', 'installation_id']]


# In[ ]:


to_exclude = [] 
ajusted_test = reduce_test.copy()
for feature in ajusted_test.columns:
    if feature not in ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title']:
        data = reduce_train[feature]
        train_mean = data.mean()
        data = ajusted_test[feature] 
        test_mean = data.mean()
        try:
            error = stract_hists(feature, adjust=True)
            ajust_factor = train_mean / test_mean
            if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:
                to_exclude.append(feature)
                print(feature, train_mean, test_mean, error)
            else:
                ajusted_test[feature] *= ajust_factor
        except:
            to_exclude.append(feature)
            print(feature, train_mean, test_mean)


# In[ ]:


features = [x for x in features if x not in (to_exclude)]
reduce_train[features].shape


# In[ ]:


def eval_qwk_lgb_regr(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """
    dist = Counter(reduce_train['accuracy_group'])
    for k in dist:
        dist[k] /= len(reduce_train)
    reduce_train['accuracy_group'].hist()
    
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)

    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True
def cohenkappa(ypred, y):
    y = y.get_label().astype("int")
    ypred = ypred.reshape((4, -1)).argmax(axis = 0)
    loss = cohenkappascore(y, y_pred, weights = 'quadratic')
    return "cappa", loss, True


# In[ ]:


lgb_model = Lgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals)
final_pred = lgb_model.y_pred 

final_pred=np.array(y_pred_j)


# In[ ]:


dist = Counter(reduce_train['accuracy_group'])
for k in dist:
    dist[k] /= len(reduce_train)
reduce_train['accuracy_group'].hist()

acum = 0
bound = {}
for i in range(3):
    acum += dist[i]
    #avg of all model 
    bound[i] = sum([np.percentile(final_pred[0,:], acum * 100) for i in range(5)])/5
    print(f'accum {acum} and bound {bound}')
print(bound)


# In[ ]:



def classify(x):
    if x <= bound[0]:
        return 0
    elif x <= bound[1]:
        return 1
    elif x <= bound[2]:
        return 2
    else:
        return 3

for i in range(5):
    final_pred[i,:] = np.array(list(map(classify, final_pred[i,:])))
    


# In[ ]:


public_predictions = np.asarray(final_pred).copy()
public_predictions


# In[ ]:


predictions_combined = np.concatenate([predictions_combined, public_predictions], axis=0)


# In[ ]:


predictions_combined.shape


# In[ ]:


def the_judge(x, preds):
    if Counter(x).most_common(1)[0][1]>=11:
        return Counter(x).most_common(1)[0][0]
    else:
        candidates = (Counter(x).most_common(2)[0][0], Counter(x).most_common(2)[1][0])
        left_candidate = int(min(candidates[0], candidates[1]))
        right_candidate = int(max(candidates[0], candidates[1]))
        left_candidate_vote = 0
        right_candidate_vote = 0
        left_boundary_index = left_candidate
        right_boundary_index = right_candidate - 1
        left_margin = 0
        right_margin = 0
        for i in range(fold_coefficients_1.shape[0]):
            if x[i]==left_candidate:
                left_candidate_vote+=1
                left_margin += (fold_coefficients_1[i,left_boundary_index] - preds[i])
            elif x[i]==right_candidate:
                right_candidate_vote+=1
                right_margin += (preds[i] - fold_coefficients_1[i,right_boundary_index])
        offset = fold_coefficients_1.shape[0]
        for i in range(fold_coefficients_2.shape[0]):
            if x[offset+i]==left_candidate:
                left_candidate_vote+=1
                left_margin += (fold_coefficients_2[i,left_boundary_index] - preds[offset+i])
            elif x[offset+i]==right_candidate:
                right_candidate_vote+=1
                right_margin += (preds[offset+i] - fold_coefficients_2[i,right_boundary_index])
        left_margin_avg = left_margin/left_candidate_vote if left_candidate_vote > 0 else 0
        right_margin_avg = right_margin/right_candidate_vote if right_candidate_vote > 0 else 0
        
        if left_margin_avg > right_margin_avg:
            return left_candidate
        else:
            return right_candidate


# In[ ]:


predictions_raw_combined = np.concatenate([predictions_1_raw, predictions_2_raw], axis=0)


# In[ ]:


predictions_raw_combined.shape


# In[ ]:


y_pred = []
for i in range(predictions_combined.shape[1]):
    y_pred.append(the_judge(predictions_combined[:,i], predictions_raw_combined[:,i]))


# In[ ]:


Counter(y_pred)


# In[ ]:


submission =pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')


# In[ ]:


submission['accuracy_group'] = np.int64(y_pred)
submission


# In[ ]:


submission.to_csv('submission.csv', index=None)
submission.head()


# In[ ]:


sns.countplot(submission['accuracy_group'])

