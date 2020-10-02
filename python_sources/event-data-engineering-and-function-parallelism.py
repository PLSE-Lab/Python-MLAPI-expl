#!/usr/bin/env python
# coding: utf-8

# <h3>**Parallelization of the assessments function**</h3>
# <p>
# In data science speed is very important because sometimes we have to experiment a lot to find out the best solution for the problem we are trying to solve.
# That is where the (data-) parallization comes in. It reduces the computation time of a function so that we can try other experiments if we are not satisfied with the result.
# </p>
# <p>
#     After parallelizing the function I could reduce the computation time of the "add_results(groupedBy, filename)" method by 75%.
# </p>
# 
# <h4 style="color:Blue">
#     **Vote up if you think it is helpfull**
# </h4>

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


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:25:56 2019

@author: D. Mp. Bazola
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, make_scorer, confusion_matrix, f1_score, fbeta_score
import dask.dataframe as dd
import datetime as dt
from sklearn.compose import ColumnTransformer
import pickle
import os
import json
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import SparsePCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from dask import delayed # to allow parallel computation
import multiprocessing as mp

import lightgbm
import math
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Process, Manager, Queue
import multiprocessing as mp


cpu_n = mp.cpu_count()


# In[ ]:



specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
ev_ids = specs['event_id'].values


train_raw = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", #nrows=500000,
                         dtype={
                                 'event_id': np.object,
                                 'game_session': np.object,
                                 'event_data': np.object,
                                 'installation_id': np.object,
                                 'event_count': np.int64,
                                 'event_code': np.int64,
                                 'game_time': np.int64,
                                 'title': np.object,
                                 'type': np.object,
                                 'world': np.object
                                }, low_memory=False)
#timestamp
train_raw['timestamp'] = pd.to_datetime(train_raw['timestamp'], errors='coerce')
#train_raw['year'] = train_raw.timestamp.dt.year
train_raw['hours'] = train_raw.timestamp.dt.hour
#train_raw['day'] = train_raw.timestamp.dt.day
#train_raw['month'] = train_raw.timestamp.dt.month 
#train_raw['min'] = train_raw.timestamp.dt.minute
train_raw['dayofweek'] = train_raw.timestamp.dt.day_name()
# 
print('big train shape', train_raw.shape)
#np.random.seed(0) 


# In[ ]:


print('reading test data set - started -')
pred_raw = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv",
                         dtype={
                                 'event_id': np.object,
                                 'game_session': np.object,
                                 'event_data': np.object,
                                 'installation_id': np.object,
                                 'event_count': np.int64,
                                 'event_code': np.int64,
                                 'game_time': np.int64,
                                 'title': np.object,
                                 'type': np.object,
                                 'world': np.object
                                }, low_memory=False)
#timestamp
pred_raw['timestamp'] = pd.to_datetime(pred_raw['timestamp'], errors='coerce')
#pred_raw['year'] = pred_raw.timestamp.dt.year
pred_raw['hours'] = pred_raw.timestamp.dt.hour
#pred_raw['day'] = pred_raw.timestamp.dt.day
#pred_raw['month'] = pred_raw.timestamp.dt.month
#pred_raw['min'] = pred_raw.timestamp.dt.minute
pred_raw['dayofweek'] = pred_raw.timestamp.dt.day_name()
print('reading test data set - finish -')
#


# In[ ]:



##########################################################################################################
global df_main
global df_main_columns
#
df_main_columns = ['accuracy_group', 'event_id', 'game_session', 'timestamp', 
                   'event_data', 'installation_id', 'event_count', 'event_code', 'game_time', 'title', 
                  'type', 'world', 'hours', 'dayofweek']


titles = ['12 Monkeys', 'Air Show', 'All Star Sorting', 'Balancing Act', 'Bird Measurer (Assessment)', 
          'Bottle Filler (Activity)', 'Bubble Bath', 'Bug Measurer (Activity)', 'Cart Balancer (Assessment)', 
          'Cauldron Filler (Assessment)', 'Chest Sorter (Assessment)', 'Chicken Balancer (Activity)', 
          'Chow Time', 'Costume Box', 'Crystal Caves - Level 1', 'Crystal Caves - Level 2', 
          'Crystal Caves - Level 3', 'Crystals Rule', 'Dino Dive', 'Dino Drink', 'Egg Dropper (Activity)', 
          'Fireworks (Activity)', 'Flower Waterer (Activity)', 'Happy Camel', 'Heavy, Heavier, Heaviest', 
          'Honey Cake', 'Leaf Leader', 'Lifting Heavy Things', 'Magma Peak - Level 1', 'Magma Peak - Level 2', 
          'Mushroom Sorter (Assessment)', 'Ordering Spheres', 'Pan Balance', "Pirate's Tale", 'Rulers', 
          'Sandcastle Builder (Activity)', 'Scrub-A-Dub', 'Slop Problem', 'Treasure Map', 
          'Tree Top City - Level 1', 'Tree Top City - Level 2', 'Tree Top City - Level 3', 
          'Watering Hole (Activity)', 'Welcome to Lost Lagoon!']


'''
train_labels_raw = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv", 
                         dtype={
                                 'game_session': np.object,
                                 'installation_id': np.object,
                                 'title': np.object,
                                 'num_correct': np.int64,
                                 'num_incorrect': np.int64,
                                 'accuracy': np.float64,
                                 'accuracy_group': np.int64
                                }, low_memory=False)

'''

#__________________________________________________________________________________________________


event_data_incorrect = ['df4fe8b6', 'd88e8f25', 'c277e121', '160654fd', 'ea296733', '5859dfb6', 
      'e04fb33d', '28a4eb9a', '7423acbc', 'e57dd7af', '04df9b66', '2230fab4', 
      'c51d8688', '1af8be29', '89aace00', '763fc34e', '5290eab1', '90ea0bac', 
      '8b757ab8', 'e5734469', '9de5e594', 'd45ed6a1', 'ac92046e', 'ad2fc29c', 
      '5de79a6a', '88d4a5be', '907a054b', 'e37a2b78', '31973d56', '44cb4907', 
      '0330ab6a', '3bf1cf26']
#
event_data_correct = ['df4fe8b6', 'd88e8f25', 'c277e121', '160654fd', 'ea296733', '5859dfb6', 
      'e04fb33d', '28a4eb9a', '7423acbc', 'e57dd7af', '04df9b66', '2230fab4', 
      'c51d8688', '1af8be29', '89aace00', '763fc34e', '5290eab1', '90ea0bac', 
      '8b757ab8', 'e5734469', '9de5e594', 'd45ed6a1', 'ac92046e', 'ad2fc29c', 
      '5de79a6a', '88d4a5be', '907a054b', 'e37a2b78', '31973d56', '44cb4907', 
      '0330ab6a', '3bf1cf26', 'df4fe8b6', 'd88e8f25', 'c277e121', '160654fd', 
      'ea296733', '5859dfb6', 'e04fb33d', '28a4eb9a', '7423acbc', 'e57dd7af', 
      '04df9b66', '2230fab4', 'c51d8688', '1af8be29', '89aace00', '763fc34e', 
      '5290eab1', '90ea0bac', '8b757ab8', 'e5734469', '9de5e594', 'd45ed6a1', 
      'ac92046e', 'ad2fc29c', '5de79a6a', '88d4a5be', '907a054b', 'e37a2b78', 
      '31973d56', '44cb4907', '0330ab6a', '3bf1cf26']
#
 


# ASSESSMENTS
def add_results(groupedBy, filename):
    print('add_results - started -')
    assessments = groupedBy[((groupedBy['event_code']==4100) & (groupedBy['type']=='Assessment') & (groupedBy['title']!='Bird Measurer (Assessment)')) | ((groupedBy['event_code']==4110) & (groupedBy['type']=='Assessment') & (groupedBy['title']=='Bird Measurer (Assessment)'))]  
    #
    for column in assessments:
        if column == 'event_data':
            correct_df = pd.io.json.json_normalize(assessments[column].apply(json.loads))
            assessments['num_correct'] = [1 if c==True else 0 for c in correct_df['correct']]
            assessments['num_incorrect'] = [1 if c==False else 0 for c in correct_df['correct']]
    #    
    df_main_columns = assessments.columns.values
    print('add_results - finish -')
    return make_assessment2(assessments, filename)
#




def make_assessment2(assessments, filename):
    #print('make_assessment - started -')
    df_main = pd.DataFrame(columns=df_main_columns)
    users = np.unique(assessments.index.get_level_values(0))
    for idx,u in tqdm(enumerate(users), total=len(users)):
        gss = np.unique(assessments.loc[u].index.get_level_values(0))
        for idxx,ii in enumerate(gss):
            #SUM THEM
            corrects = list()
            in_corrects = list() 
            in_corrects.append(assessments.loc[(u,ii), 'num_incorrect'].sum())
            corrects.append(assessments.loc[(u,ii), 'num_correct'].sum())  
            #
            new_df = assessments.loc[(u,ii)].tail(1)
            if isinstance(assessments.loc[(u,ii)], pd.Series):
                new_df = assessments.loc[(u,ii)].to_frame().T
            #
            # LABEL THEM
            new_df["accuracy_group"] = np.where((np.asarray(corrects) == 1) & (np.asarray(in_corrects) == 0), 3,
            np.where((np.asarray(corrects) == 1) & (np.asarray(in_corrects) == 1), 2,
            np.where((np.asarray(corrects) == 1) & (np.asarray(in_corrects) > 1), 1, 0)))
            #
            df_main = df_main.append(new_df, ignore_index=True, sort=False)
            #
        #
        df_main = add_assessment_count_cumsum(df_main)   
        #
        #print('*', (idx + 1), '/', len(users), '*')
    # 
    df_main.reset_index(drop=True, inplace=True)
    #
    return df_main


def make_assessment(groupedBy, filename):
    print('make_assessment - starting... -')
    df_main = pd.DataFrame(columns=df_main_columns)
    #print('groupedBy.index', groupedBy.index)
    l = [i for i,g in groupedBy.index]
    ll = list(dict.fromkeys(l))
    for ii in ll:
        #SUM THEM
        corrects = list()
        in_corrects = list() 
        in_corrects.append(groupedBy['num_incorrect'][ii].sum())
        corrects.append(groupedBy['num_correct'][ii].sum())
        #add new rows to main dataframe
        new_df = groupedBy.loc[ii].tail(1)
        # LABEL THEM
        new_df["accuracy_group"] = np.where((np.asarray(corrects) == 1) & (np.asarray(in_corrects) == 0), 3,
        np.where((np.asarray(corrects) == 1) & (np.asarray(in_corrects) == 1), 2,
        np.where((np.asarray(corrects) == 1) & (np.asarray(in_corrects) > 1), 1, 0)))
        #append that row
        df_main = df_main.append(new_df, ignore_index=True, sort=False)
    #
    df_main.reset_index(drop=True, inplace=True)
    df_main = df_main.drop([v for v in df_main.columns.values if v not in df_main_columns], 1)
    #print(df_main['accuracy_group'])
    #history_old *starts*
    pd.DataFrame(df_main).to_csv(filename + '.csv', encoding='utf-8', index=False) #
    print('history_old file created!')
    #history_old *ends*
    print('make_assessment - finish -')
    return df_main
    #return df_main



def train_set_gen_cat(user, start_date, end_date):
    tsg_df = pd.DataFrame(columns=['title'])
    #
    ranged_user = user[(user.index >= start_date) & (user.index <= end_date)]
    ranged_user['timestamp_diff'] = (end_date - start_date).total_seconds()
    ranged_user = ranged_user.reset_index(drop=True)
    #
    aggregations = {
        'game_time': ['mean'],
        'event_count': ['mean']
    }          
    #
    tsg_df=tsg_df.append(super_df_title).reset_index(drop=True)
    return tsg_df



 


def train_set_gen(train_raw, history, filename):  
    t1 = time.time()
    print('train_set_gen - starting... -')
    tsg_df = pd.DataFrame(columns=['game_time', 'event_count', 'event_code'])
    print('userData reading...')
    userData = train_raw.groupby(['installation_id']).apply(lambda x: x.reset_index(drop=True))
    print('userData finish')
    #
    for idx, user_history_value in tqdm(enumerate(np.array(history.values)), total=history.shape[0]): 
        if filename == 'test':
            user_id = user_history_value[0]
            game_session = user_history_value[2]
        else:
            user_id = user_history_value[5]
            game_session = user_history_value[2]
        #
        user = userData.loc[user_id] #get the first action 
        user = user.set_index(pd.DatetimeIndex(user['timestamp']))
        #start and end date to get users dataframe before the assessment
        start_date = user.index[0] #get the first action timestamp
        end_date = (user.loc[((user.game_session==game_session)&(user.event_count==1)&(user.event_code==2000)&(user.game_time==0))]['timestamp'])[0]
        #get the users dataframe before the assessment
        ranged_user = user[(user.index >= start_date) & (user.index <= end_date)]
        ranged_user['timestamp_diff'] = (end_date - start_date).total_seconds()
        ranged_user = ranged_user.reset_index(drop=True)
        #
        aggregations = {
            'game_time': ['mean'],
            'event_count': ['mean'],
            'title': ['count']
        }
        #
        ############### numeric and title ###########
        super_df = (ranged_user.groupby(['installation_id']).agg(aggregations).reset_index(drop=True))
        super_df.columns = ["_".join(x) for x in super_df.columns.ravel()]
        #
        for n in ['title', 'type', 'world', 'event_code', 'event_id']:
            for agg2 in [{'game_time': ['mean']}, {'event_count': ['mean']}, {'title': ['count']}]:
                name = ranged_user.groupby([n]).agg(agg2)
                name = pd.DataFrame(name).transpose()  
                name.columns = [''.join(n)+"_"+str(x).replace(" ", "_") for x in name.columns.ravel()]  
                name=name.reset_index(drop=True) 
                super_df[name.columns.values] = name
        
        ############### categories #################
        for n in ['title', 'type', 'world', 'event_code', 'event_id']:
            name = ranged_user.groupby([n]).agg(aggregations).mean()  
            name = pd.DataFrame(name).transpose()  
            name.columns = [n+"_".join(x) for x in name.columns.ravel()]  
            super_df[name.columns.values] = name 
        #
        for n in [['title', 'type', 'world'],['title', 'world'],['title', 'type'], ['event_code', 'event_id'], ['event_id', 'world'], ['event_code', 'world'],['title', 'event_code'],['title', 'event_id'], ['type', 'event_id'], ['type', 'event_code']]:
            name = ranged_user.groupby(n).agg(aggregations).mean()  
            name = pd.DataFrame(name).transpose()  
            name.columns = [''.join(n)+"_".join(str(x)) for x in name.columns.ravel()]  
            super_df[name.columns.values] = name 
        #
        super_df['timestamp_diff_mean'] = ranged_user['timestamp_diff'].mean()
        #
        ############### another method -Done!- #################
        
        
        ############## event data generation starts ##################
        user_ev_data = pd.io.json.json_normalize(ranged_user.event_data.apply(json.loads))
        user_ev_data = pd.DataFrame(user_ev_data)
        col_list = user_ev_data.columns.values 
        for c in col_list:
            l_list = user_ev_data[c].values.tolist()
            arr = []
            for l in l_list:
                if isinstance(l, list):
                    l = ''.join(str(e) for e in l)
                arr.append(l)
            user_ev_data[c] = arr
            ev_da_list = user_ev_data[c].value_counts().values
            #
            super_df['ev_data_mean_'+c] = sum(ev_da_list)/len(ev_da_list) 
            super_df['ev_data_count_'+c] = len(ev_da_list) 
            super_df['ev_data_sum_'+c] = sum(ev_da_list) 
            super_df['ev_data_max_'+c] = np.amax(ev_da_list) 
            super_df['ev_data_min_'+c] = np.amin(ev_da_list) 
        ############## event data generaton - ends - #################
                
        #append the row
        tsg_df=tsg_df.append(super_df).reset_index(drop=True)
        #tsg_df=tsg_df.dropna(axis='columns')
        tsg_df=tsg_df.fillna(0)
        #
        #print((idx + 1), ' von ', history.shape[0])
        #
        if filename == 'train':
            pd.DataFrame(tsg_df).to_csv('train_set_gen' + '.csv', encoding='utf-8', index=False) #
    print('train_set_gen - finish -')
    print(time.time() - t1)
    return tsg_df



def train_set_ev_data_gen(train_raw, history, filename):  
    t1 = time.time()
    print('train_set_gen - starting... -')
    tsg_df = pd.DataFrame(columns=['timestamp_diff'])
    print('userData reading...')
    userData = train_raw.groupby(['installation_id']).apply(lambda x: x.reset_index(drop=True))
    print('userData finish')
    #
    for idx, user_history_value in tqdm(enumerate(np.array(history.values)), total=history.shape[0]): 
        if filename == 'test':
            user_id = user_history_value[0]
            game_session = user_history_value[2]
        else:
            user_id = user_history_value[5]
            game_session = user_history_value[2]
        #
        user = userData.loc[user_id] #get the first action 
        user = user.set_index(pd.DatetimeIndex(user['timestamp']))
        #start and end date to get users dataframe before the assessment
        start_date = user.index[0] #get the first action timestamp
        end_date = (user.loc[((user.game_session==game_session)&(user.event_count==1)&(user.event_code==2000)&(user.game_time==0))]['timestamp'])[0]
        #get the users dataframe before the assessment
        ranged_user = user[(user.index >= start_date) & (user.index <= end_date)]
        ranged_user['timestamp_diff'] = (end_date - start_date).total_seconds()
        ranged_user = ranged_user.reset_index(drop=True)
        #
        super_df = pd.DataFrame(data=[ranged_user['timestamp_diff'].mean()], columns=['timestamp_diff_mean'])
        #
        user_ev_data = pd.io.json.json_normalize(ranged_user.event_data.apply(json.loads))
        user_ev_data = pd.DataFrame(user_ev_data)
        col_list = user_ev_data.columns.values 
        #
        for c in col_list:
            l_list = user_ev_data[c].values.tolist()
            arr = []
            for l in l_list:
                if isinstance(l, list):
                    l = ''.join(str(e) for e in l)
                arr.append(l)
            user_ev_data[c] = arr
            #print(user_ev_data[c].values.tolist())
            ev_da_list = user_ev_data[c].value_counts().values
            #
            super_df['ev_data_mean_'+c] = sum(ev_da_list)/len(ev_da_list) 
            super_df['ev_data_count_'+c] = len(ev_da_list) 
            super_df['ev_data_sum_'+c] = sum(ev_da_list) 
            super_df['ev_data_max_'+c] = np.amax(ev_da_list) 
            super_df['ev_data_min_'+c] = np.amin(ev_da_list) 
         
        ############### another method -Done!- #################
        #append the row
        tsg_df=tsg_df.append(super_df).reset_index(drop=True)
        #tsg_df=tsg_df.dropna(axis='columns')
        tsg_df=tsg_df.fillna(0)
        #
        #print((idx + 1), ' von ', history.shape[0])
        #
        if filename == 'train':
            pd.DataFrame(tsg_df).to_csv('train_set_event_data' + '.csv', encoding='utf-8', index=False) #
        #
    print(tsg_df.columns.values)
    print(tsg_df)
    print('train_set_gen - finish -')
    print(time.time() - t1)
    return tsg_df


def train_parallel_feat_gen(x):
    return train_set_gen(train_raw, x,'train')

def test_parallel_feat_gen(x):
    return train_set_gen(train_raw,x,'test')


# In[ ]:



##RUN ASSEMENTS METHODS 
####################################################### 
#history_old = add_results(train_raw.groupby(['game_session']).apply(lambda x: x.reset_index()), 'history')#history data set
history_old = pd.read_csv("/kaggle/input/historiaa/history.csv")
print('history_old', history_old.shape)
#######################################################
tsg_res = pd.read_csv("/kaggle/input/train-feat-gen/train_feat_gen.csv")
#tsg_res = train_set_gen(train_raw, history_old, 'train')
#######################################################
#event_data_df = train_set_ev_data_gen(train_raw, history_old, 'train')#1090
event_data_df = pd.read_csv("/kaggle/input/event-data-df/event_data_df.csv")

#########################################################
train_raw = history_old
#########################################################
#
#tsg_res = tsg_res.drop(['game_time', 'event_count'], 1)
#
train_raw[tsg_res.columns.values] = tsg_res
train_raw[event_data_df.columns.values] = event_data_df
######################################################### 


train_raw = train_raw.dropna(0)
print('- TRAINSET - ', train_raw.shape)
#print(train_raw.head())


# In[ ]:


def add_cumsum_columns(df):
    df['accuracy_group_cumsum'] = df.groupby(['installation_id'])['accuracy_group'].apply(lambda x: x.cumsum())
    df['game_time_cumsum'] = df.groupby(['installation_id'])['game_time'].apply(lambda x: x.cumsum())
    df['event_count_cumsum'] = df.groupby(['installation_id'])['event_count'].apply(lambda x: x.cumsum())#
    return df


def add_assessment_count_cumsum(df):
    types = np.unique(df['type'].values).tolist()
    titles = np.unique(df['title'].values).tolist()
    worlds = np.unique(df['world'].values).tolist()
    hours_=[str(v) for v in np.unique(df['hours'].values).tolist()]
    dayofweek_ = np.unique(df['dayofweek'].values).tolist()
    
    for c in types:
        df['counter_'+c] = 1
        df['counter_cumsum_'+c] = df.groupby(['installation_id'])['counter_'+c].apply(lambda x: x.cumsum())
        df = df.drop(['counter_'+c],1)
    #
    return df
train_raw = add_assessment_count_cumsum(train_raw)



def drop_multi_colinear_features(df):
    print('dropping multi colinear features...')
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.995)]
    # Drop features 
    print('multi colinear features dropped!')
    print('to_drop', to_drop)
    df = df.drop(to_drop, axis=1)
    return df

###################################################
print('Remove colinear features')
#train_raw = drop_multi_colinear_features(train_raw)
###################################################


# In[ ]:



#FEATURE ENGINEERING: TRAIN SET
not_to_labels = ['accuracy_group', 'game_time', 'event_count', 'installation_id', 'event_code']  
not_to_labels += tsg_res.columns.values.tolist()
#
no_trainable_features = ['game_session', 'accuracy_group', 'timestamp', 'event_id', 'event_data', 'event_count', 'event_code', 'game_time'] #'event_id', 'event_data', 'title' 

#
le_title = LabelEncoder()
le_type = LabelEncoder()
le_world = LabelEncoder()
le_event_id = LabelEncoder()
le_event_data = LabelEncoder()
le_dayofweek = LabelEncoder()
le_game_session = LabelEncoder()
le_timestamp = LabelEncoder()
le_installation_id = LabelEncoder()
#
train_raw.loc[:, 'title'] = le_title.fit_transform(train_raw.loc[:, 'title'])
train_raw.loc[:, 'type'] = le_type.fit_transform(train_raw.loc[:, 'type'])
train_raw.loc[:, 'world'] = le_world.fit_transform(train_raw.loc[:, 'world'])
train_raw.loc[:, 'event_id'] = le_event_id.fit_transform(train_raw.loc[:, 'event_id'])
train_raw.loc[:, 'dayofweek'] = le_dayofweek.fit_transform(train_raw.loc[:, 'dayofweek'])
train_raw.loc[:, 'event_data'] = le_event_data.fit_transform(train_raw.loc[:, 'event_data'])
train_raw.loc[:, 'game_session'] = le_game_session.fit_transform(train_raw.loc[:, 'game_session'])
train_raw.loc[:, 'timestamp'] = le_timestamp.fit_transform(train_raw.loc[:, 'timestamp'])
train_raw.loc[:, 'installation_id'] = le_installation_id.fit_transform(train_raw.loc[:, 'installation_id'])

#
y = train_raw.loc[:, 'accuracy_group'].values
y=y.astype('int')

#
hm_data = train_raw.drop(['installation_id'], 1)
#
#regression data
train_data_for_regressor_after_label_encoder = train_raw.copy()
#
train_raw = train_raw.drop(no_trainable_features, 1)
#Features
x_values = train_raw.values
cols = train_raw.columns.values 


# In[ ]:


### EXPLORATORY DATA ANALYSIS (EDA)

def qwk_loss(a1, a2):
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



def plotter(x_std, y_train):
    plt.scatter(x_std[y_train==0, 0], x_std[y_train==0,1], color="red", marker='^', alpha=0.5, label='class 0')
    plt.scatter(x_std[y_train==1, 0], x_std[y_train==1,1], color="blue", marker='o', alpha=0.5, label='class 1')
    plt.scatter(x_std[y_train==2, 0], x_std[y_train==2,1], color="green", marker='x', alpha=0.5, label='class 2')
    plt.scatter(x_std[y_train==3, 0], x_std[y_train==3,1], color="darkmagenta", marker='+', alpha=0.5, label='class 3')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()
  


# In[ ]:



print('IMPORTANT FEATURES SELECTION USING RANDOM FOREST -starting...-')
  
'''
##heatmap
sns.set(font_scale=1.5)
print('Korrelationismatrix: Interesse an hohe Korrelationen mit der Zielvariable')
all_num_data=hm_data.values.astype('int')  
cm = np.corrcoef(all_num_data.T)
plt.figure(figsize = (18,10))
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',
         annot_kws={'size':12},yticklabels=hm_data.columns.values,xticklabels=hm_data.columns.values)

##data description
##Streudiagrammatrix: Grafische Zusammenfassung 
sns.set(style='whitegrid')
sns.pairplot(hm_data, height=2.5, hue="accuracy_group", palette="husl") 
'''


#FEATURE IMPORTANCE AND SELECTION WITH RANDOM FOREST
x_labels = [c for c in cols if c != 'accuracy_group']
forest = RandomForestClassifier(n_estimators=2000, random_state=0,n_jobs=-1)

forest.fit(x_values, y)
#print('Forest Training',forest.score(x_values, y))
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

feat_to_drop = []
print('Feature ranking:')
for f in range(x_values.shape[1]):
    #if f > 2:
    print("%2d) %-*s %f" % (f, 30, x_labels[indices[f]], importances[indices[f]])) #indices[f] 0.036529 bei 0.496
    if f > 35:
        feat_to_drop.append(x_labels[indices[f]])
print(feat_to_drop)
#___________________________________________________________#



'''
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_values.shape[1]), importances[indices], color="lightblue", yerr=std[indices], align="center")
plt.xticks(range(x_values.shape[1]), [x_labels[i] for i in indices], rotation=90)
plt.xlim([-1, x_values.shape[1]])
plt.tight_layout()
plt.show()
#___________________________________________________________#
'''


print('IMPORTANT FEATURES SELECTION USING RANDOM FOREST -finish-')


# In[ ]:


#IMPORTANT FEATURES 
#FEATURE SELECTION
print('selecting best features based on forest random model...')
#mfs = [300, 600, 900, 1200, None]
#mfs = [300, 400, 500, 600]
#mfs = range(80, 100, 2)
#mfs = range(88, 92, 1)
#mfs = [10, 20, 30, 40, 50]
#mfs = [2, 4, 6, 8, 10]
#mfs = [None]
mfs = [600]
#mfs = [400]
#
featies = []
for idmf,mf in enumerate(mfs):
    sfm = SelectFromModel(forest, threshold=None, max_features=mf, prefit=True)# best score: 0.491
    x_selected = sfm.transform(x_values)
    feature_idx = sfm.get_support()
    feature_names = cols[feature_idx] 
    ##print('feature_names', feature_names)
    #print('feature_idx', feature_idx)
    #feature_names = np.append(feature_names, ['installation_id']) 
    featies.append(feature_names)


# In[ ]:


# get LDA values
#x_train_std = lda_x
#x_test_std = lda_x_test

# REGRESSION
scores = []
def predict(models, X_test, averaging: str = 'usual'):
    full_prediction = np.zeros((X_test.shape[0], 1))
    for i in range(len(models)):
        X_t = X_test.copy()
        y_pred = models[i].predict(X_t).reshape(-1, full_prediction.shape[1])
        if full_prediction.shape[0] != len(y_pred):
            full_prediction = np.zeros((y_pred.shape[0], 1))
        if averaging == 'usual':
            full_prediction += y_pred
        elif averaging == 'rank':
            full_prediction += pd.Series(y_pred).rank().values
    return full_prediction / len(models)


def get_pred_classes(preds):    
    coefficients = [1.12232214, 1.73925866, 2.22506454]
    preds[preds <= coefficients[0]] = 0
    preds[np.where(np.logical_and(preds > coefficients[0], preds <= coefficients[1]))] = 1
    preds[np.where(np.logical_and(preds > coefficients[1], preds <= coefficients[2]))] = 2
    preds[preds > coefficients[2]] = 3
    return preds.astype(int)
#

params = {'verbose': 0,
          'learning_rate': 0.010514633017309072,
          'metric': 'rmse',
          'bagging_freq': 3,
          'boosting_type': 'gbdt',
          'eval_metric': 'cappa',
          'lambda_l1': 4.8999704874480745,
          'colsample_bytree': 0.4236269531042225,
          'early_stopping_rounds': 100,
          'lambda_l2': 0.054084652510602016,
          'bagging_fraction': 0.7931423220563563,
          'n_jobs': -1,
          'n_estimators': 5000,
          'objective': 'regression',
          'seed': 42,
          'num_leaves':5, 
          'max_depth':3,
          'class_weight':'balanced',
          'learning_rate':.005}
#

qwk_means = []
for feat_names in featies:
    #SPLITTING
    categorical_features = [idx for idx,col in enumerate(feat_names) if col in ['title']]
    #
    x_regr_df = train_data_for_regressor_after_label_encoder[feat_names]
    print('Distribution of classes in training set')
    y_df = pd.DataFrame(data=y, columns=['accuracy_group'])
    y_df.hist()
    #
    x_train, x_test, y_train, y_test = train_test_split(x_regr_df, y, test_size=0.3, random_state=0)
    #train and validation set
    groups_train = x_train[['installation_id']]
    x_train = x_train.drop(['installation_id'], 1)
    #the test set
    groups_test = x_test[['installation_id']]
    x_test = x_test.drop(['installation_id'], 1)
    #
    ##STANDARDISIEREN
    stdsc = StandardScaler(with_mean=False)
    ##stdsc = StandardScaler()
    x_train_std = (stdsc.fit_transform(x_train))
    x_test_std = (stdsc.transform(x_test))
    #ONE-HOT-ENCODE wird in dem lightgbm-Modell geregelt
    #
    plotter(x_train_std, y_train)
    #
    # CROSS VALIDATION
    folds = GroupKFold(n_splits=8)
    model_from_folds = []
    y_pred_from_folds = []
    cappas = []
    i=0
    for train_index, test_index in tqdm(folds.split(x_train_std, y_train, groups_train)):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train_std_fold, X_test_std_fold = x_train_std[train_index], x_train_std[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        #
        plotter(X_train_std_fold, y_train_fold)
        #
        print('Distribution of classes in a training set per folder')
        y_df = pd.DataFrame(data=y_train_fold, columns=['accuracy_group'])
        y_df.hist()
        print(y_df['accuracy_group'].value_counts())
        #
        print(X_train_std_fold, X_test_std_fold, y_train_fold, y_test_fold)
        # train and predict
        train_data = lightgbm.Dataset(X_train_std_fold, label=y_train_fold, categorical_feature=categorical_features)
        valid_data = lightgbm.Dataset(X_test_std_fold, label=y_test_fold, categorical_feature=categorical_features)
        # train the model with the train folder set and validate with the validation set
        model = lightgbm.train(params,train_data, valid_sets=valid_data, num_boost_round=5000, early_stopping_rounds=1000, categorical_feature=categorical_features)
        #test the model with the test set
        y_pred_result = model.predict(X_test_std_fold).reshape(-1, 1)
        y_pred_from_folds.append(y_pred_result)
        #
        print('Distribution of prediction classes per folder')
        y_df = pd.DataFrame(data=y_train_fold, columns=['accuracy_group'])
        y_df.hist()
        print(y_df['accuracy_group'].value_counts())
        #
        cappa = cohen_kappa_score(y_test_fold, get_pred_classes(y_pred_result), labels=None, weights='quadratic', sample_weight=None)
        cappas.append(cappa)
        print('index:', i)
        print('quadric cohen_kappa_score: %.3f' % cappa)
        #
        i += 1
        model_from_folds.append(model)
        #
        
    #
    qwk_mean = (sum(cappas)/len(cappas))
    qwk_means.append(qwk_mean)
    print('index:', idmf, 'qwk_mean', qwk_mean)
#
print('qwk_means:', qwk_means)


feature_names = [ x for x in feature_names if x != 'installation_id']


# In[ ]:


def parallel_process_func(chunk, q):
    pur_t = time.time()
    n_df = pd.DataFrame(columns=['event_count_cumsum'])
    q.put(add_results(chunk, 'test'))
    print(time.time()-pur_t)


# In[ ]:


print('function area')
def multiprocess(big_array, process_n):
    q = Queue()
    process_list = [Process(target=parallel_process_func, args=(chunk,q,)) for chunk in np.array_split(big_array, (process_n))]
    [p.start() for p in process_list]
    [results_mp.append(q.get()) for r in process_list]
    [p.join() for p in process_list]


# In[ ]:


print('reading aggr_test_set...')
aggr_test_set = pred_raw.groupby(['installation_id', 'game_session']).apply(lambda x: x.reset_index(drop=True))
print('aggr_test_set read')


# In[ ]:


print('RUN multi process feature generation')
results_mp = []
multiprocess(aggr_test_set, cpu_n)


# In[ ]:


print('mergin...')
print('1. results_mp', results_mp[0].shape)
print('2. results_mp', results_mp[1].shape)
print('3. results_mp', results_mp[2].shape)
print('4. results_mp', results_mp[3].shape)
#
df_concat_res = pd.concat([v for v in results_mp if (v.empty) is False], sort=True)
print(df_concat_res.shape)
print('dataframes merged')
#
dist_cols = ['game_session', 'installation_id', 'timestamp', 'counter_cumsum_Assessment']
cumsum_df = df_concat_res[dist_cols]
cumsum_df[['counter_cumsum_Assessment']].astype(float).hist()
#


# In[ ]:


print('feature engineering for test data - starting... -')  

#FEATURE ENGINEERING: PRED SET
######################################################### 
test_user_history = pred_raw
#test_user_to_predict
test_user_history.sort_values(['installation_id', 'timestamp'], inplace=True)
pred_user_raw = test_user_history.groupby("installation_id").last().reset_index()
last_ids = pred_user_raw['installation_id']
#
print(pred_user_raw)
tsg_res = train_set_gen(pred_raw, pred_user_raw, 'test')
#

######################################################### 
pred_user_raw[tsg_res.columns.values] = tsg_res
#_______________________________________________________________________________________________#

pred_user_raw = pred_user_raw.fillna(0)

print(pred_user_raw.head())
print(pred_user_raw.shape)
print('feature engineering for test data - finish -')


# In[ ]:


pred_user_raw_copy = pred_user_raw.copy()


# In[ ]:


cumsum_cols = ['counter_cumsum_Assessment']


# In[ ]:


cumsum_df_copy = cumsum_df.copy()
cum_ersatz = pd.DataFrame(columns=cumsum_cols, data=[0])
#
#get the last assessemtns
cumsum_df_copy.sort_values(['installation_id', 'timestamp'], inplace=True)
cumsum_df_copy_last = cumsum_df_copy.groupby("installation_id").last().reset_index()
#
matched_list = [cumsum_df_copy_last.loc[cumsum_df_copy_last['installation_id']==v, cumsum_cols] if v in 
                np.unique(cumsum_df_copy_last['installation_id'].values.tolist()) else cum_ersatz for v in pred_user_raw['installation_id'].values.tolist()]
matched_list = pd.concat(matched_list).reset_index(drop=True)
matched_list['counter_cumsum_Assessment'] += 1
#
pred_user_raw = pred_user_raw_copy
pred_user_raw[cumsum_cols] = matched_list
#
print(matched_list)
matched_list[cumsum_cols].astype(float).hist()


# In[ ]:


pred_user_raw[cumsum_cols].hist()


# In[ ]:


#PREDICTION
#########################################################
print('labeled test data features - started -')
print('pred_raw-columns:', pred_user_raw.columns.values) 
#label: string to int
pred_user_raw.loc[:, 'title'] = le_title.transform(pred_user_raw.loc[:, 'title'])
pred_user_raw.loc[:, 'type'] = le_type.transform(pred_user_raw.loc[:, 'type'])
pred_user_raw.loc[:, 'world'] = le_world.transform(pred_user_raw.loc[:, 'world'])
#pred_user_raw.loc[:, 'event_id'] = le_event_id.transform(pred_user_raw.loc[:, 'event_id'])
pred_user_raw.loc[:, 'dayofweek'] = le_dayofweek.transform(pred_user_raw.loc[:, 'dayofweek'])
#pred_user_raw.loc[:, 'game_session'] = le_game_session.transform(pred_user_raw.loc[:, 'game_session'])
#pred_user_raw.loc[:, 'timestamp'] = le_timestamp.transform(pred_user_raw.loc[:, 'timestamp'])
#pred_user_raw.loc[:, 'event_data'] = le_event_data.transform(pred_user_raw.loc[:, 'event_data'])
#
pred_user_raw = pred_user_raw.drop([v for v in no_trainable_features if v != 'accuracy_group'], 1)
#


# In[ ]:


#LIGHTGBM
print('LIGHTGBM is starting...')
#
#STANDARDISIEREN
print(pred_user_raw[feature_names])
stdsc = StandardScaler(with_mean=False)
pred_x_std = stdsc.fit_transform(pred_user_raw[feature_names]) 
#
preds = predict(model_from_folds, pred_x_std)
#
resultList = get_pred_classes(preds)
resultList = [v[0] for v in resultList]

print('length', len(resultList))


#SUBMISSION
print(last_ids[:500].tolist())
print(resultList[:1000])
print('predict the future - finish -')
#
sub = pd.DataFrame({'installation_id':last_ids, 'accuracy_group': resultList})
sub.to_csv('submission.csv', encoding='utf-8', index=False) #
print('Submission file created!')
#
print('LIGHTGBM ends')


# In[ ]:


sub['accuracy_group'].hist()
print(sub['accuracy_group'].value_counts())

