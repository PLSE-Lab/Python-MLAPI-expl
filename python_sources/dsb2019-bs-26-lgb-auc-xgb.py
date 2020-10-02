#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from random import seed
from random import randint
# seed random number generator
seed(44)

from tqdm import tqdm
import pickle
from numba import jit 
from scipy import stats
import json
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization
import lightgbm as lgb
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

import gc
import os
import psutil


import warnings
warnings.filterwarnings("ignore")


def rmse (y_true, y_pred):
    return np.sqrt ( mean_squared_error (y_true, y_pred) )

def eval_qwk_lgb_regr(y_pred, accuracy_groups):
    """
    Fast cappa eval function for lgb.
    """
    dist = Counter(accuracy_groups)
    for k in dist:
        dist[k] /= len(accuracy_groups)
    
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

    y_pred = np.array(list(map(classify, y_pred)))
    
    return y_pred


@jit
def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
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

def round_prediction (x,a,b,c):
    x = np.where(x < a, 0, x)  
    x = np.where((a < x) & (x <= a + b), 1, x)  
    x = np.where((a + b < x) & (x <= a + b + c), 2, x)  
    x = np.where((a + b + c < x) , 3, x)  
    return x




# In[ ]:


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
    #start = time.time()

    print("Start encoding data")
    # encode title
    train['title_event_code'] = sorted(list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code'])))
    test['title_event_code'] = sorted(list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code'])))
    all_title_event_code = sorted(list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique())))

    train['type_world'] = sorted(list(map(lambda x, y: str(x) + '_' + str(y), train['type'], train['world'])))
    test['type_world'] = sorted(list(map(lambda x, y: str(x) + '_' + str(y), test['type'], test['world'])))
    all_type_world = sorted(list(set(train["type_world"].unique()).union(test["type_world"].unique())))

    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = sorted(list(set(train['title'].unique()).union(set(test['title'].unique()))))
    
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = sorted(list(set(train['event_code'].unique()).union(set(test['event_code'].unique()))))
    list_of_event_id = sorted(list(set(train['event_id'].unique()).union(set(test['event_id'].unique()))))
    
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = sorted(list(set(train['world'].unique()).union(set(test['world'].unique()))))
    
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = sorted(list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(
        set(test[test['type'] == 'Assessment']['title'].value_counts().index))))
    
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    #print("End encoding data, time - ", time.time() - start)


    event_data = {}
    event_data["train_labels"] = train_labels
    event_data["win_code"] = win_code
    event_data["list_of_user_activities"] = list_of_user_activities
    event_data["list_of_event_code"] = list_of_event_code
    event_data["activities_labels"] = activities_labels
    event_data["assess_titles"] = assess_titles
    event_data["list_of_event_id"] = list_of_event_id
    event_data["all_title_event_code"] = all_title_event_code
    event_data["activities_map"] = activities_map
    event_data["all_type_world"] = all_type_world

    return train, test, event_data

def get_all_features(feature_dict, ac_data):
    if len(ac_data['durations']) > 0:
        feature_dict['installation_duration_mean'] = np.mean(ac_data['durations'])
        feature_dict['installation_duration_sum'] = np.sum(ac_data['durations'])
    else:
        feature_dict['installation_duration_mean'] = 0
        feature_dict['installation_duration_sum'] = 0

    return feature_dict

def get_data(user_sample, event_data, test_set):
    '''
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_assesment = {}

    last_activity = 0

    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}

    assess_4020_acc_dict = {'Cauldron Filler (Assessment)_4020_accuracy': 0,
                            'Mushroom Sorter (Assessment)_4020_accuracy': 0,
                            'Bird Measurer (Assessment)_4020_accuracy': 0,
                            'Chest Sorter (Assessment)_4020_accuracy': 0}

    game_time_dict = {'Clip_gametime': 0, 'Game_gametime': 0,
                      'Activity_gametime': 0, 'Assessment_gametime': 0}

    last_session_time_sec = 0
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0

    # Newly added features
    accumulated_game_miss = 0
    Cauldron_Filler_4025 = 0
    mean_game_round = 0
    mean_game_duration = 0
    mean_game_level = 0
    Assessment_mean_event_count = 0
    Game_mean_event_count = 0
    Activity_mean_event_count = 0
    chest_assessment_uncorrect_sum = 0

    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    durations_game = []
    durations_activity = []
    last_accuracy_title = {'acc_' + title: -1 for title in event_data["assess_titles"]}
    last_game_time_title = {'lgt_' + title: 0 for title in event_data["assess_titles"]}
    ac_game_time_title = {'agt_' + title: 0 for title in event_data["assess_titles"]}
    ac_true_attempts_title = {'ata_' + title: 0 for title in event_data["assess_titles"]}
    ac_false_attempts_title = {'afa_' + title: 0 for title in event_data["assess_titles"]}
    event_code_count: dict[str, int] = {ev: 0 for ev in event_data["list_of_event_code"]}
    event_code_proc_count = {str(ev) + "_proc" : 0. for ev in event_data["list_of_event_code"]}
    event_id_count: dict[str, int] = {eve: 0 for eve in event_data["list_of_event_id"]}
    title_count: dict[str, int] = {eve: 0 for eve in event_data["activities_labels"].values()}
    title_event_code_count: dict[str, int] = {t_eve: 0 for t_eve in event_data["all_title_event_code"]}
    type_world_count: dict[str, int] = {w_eve: 0 for w_eve in event_data["all_type_world"]}
    session_count = 0
    
    Activity_game_durations = [] 

    last_Game_Features = {}
    last_game_session_correct_true = np.nan
    last_game_session_correct_false = np.nan

    last_2_game_session_correct_true = np.nan
    last_2_game_session_correct_false = np.nan

    acc_game_session_correct_true = 0
    acc_game_session_correct_false = 0

    last_Assessment_Features = {}

    
    last_Activity_Features = {}
    
    session_type_story = []
    
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = event_data["activities_labels"][session_title]
        game_session = session['game_session'].iloc[0]
        
        session_world = session['world'].iloc[0]
        
        timestamp = session['timestamp'].iloc[0]
        
        if session_type == "Activity":
            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1]) / 2.0
            durations_activity.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            
            last_Activity_Features = {
                "last_activity_event_count":session["event_count"].values[-1],
                "last_activity_event_count_nunique" : session["event_count"].nunique(),
                "last_activity_timestamp": timestamp,
                "last_activity_world": session_world, 
                "last_activity_session_title": session_title
            }
            
        if session_type == "Game":
            
            Game_mean_event_count = (Game_mean_event_count + session['event_count'].iloc[-1]) / 2.0

            game_s = session[session.event_code == 2030]
            misses_cnt = cnt_miss(game_s)
            accumulated_game_miss += misses_cnt

            try:
                game_round = json.loads(session['event_data'].iloc[-1])["round"]
                mean_game_round = (mean_game_round + game_round) / 2.0
            except:
                pass

            try:
                game_duration = json.loads(session['event_data'].iloc[-1])["duration"]
                mean_game_duration = (mean_game_duration + game_duration) / 2.0
            except:
                pass

            try:
                game_level = json.loads(session['event_data'].iloc[-1])["level"]
                mean_game_level = (mean_game_level + game_level) / 2.0
            except:
                pass

            
            game_session_correct =session [ session["event_data"].map(lambda x: '"correct"' in x ) ]["event_data"].map(lambda x: json.loads(x)["correct"])
            last_2_game_session_correct_true = last_game_session_correct_true
            last_2_game_session_correct_false = last_game_session_correct_false
            
            last_game_session_correct_true = game_session_correct.sum()
            last_game_session_correct_false = game_session_correct.shape[0] - last_game_session_correct_true
            acc_game_session_correct_true += last_game_session_correct_true
            acc_game_session_correct_false += last_game_session_correct_false
        
            durations_game.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            
            last_Game_Features = {
                "last_game_event_count":session["event_count"].values[-1],
                "last_game_event_count_nunique" : session["event_count"].nunique(),
                "last_game_timestamp": timestamp,
                "last_game_world": session_world,
                "last_game_session_title": session_title
            }

            
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session) > 1):
            
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {event_data["win_code"][session_title]}')
            
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            
            # copy a dict to use as feature template, it's initialized with some itens:
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(title_count.copy())
            features.update(game_time_dict.copy())
            features.update(event_id_count.copy())
            features.update(title_event_code_count.copy())
            features.update(assess_4020_acc_dict.copy())
            features.update(type_world_count.copy())
            features.update(last_game_time_title.copy())
            features.update(ac_game_time_title.copy())
            features.update(ac_true_attempts_title.copy())
            features.update(ac_false_attempts_title.copy())

            features.update(event_code_proc_count.copy())
            features['installation_session_count'] = session_count
            features['game_session'] =  game_session
            features['timestamp'] =  timestamp
            features['accumulated_game_miss'] = accumulated_game_miss
            features['mean_game_round'] = mean_game_round
            features['mean_game_duration'] = mean_game_duration
            features['mean_game_level'] = mean_game_level
            features['Assessment_mean_event_count'] = Assessment_mean_event_count
            features['Game_mean_event_count'] = Game_mean_event_count
            features['Activity_mean_event_count'] = Activity_mean_event_count
            features['chest_assessment_uncorrect_sum'] = chest_assessment_uncorrect_sum

            variety_features = [('var_event_code', event_code_count),
                                ('var_event_id', event_id_count),
                                ('var_title', title_count),
                                ('var_title_event_code', title_event_code_count),
                                ('var_type_world', type_world_count)]

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

            # ----------------------------------------------
            ac_true_attempts_title['ata_' + session_title_text] += true_attempts
            ac_false_attempts_title['afa_' + session_title_text] += false_attempts

            last_game_time_title['lgt_' + session_title_text] = session['game_time'].iloc[-1]
            ac_game_time_title['agt_' + session_title_text] += session['game_time'].iloc[-1]
            # ----------------------------------------------
            
            
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = np.nan
                features['duration_std'] = np.nan
                features['last_duration'] = np.nan
                features['last-2_duration'] = np.nan
                features['duration_max'] = np.nan
                #features['duration_min'] = np.nan
            else:
                features['duration_mean'] = np.mean(durations)
                features['duration_std'] = np.std(durations)
                features['last_duration'] = durations[-1]
                if len(durations)>1:
                    features['last-2_duration'] = durations[-2]
                else:
                    features['last-2_duration'] = np.nan
                features['duration_max'] = np.max(durations)
                #features['duration_min'] = np.min(durations)
            
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

            if durations_game == []:
                features['duration_game_mean'] = np.nan
                features['duration_game_std'] = np.nan
                features['game_last_duration'] = np.nan
                features['game_last-2_duration'] = np.nan
                features['game_max_duration'] = np.nan
                #features['game_min_duration'] = np.nan
            else:
                features['duration_game_mean'] = np.mean(durations_game)
                features['duration_game_std'] = np.std(durations_game)
                features['game_last_duration'] = durations_game[-1]
                if len(durations_game)>1:
                    features['game_last-2_duration'] = durations_game[-2]
                else:
                    features['game_last-2_duration'] = np.nan
                features['game_max_duration'] = np.max(durations_game)
                #features['game_min_duration'] = np.min(durations_game)

            if last_Game_Features  == {}:
                features["last_game_event_count"] = np.nan
                features["last_game_event_count_nunique"] = np.nan
                features["last_game_timestamp"] = np.nan
                features["last_game_world_is_the_same"] = np.nan
                features["last_game_session_title"] = np.nan
            else:
                features["last_game_event_count"] =  last_Game_Features["last_game_event_count"]
                features["last_game_event_count_nunique"] =  last_Game_Features["last_game_event_count_nunique"]
                features["last_game_timestamp"] = last_Game_Features["last_game_timestamp"]
                features["last_game_world_is_the_same"] = int(last_Game_Features["last_game_world"] == session_world)
                features["last_game_session_title"] = last_Game_Features["last_game_session_title"]

            if last_Assessment_Features  == {}:
                features["last_assessment_event_count"] = np.nan
                features["last_assessment_event_count_nunique"] = np.nan
                features["last_assessment_timestamp"] = np.nan
                features["last_assessment_world_is_the_same"] = np.nan
                features["last_assessment_title_is_the_same"] = np.nan
                features["last_assessment_accuracy_group"] = np.nan
                features["last_assessment_session_title"] = np.nan
                
            else:
                features["last_assessment_event_count"] =  last_Assessment_Features["last_assessment_event_count"]
                features["last_assessment_event_count_nunique"] =  last_Assessment_Features["last_assessment_event_count_nunique"]
                features["last_assessment_timestamp"] = last_Assessment_Features["last_assessment_timestamp"]
                features["last_assessment_world_is_the_same"] = int(last_Assessment_Features["last_assessment_world"] == session_world)
                features["last_assessment_title_is_the_same"] = int(last_Assessment_Features["last_assessment_title"] == session_title)
                features["last_assessment_accuracy_group"] = last_Assessment_Features["last_assessment_accuracy_group"]
                features["last_assessment_session_title"] = last_Assessment_Features["last_assessment_title"]

                
            if last_Activity_Features  == {}:
                features["last_activity_event_count"] = np.nan
                features["last_activity_event_count_nunique"] = np.nan
                features["last_activity_timestamp"] = np.nan
                features["last_activity_world_is_the_same"] = np.nan
                features["last_activity_session_title"] = np.nan
            else:
                features["last_activity_event_count"] =  last_Activity_Features["last_activity_event_count"]
                features["last_activity_event_count_nunique"] =  last_Activity_Features["last_activity_event_count_nunique"]
                features["last_activity_timestamp"] = last_Activity_Features["last_activity_timestamp"]
                features["last_activity_world_is_the_same"] = int(last_Activity_Features["last_activity_world"] == session_world)
                features["last_activity_session_title"] = last_Activity_Features["last_activity_session_title"]
                
                
            if durations_activity == []:
                #features['duration_activity_mean'] = np.nan
                #features['duration_activity_std'] = np.nan
                #features['activity_last_duration'] = np.nan
                #features['activity_last-2_duration'] = np.nan
                #features['activity_max_duration'] = np.nan
                features['activity_min_duration'] = np.nan
            else:
                #features['duration_activity_mean'] = np.mean(durations_activity)
                #features['duration_activity_std'] = np.std(durations_activity)
                #features['activity_last_duration'] = durations_activity[-1]
                #if len(durations_activity)>1:
                #    features['activity_last-2_duration'] = durations_activity[-2]
                #else:
                #    features['activity_last-2_duration'] = np.nan
                #features['activity_max_duration'] = np.max(durations_activity)
                features['activity_min_duration'] = np.min(durations_activity)

            # the accuracy is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0
            # --------------------------
            features['Cauldron_Filler_4025'] = Cauldron_Filler_4025 / counter if counter > 0 else 0

            Assess_4025 = session[(session.event_code == 4025) & (session.title == 'Cauldron Filler (Assessment)')]
            true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()
            false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()

            cau_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (true_attempts_ + false_attempts_) != 0 else 0
            Cauldron_Filler_4025 += cau_assess_accuracy_

            chest_assessment_uncorrect_sum += len(session[session.event_id == "df4fe8b6"])

            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1]) / 2.0
            # ----------------------------
            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
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
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            features["acc_current_assessment"] = features["acc_"+session_title_text]
            features["lgt_current_assessment"] = features["lgt_"+session_title_text]
            features["agt_current_assessment"] = features["agt_"+session_title_text]
            features["ata_current_assessment"] = features["ata_"+session_title_text]
            features["afa_current_assessment"] = features["afa_"+session_title_text]
            
            features["current_4020_accuracy"] = features.get(session_title_text + "_4020_accuracy",-1)
  
            #for _event in ['2000','2010','2020','2030','3010','3020','3021','3110','3120',
            #              '3121','4020','4030','4035','4040','4070','4080','4090','4100']:
            #    features["current_assessment_" + _event] = features.get(session_title_text + "_" + _event,0)

            
            features["last_game_session_correct_true"] = last_game_session_correct_true
            features["last_game_session_correct_false"] = last_game_session_correct_false
            features["last_2_game_session_correct_true"] = last_2_game_session_correct_true
            features["last_2_game_session_correct_false"] = last_2_game_session_correct_false
            
            features["acc_game_session_correct_true"] = acc_game_session_correct_true
            features["acc_game_session_correct_false"] = acc_game_session_correct_false
            
            session_type_story_length = len(session_type_story)
            features["session_type_story_length"] = session_type_story_length
            features["session_type_story_count_clip"] =session_type_story.count('Clip')
            features["session_type_story_count_game"] =session_type_story.count('Game')
            features["session_type_story_count_activity"] =session_type_story.count('Activity')
            features["session_type_story_count_assessment"] =session_type_story.count('Assessment')
            
            session_type_story = []
            
            last_Assessment_Features = {
                "last_assessment_event_count":session["event_count"].values[-1],
                "last_assessment_event_count_nunique" : session["event_count"].nunique(),
                "last_assessment_timestamp": timestamp,
                "last_assessment_world": session_world,
                "last_assessment_title": session_title,
                "last_assessment_accuracy_group": features['accuracy_group'] 
            }


            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                last_assesment = features.copy()

            if true_attempts + false_attempts > 0:
                all_assessments.append(features)

                
            counter += 1


        session_count += 1

        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
            num_of_session_count = Counter(session[col])
            for k in num_of_session_count.keys():
                x = k
                if col == 'title':
                    x = event_data["activities_labels"][k]
                counter[x] += num_of_session_count[k]
            return counter

        def update_proc(count: dict):
            res = {}
            for k, val in count.items():
                res[str(k) + "_proc"] = (float(val) * 100.0) / accumulated_actions
            return res

        event_code_count = update_counters(event_code_count, "event_code")


        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')
        type_world_count = update_counters(type_world_count, 'type_world')

        assess_4020_acc_dict = get_4020_acc(session, assess_4020_acc_dict, event_data)
        game_time_dict[session_type + '_gametime'] = (game_time_dict[session_type + '_gametime'] + (
                    session['game_time'].iloc[-1] / 1000.0)) / 2.0

        
        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        event_code_proc_count = update_proc(event_code_count)

        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type
       
        session_type_story.append ( session_type ) 

    # if it't the test_set, only the last assessment must be predicted, the previous goes to the dataset
    if test_set:
        return last_assesment, all_assessments
    # in the train_set, all assessments goes to the dataset
    return all_assessments


def cnt_miss(df):
    cnt = 0
    for e in range(len(df)):
        x = df['event_data'].iloc[e]
        y = json.loads(x)['misses']
        cnt += y
    return cnt

def get_4020_acc(df, counter_dict, event_data):
    for e in ['Cauldron Filler (Assessment)', 'Bird Measurer (Assessment)',
              'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)']:
        Assess_4020 = df[(df.event_code == 4020) & (df.title == event_data["activities_map"][e])]
        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()
        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()

        measure_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (
                                                                                                      true_attempts_ + false_attempts_) != 0 else 0
        counter_dict[e + "_4020_accuracy"] += (counter_dict[e + "_4020_accuracy"] + measure_assess_accuracy_) / 2.0

    return counter_dict

def get_users_data(users_list, return_dict,  event_data, test_set):
    if test_set:
        for user in users_list:
            return_dict.append(get_data(user, event_data, test_set))
    else:
        answer = []
        for user in users_list:
            answer += get_data(user, event_data, test_set)
        return_dict += answer

        
def get_train_and_test_single_proc(train, test, event_data, load_train = False):

    if load_train :
        reduce_train = pd.read_pickle (PROCESSED_TRAIN_PATH)
    else:    
        compiled_train = []
        for ins_id, user_sample in tqdm(train.groupby('installation_id', sort=False), total=17000):
            compiled_train += get_data(user_sample, event_data, False)
        reduce_train = pd.DataFrame(compiled_train)
    
    compiled_test = []
    compiled_test_to_train = []
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):
        test_data = get_data(user_sample, event_data, True)
        compiled_test.append(test_data[0])
        compiled_test_to_train += test_data[1]


    reduce_test = pd.DataFrame(compiled_test)
    reduce_test_to_train = pd.DataFrame(compiled_test_to_train)

    return reduce_train, reduce_test, reduce_test_to_train        


# In[ ]:


def remove_correlated_features(reduce_train,features ):
    counter = 0
    to_remove = []
    for feat_a in features:
        for feat_b in features:
            if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
                c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]
                if c > 0.9999:
                    counter += 1
                    to_remove.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
    return to_remove


# In[ ]:


def post_process (df):
    df["hour"]=df["timestamp"].dt.hour
    df["dayofyear"]=df["timestamp"].dt.dayofyear
    df["dayofweek"]=df["timestamp"].dt.dayofweek
    df['sin_hour'] = np.sin(2*np.pi*df["hour"]/24)
    df['cos_hour'] = np.cos(2*np.pi*df["hour"]/24)
    df = df.drop (["hour"], axis = 1)  
    
    df["timestamp"]=df["timestamp"].astype(int)
    df["last_game_timestamp"]=(df["timestamp"] - df["last_game_timestamp"].astype(int)) // 1e6
    df["last_assessment_timestamp"]=(df["timestamp"] - df["last_assessment_timestamp"].astype(int)) // 1e6
    df["last_activity_timestamp"]=(df["timestamp"] - df["last_activity_timestamp"].astype(int)) // 1e6
    
    df["last_game_session_correct_accuracy"] = df["last_game_session_correct_true"]/(df["last_game_session_correct_true"]+df["last_game_session_correct_false"])
    df["last_2_game_session_correct_accuracy"] = df["last_2_game_session_correct_true"]/(df["last_2_game_session_correct_true"]+df["last_2_game_session_correct_false"])
    df["last_game_session_correct_accuracy*last_2_game_session_correct_accuracy"] = df["last_game_session_correct_accuracy"]*df["last_2_game_session_correct_accuracy"]
    
    df["acc_game_session_correct_accuracy"] = df["acc_game_session_correct_true"]/(df["acc_game_session_correct_true"]+df["acc_game_session_correct_false"])
    
    
    
    
    return df
    


# In[ ]:


def truncate_generator (X, y, groups):
    X = X.copy()
    X["__y__"] = y
    X["__groups__"] = groups
    state = 44
    while True:
        X = X.sample (frac=1.0, random_state=state)
        state +=1
        _X = X.drop_duplicates(["__groups__"], keep="first")
        yield _X.drop(["__y__","__groups__"],axis=1), _X["__y__"].values, _X["__groups__"].values  


def trunc_rmse_qwk_score(y,pred,groups, nsamples=5000):
    rmse_scores = np.zeros ( (nsamples, ))
    oof_cohen_scores = np.zeros ( (nsamples, ))
    
    X=pd.DataFrame()
    X["pred"] = pred
    truncate_gen = truncate_generator (X, y, groups)
    for i in range (nsamples):
        X, y, groups = truncate_gen.__next__()
        pred = X["pred"]
        rmse_scores [i]= rmse(y, pred)
        oof_cohen_scores [i]= cohen_kappa_score(y, eval_qwk_lgb_regr(pred, y), weights = 'quadratic')    
    
    return rmse_scores, oof_cohen_scores

def trunc_qwk_score(y,pred,groups, nsamples=5000):
    oof_cohen_scores = np.zeros ( (nsamples, ))
    
    X=pd.DataFrame()
    X["pred"] = pred
    truncate_gen = truncate_generator (X, y, groups)
    for i in range (nsamples):
        X, y, groups = truncate_gen.__next__()
        pred = X["pred"]
        oof_cohen_scores [i]= qwk3(pred, y)    
    
    return oof_cohen_scores

def trunc_auc_score(y,pred,groups, nsamples=5000):
    scores = np.zeros ( (nsamples, ))
    
    X=pd.DataFrame()
    X["pred"] = pred
    truncate_gen = truncate_generator (X, y, groups)
    for i in range (nsamples):
        X, y, groups = truncate_gen.__next__()
        pred = X["pred"]
        scores [i]= roc_auc_score(y,pred)    
    
    return scores


def oof_lgb (X, ids, y, groups, num_boost_round, early_stopping_rounds, params, categoricals, splits, verbose_eval):

    columns = [c for c in X.columns]
    
    oof = pd.DataFrame({"id":ids})
    y_oof = np.zeros((X.shape[0], ))
    
    models= []
    

    feature_importances = pd.DataFrame()
    feature_importances['feature'] = columns 

    
    
    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        #group_train = groups.iloc [train_index] 
        #train_truncate_gen = truncate_generator ( X_train, y_train, group_train)

        
        group_valid = groups.iloc [valid_index] 
        
        truncate_gen = truncate_generator ( X_valid, y_valid, group_valid) 
        
        X_valid_trunc, y_valid_trunc, _ = truncate_gen.__next__()
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid_trunc, label=y_valid_trunc)

        model = lgb.train(params, dtrain, num_boost_round, 
                        valid_sets = [dtrain, dvalid],
                        categorical_feature = categoricals,
                        verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds)

        y_pred = model.predict(X_valid)
        
        
        #if verbose_eval > 0:
        #   rmse_scores, oof_cohen_scores = trunc_rmse_qwk_score (y_valid.values, y_pred, group_valid.values,nsamples=5000)
        #    print (f'fold: {fold_n}, rmse: {np.median(rmse_scores)} std:{rmse_scores.std()} , oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}' )
        
        models.append(model)
 
        feature_importances[f'fold_{fold_n + 1}'] = model.feature_importance()

        y_oof[valid_index] = y_pred

        del X_train, X_valid, y_train, y_valid
        gc.collect()

    #rmse_scores, oof_cohen_scores = trunc_rmse_qwk_score (y.values, y_oof, groups.values, nsamples=5000)
    #if verbose_eval > -1:
    #    print (f'oof rmse: {np.median(rmse_scores)} std:{rmse_scores.std()} , oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')
        
    
    feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(len(splits))]].mean(axis=1)
    feature_importances = feature_importances.sort_values (by="average", ascending = False)
    feature_importances.to_csv("feature_importance.csv", index=False)
    
    return models, y_oof

def run_lgb (df_train, num_boost_round, early_stopping_rounds,  params, model_feats,categorical_features, nfolds, verbose_eval  ):

    if verbose_eval > 0:
        print(f'nfolds:{nfolds}, features:{len(model_feats)}, categorical:{len(categorical_features)}')
    
    X = df_train[model_feats]
    y = df_train["accuracy_group"]
    ids = df_train["sample"]
    groups = df_train["installation_id"]

    folds = GroupKFold(n_splits=nfolds) 

    splits = []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,groups=groups)):
        splits.append((train_index, valid_index)) 

    lgb_model, y_oof = oof_lgb (X, ids, y, groups,  num_boost_round, early_stopping_rounds, params, categoricals=categorical_features, splits=splits, verbose_eval = verbose_eval)
    
    return lgb_model, y_oof  #, rmse_scores, oof_cohen_scores

def run_lgb_ensemble (df_train, model_feats, categorical_features, num_boost_round, early_stopping_rounds,  params, nmodel,nfolds, verbose_eval ):
    y_oof = np.zeros ( (df_train.shape[0],) )
    for i in range(nmodels):
        lgb_i_models, y_i_oof  = run_lgb(df_train, num_boost_round, early_stopping_rounds, params, model_feats, categorical_features,  nfolds, verbose_eval )
        y_oof += y_i_oof 
        for n,model in enumerate(lgb_i_models):
            model.save_model(f'lgb_{i}_model_fold_{n}.txt')
        params['seed'] +=1
        params['feature_fraction_seed'] +=1
        params['bagging_seed'] +=1
        params['drop_seed'] +=1
        params['data_random_seed'] +=1            
    
    y_oof = y_oof / nmodels
    
    rmse_scores,  oof_cohen_scores  = trunc_rmse_qwk_score ( df_train["accuracy_group"].values, y_oof, df_train["installation_id"].values, nsamples=5000)
    print (f'lgb {nmodels} models rmse: {np.median(rmse_scores)} std:{rmse_scores.std()} , oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')
    
    return y_oof, rmse_scores,  oof_cohen_scores

def run_lgb_auc (df_train, y, num_boost_round, early_stopping_rounds,  params, model_feats,categorical_features, nfolds, verbose_eval  ):

    if verbose_eval > 0:
        print(f'nfolds:{nfolds}, features:{len(model_feats)}, categorical:{len(categorical_features)}')
    
    X = df_train[model_feats]

    ids = df_train["sample"]
    groups = df_train["installation_id"]

    folds = GroupKFold(n_splits=nfolds) 

    splits = []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,groups=groups)):
        splits.append((train_index, valid_index)) 

    lgb_model, y_oof = oof_lgb (X, ids, y, groups,  num_boost_round, early_stopping_rounds, params, categoricals=categorical_features, splits=splits, verbose_eval = verbose_eval)
    
    return lgb_model, y_oof  #, rmse_scores, oof_cohen_scores

def run_lgb_auc_ensemble (df_train, y, name,  model_feats, categorical_features, num_boost_round, early_stopping_rounds,  params, nmodel,nfolds, verbose_eval ):
    y_oof = np.zeros ( (df_train.shape[0],) )
    for i in range(nmodels):
        lgb_i_models, y_i_oof  = run_lgb_auc(df_train, y, num_boost_round, early_stopping_rounds, params, model_feats, categorical_features,  nfolds, verbose_eval )
        y_oof += y_i_oof 
        for n,model in enumerate(lgb_i_models):
            model.save_model(f'lgb_{name}_{i}_model_fold_{n}.txt')
        params['seed'] +=1
        params['feature_fraction_seed'] +=1
        params['bagging_seed'] +=1
        params['drop_seed'] +=1
        params['data_random_seed'] +=1            
    
    y_oof = y_oof / nmodels
    
    auc_scores  = trunc_auc_score ( y.values, y_oof, df_train["installation_id"].values, nsamples=5000)
    print (f'lgb {nmodels} models auc: {np.median(auc_scores)} std:{auc_scores.std()}')
    
    return y_oof, auc_scores


def oof_xgb (X, ids, y, groups,num_boost_round, early_stopping_rounds, params, splits, verbose_eval):
    columns = [c for c in X.columns]
    
    oof = pd.DataFrame({"id":ids})
    y_oof = np.zeros((X.shape[0], ))
    
    models= []
    score = 0
    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        group_valid = groups.iloc [valid_index] 
        
        truncate_gen = truncate_generator ( X_valid, y_valid, group_valid) 
        
        X_valid_trunc, y_valid_trunc, _ = truncate_gen.__next__()
      
    
        dtrain = xgb.DMatrix(X_train,label=y_train)
        dvalid = xgb.DMatrix(X_valid_trunc,label=y_valid_trunc)
       
        model = xgb.train(params, dtrain=dtrain, num_boost_round=num_boost_round, evals =  [(dtrain, 'train'),(dvalid, 'valid')],
                          early_stopping_rounds=early_stopping_rounds, maximize=False, verbose_eval=verbose_eval)


        dvalid = xgb.DMatrix(X_valid,label=y_valid)
        y_pred = model.predict(dvalid)
        
           
            
        models.append(model)

        y_oof[valid_index] = y_pred

        del X_train, X_valid, y_train, y_valid
        gc.collect()

    #rmse_scores, oof_cohen_scores = trunc_rmse_qwk_score (y, y_oof, groups)
    #avg_score = score/len(splits)    
    #if verbose_eval > 0:
    #    print (f'oof rmse: {rmse_scores.median()}, mean rmse: {avg_score}, cohen kappa score : {oof_cohen_scores.median()}')
    
    return models, y_oof



def run_xgb (df_train, num_boost_round, early_stopping_rounds,  params, model_feats,categorical_features, nfolds  , verbose_eval   ):

    if verbose_eval > 0:
        print(f'nfolds:{nfolds}, features:{len(model_feats)}, categorical:{len(categorical_features)}')
    
    X = df_train[model_feats].copy()
    
    X = pd.get_dummies(X, dummy_na=True, columns=categorical_features)
    
    y = df_train["accuracy_group"]
    ids = df_train["sample"]
    groups = df_train["installation_id"]

    folds = GroupKFold(n_splits=nfolds) 

    splits = []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,groups=groups)):
        splits.append((train_index, valid_index)) 

    xgb_model, y_oof = oof_xgb (X, ids, y, groups, num_boost_round, early_stopping_rounds, params,  splits=splits, verbose_eval = verbose_eval)
    
    return xgb_model, y_oof


def run_xgb_ensemble (df_train, model_feats, categorical_features, num_boost_round, early_stopping_rounds,  params, nmodel,nfolds, verbose_eval ):
    y_oof = np.zeros ( (df_train.shape[0],) )
    for i in range(nmodels):
        xgb_i_models, y_i_oof  = run_xgb(df_train, num_boost_round, early_stopping_rounds, params, model_feats, categorical_features,  nfolds, verbose_eval )
        y_oof += y_i_oof 
        for n,model in enumerate(xgb_i_models):
            model.save_model(f'xgb_{i}_model_fold_{n}.txt')
        params['seed'] +=1       
    
    y_oof = y_oof / nmodels
    
    rmse_scores,  oof_cohen_scores  = trunc_rmse_qwk_score ( df_train["accuracy_group"].values, y_oof, df_train["installation_id"].values, nsamples=5000)
    print (f'xgb {nmodels} models rmse: {np.median(rmse_scores)} std:{rmse_scores.std()} , oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')
    
    return y_oof, rmse_scores,  oof_cohen_scores


def predict_lgb (nmodel, nfolds):
    y_blend = np.zeros((test.shape[0], ))     
    
    for i in range(nmodels):
        y_test = np.zeros((test.shape[0], ))     

        for n in range(nfolds):
            model_file = f'./lgb_{i}_model_fold_{n}.txt'
            model = lgb.Booster(model_file=model_file)
            pred = model.predict( test[model_feats], num_iteration=model.best_iteration )
            y_test += pred
        
        y_blend += y_test / nfolds
    
    y_blend = y_blend / nmodels
    
    return y_blend

def predict_auc_lgb (name, nmodel, nfolds):
    y_blend = np.zeros((test.shape[0], ))     
    
    for i in range(nmodels):
        y_test = np.zeros((test.shape[0], ))     

        for n in range(nfolds):
            model_file = f'./lgb_{name}_{i}_model_fold_{n}.txt'
            model = lgb.Booster(model_file=model_file)
            pred = model.predict( test[model_feats], num_iteration=model.best_iteration )
            y_test += pred
        
        y_blend += y_test / nfolds
    
    y_blend = y_blend / nmodels
    
    return y_blend



def predict_xgb (nmodel, nfolds):
    y_blend = np.zeros((test.shape[0], ))     

    test_xgb = test[model_feats].copy()
    test_xgb = pd.get_dummies(test_xgb,  dummy_na=True, columns=categorical_features)
    
    for i in range(nmodels):
        y_test = np.zeros((test.shape[0], ))     

        for n in range(nfolds):
            model_file = f'./xgb_{i}_model_fold_{n}.txt'
            model = xgb.Booster() #init model
            model.load_model(model_file) # load data
            dtest =  xgb.DMatrix (test_xgb)
            pred = model.predict(dtest)           
            y_test += pred
        
        y_blend += y_test / nfolds
    
    y_blend = y_blend / nmodels
    
    return y_blend

def predict(sample_submission, y_pred):
    sample_submission['accuracy_group'] = y_pred
    sample_submission['accuracy_group'] = sample_submission['accuracy_group'].astype(int)
    sample_submission.to_csv('submission.csv', index = False)
    print(sample_submission['accuracy_group'].value_counts(normalize = True))
    


# In[ ]:


def calc_bound ( y_oof, groups ):

    dist = Counter(groups)
    for k in dist:
        dist[k] /= len(y_oof)

    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_oof, acum * 100)

    return bound[0], bound[1] - bound[0],  bound[2] - bound[1]  

def qwk3_optimizer ( y, y_oof, init_points = 20, n_iter = 50  ):
    pbounds = {'a': (0.0, 1.5), 'b': (0.0, 1.5), 'c': (0.0, 1.5)}

    def qwk3_opt ( x, y, a,b,c ):

        x = round_prediction ( x, a,b,c )

        return qwk3 ( x, y, max_rat=3 )


    def q (a,b,c):
        return qwk3_opt  ( y_oof, y, a,b,c )

    optimizer = BayesianOptimization(
        f=q,
        pbounds=pbounds,
        random_state=44,
    )


    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    a = optimizer.max["params"]["a"]
    b = optimizer.max["params"]["b"]
    c = optimizer.max["params"]["c"]
    t = optimizer.max["target"]
    print ( f'qwk3:{t}, a:{a}, b:{b}, c:{c}' )
    
    return a,b,c 


# In[ ]:


NFOLDS = 5
NMODELS = 5

event_data = {}

# read data
train, test, train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode
train, test, event_data_update = encode_title(train, test, train_labels)
event_data.update(event_data_update)

reduce_train, reduce_test, reduce_train_from_test = get_train_and_test_single_proc(train, test, event_data, load_train=False)

reduce_train = post_process (reduce_train)
reduce_test = post_process (reduce_test)
reduce_train_from_test = post_process (reduce_train_from_test)
reduce_train.shape,reduce_test.shape,reduce_train_from_test.shape


# In[ ]:


# delete train and test to release memory
del train, test


# In[ ]:


import gc
gc.collect()


# In[ ]:


reduce_train = reduce_train.append (reduce_train_from_test, sort = False)


# In[ ]:


test = reduce_test
df_train = reduce_train

df_train["sample"] = df_train["installation_id"] + "_" + df_train["game_session"]
test["sample"]=test["installation_id"]


model_feats = [
'last_game_session_title', 'session_title', '4070_proc', 'acc_game_session_correct_accuracy', 'last_game_session_correct_accuracy',
'4020_proc', 'activity_min_duration', '2030_proc', 'last_game_timestamp', 'acc_current_assessment', '3021_proc', 'ata_current_assessment',
'2000_proc', 'last_2_game_session_correct_accuracy', '2010_proc', 'last_assessment_timestamp', 'last_game_event_count', 'Activity_gametime',
'4025_proc', 'last_activity_event_count', 'accumulated_accuracy', 'duration_mean', 'last_duration', '4090_proc', 'Clip', 'last_assessment_event_count',
'game_last_duration', 'game_last-2_duration', 'accumulated_accuracy_group', '3020_proc', '4040_proc', '4035_proc', 'afa_current_assessment',
'mean_game_duration', 'Activity_mean_event_count', '4030_proc', '4010_proc', '4100_proc', 'duration_std', 'duration_game_mean',
'last_assessment_session_title', '2035_proc', 'sin_hour', '3120_proc', '3110_proc', 'lgt_Cauldron Filler (Assessment)', '2025_proc',
'2080_proc', '3010_proc', '4021_proc', 'session_type_story_count_clip', 'agt_Cart Balancer (Assessment)', '4095_proc', 'Game_mean_event_count',
'2075_proc', 'game_max_duration', '2060_proc', '7372e1a5', 'duration_game_std', '56817e2b', 2000, 'Assessment_gametime', 'mean_game_round',
'15a43e5b', '2083_proc', 'var_title', '3ee399c3', 'cos_hour', 'last_2_game_session_correct_true', 'duration_max', 4035, '4031_proc', '5000_proc',
'4045_proc', 'agt_Cauldron Filler (Assessment)', 'dayofweek', 'bbfe0445', '4220_proc', 'b120f2ac', 'e694a35b', 'lgt_Chest Sorter (Assessment)',
'587b5989', '84538528', 'last_game_session_correct_true', 'current_4020_accuracy', '6bf9e3e1', 'Sandcastle Builder (Activity)', '3afde5dd',
'last_assessment_event_count_nunique', 'last_activity_event_count_nunique', '2081_proc', 3010, 3021, 3020, 'Mushroom Sorter (Assessment)_4020_accuracy',
'51102b85', '1bb5fbdb', '4110_proc', 'accumulated_uncorrect_attempts', 'Bird Measurer (Assessment)_4020_accuracy', '3bf1cf26', 'session_type_story_count_game',
'agt_Chest Sorter (Assessment)', 'agt_current_assessment', '562cec5f', '499edb7c', 'last_assessment_accuracy_group', 'ca11f653', '2040_proc', 4100, 4020,
'0db6d71d', '37ee8496', 'acc_Bird Measurer (Assessment)', 'Game_CRYSTALCAVES', 'acc_game_session_correct_true', 'lgt_Bird Measurer (Assessment)',
'ata_Chest Sorter (Assessment)', '907a054b', '3babcb9b', 'acc_Mushroom Sorter (Assessment)', 'acc_Chest Sorter (Assessment)', 'Bottle Filler (Activity)_4020',
'Game_TREETOPCITY', 'session_type_story_count_activity', 'last_activity_world_is_the_same', 'last_assessment_world_is_the_same', 'last_assessment_title_is_the_same',
]

categorical_features =  [f for f in model_feats if f in ['session_title',
                                                         'last_game_session_title',
                                                         'last_assessment_session_title',
                                                         'last_activity_session_title',                                                         
                                                         'last_assessment_accuracy_group']]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'nmodels = NMODELS\nnfolds = NFOLDS\nnum_boost_round = 1600 \nearly_stopping_rounds = None \nverbose_eval = 2000\nlgb_params = {\n          \'num_leaves\': 19, \n          \'min_data_in_leaf\': 160,\n          \'min_child_weight\': 0.03,\n          \'bagging_fraction\' : 0.7,\n          \'feature_fraction\' : 0.8,\n          \'learning_rate\' : 0.01,\n          \'max_depth\': -1,\n          \'reg_alpha\': 0.02,\n          \'reg_lambda\': 0.12,\n          \'objective\': \'regression\',\n          \'seed\': 1337,\n          \'feature_fraction_seed\': 1337,\n          \'bagging_seed\': 1337,\n          \'drop_seed\': 1337,\n          \'data_random_seed\': 1337,\n          \'boosting_type\': \'gbdt\',\n          \'verbose\': 100,\n          \'boost_from_average\': False,\n          \'metric\':\'rmse\'\n}        \n\ny_oof_lgb, rmse_scores,  oof_cohen_scores = run_lgb_ensemble (df_train, model_feats, categorical_features, num_boost_round, early_stopping_rounds,  lgb_params, nmodels, nfolds, verbose_eval )\n\nfeature_importances = pd.read_csv("./feature_importance.csv")\n\nfig, ax = plt.subplots(figsize=(16, 12))\nplt.subplot(1, 2, 1)\nsns.barplot(data=feature_importances[:50], x=\'average\', y=\'feature\', orient=\'h\')\nplt.title(\'Feature importances (LGB)\')\n\nplt.subplot(1, 2, 2)\nplt.hist(df_train["accuracy_group"].values.reshape(-1, 1) - y_oof_lgb.reshape(-1, 1))\nplt.title(\'Distribution of errors (LGB)\')\nplt.show()    \n    ')


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title('RMSE LGB')
plt.hist(rmse_scores, bins=100)

plt.subplot(1, 2, 2)
plt.title('QWK LGB')
plt.hist(oof_cohen_scores, bins=100)
plt.show()

print (f'LGB {nmodels} rmse: {np.median(rmse_scores)} std:{rmse_scores.std()} , oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')


# In[ ]:


a,b,c = calc_bound (y_oof_lgb, df_train["accuracy_group"])


oof_cohen_scores  = trunc_qwk_score ( df_train["accuracy_group"].values, round_prediction (y_oof_lgb,a,b,c), df_train["installation_id"].values, nsamples=5000)

fig, ax = plt.subplots(figsize=(16, 6))
plt.subplot(1, 1, 1)
plt.title('LGB QWK (percentile opt)')
plt.hist(oof_cohen_scores, bins=100)
plt.show()

print (f'a:{a} b:{b} c:{c} oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')


# In[ ]:


a,b,c = qwk3_optimizer ( df_train["accuracy_group"], y_oof_lgb)

oof_cohen_scores  = trunc_qwk_score ( df_train["accuracy_group"].values, round_prediction (y_oof_lgb,a,b,c), df_train["installation_id"].values, nsamples=5000)

fig, ax = plt.subplots(figsize=(16, 6))
plt.subplot(1, 1, 1)
plt.title('LGB QWK (bayesian opt)')
plt.hist(oof_cohen_scores, bins=100)
plt.show()

print (f'a:{a} b:{b} c:{c} oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nnmodels = NMODELS\nnfolds = NFOLDS\nnum_boost_round = 1600 \nearly_stopping_rounds = None \nverbose_eval = 2000\nlgb_params = {\n          \'num_leaves\': 19, \n          \'min_data_in_leaf\': 160,\n          \'min_child_weight\': 0.03,\n          \'bagging_fraction\' : 0.7,\n          \'feature_fraction\' : 0.8,\n          \'learning_rate\' : 0.01,\n          \'max_depth\': -1,\n          \'reg_alpha\': 0.02,\n          \'reg_lambda\': 0.12,\n          \'objective\': \'binary\',\n          \'seed\': 1337,\n          \'feature_fraction_seed\': 1337,\n          \'bagging_seed\': 1337,\n          \'drop_seed\': 1337,\n          \'data_random_seed\': 1337,\n          \'boosting_type\': \'gbdt\',\n          \'verbose\': 100,\n          \'boost_from_average\': False,\n          \'metric\':\'auc\'\n}        \n\ncategorical_features =  [f for f in model_feats if f in [\'session_title\',\n                                                         \'last_game_session_title\',\n                                                         \'last_assessment_session_title\',\n                                                         \'last_activity_session_title\',                                                         \n                                                         \'last_assessment_accuracy_group\']]\n    \ny_lgb_solved_oof, auc_score = run_lgb_auc_ensemble (df_train, df_train["accuracy_group"].map(lambda x: 0 if x ==0 else 1) , "solved",  model_feats, categorical_features, num_boost_round, early_stopping_rounds,  lgb_params, nmodels, nfolds, verbose_eval )\ny_lgb_first_oof, auc_score = run_lgb_auc_ensemble (df_train, df_train["accuracy_group"].map(lambda x: 1 if x ==3 else 0) , "first",  model_feats, categorical_features, num_boost_round, early_stopping_rounds,  lgb_params, nmodels, nfolds, verbose_eval )')


# In[ ]:


nmodels = NMODELS
nfolds = NFOLDS
num_boost_round = 600 
early_stopping_rounds = None 
verbose_eval = 1000

xgb_params = {
            'objective':'reg:squarederror',
            'eval_metric':'rmse',
            'seed': 1337,
            'colsample_bytree': 0.8,                 
            'learning_rate': 0.01,
            'max_depth': 9,
            'subsample': 0.7,
            'min_child_weight':3,
            'gamma':0.25,
            }    

categorical_features =  [f for f in model_feats if f in ['session_title',
                                                         'last_game_session_title',
                                                         'last_assessment_session_title',
                                                         'last_activity_session_title',                                                         
                                                         'last_assessment_accuracy_group']]

y_oof_xgb, rmse_scores,  oof_cohen_scores = run_xgb_ensemble (df_train, model_feats, categorical_features, num_boost_round, early_stopping_rounds,  xgb_params, nmodels, nfolds, verbose_eval )

plt.subplot(1, 1, 1)
plt.hist(df_train["accuracy_group"].values.reshape(-1, 1) - y_oof_xgb.reshape(-1, 1))
plt.title('Distribution of errors (XGB)')
plt.show()        


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title('RMSE XGB')
plt.hist(rmse_scores, bins=100)

plt.subplot(1, 2, 2)
plt.title('QWK XGB')
plt.hist(oof_cohen_scores, bins=100)
plt.show()

print (f'XGB {nmodels} rmse: {np.median(rmse_scores)} std:{rmse_scores.std()} , oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')


# In[ ]:


a,b,c = calc_bound (y_oof_xgb, df_train["accuracy_group"])


oof_cohen_scores  = trunc_qwk_score ( df_train["accuracy_group"].values, round_prediction (y_oof_xgb,a,b,c), df_train["installation_id"].values, nsamples=5000)

fig, ax = plt.subplots(figsize=(16, 6))
plt.subplot(1, 1, 1)
plt.title('XGB QWK (percentile opt)')
plt.hist(oof_cohen_scores, bins=100)
plt.show()

print (f'a:{a} b:{b} c:{c} oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')


# In[ ]:


a,b,c = qwk3_optimizer ( df_train["accuracy_group"], y_oof_xgb)

oof_cohen_scores  = trunc_qwk_score ( df_train["accuracy_group"].values, round_prediction (y_oof_xgb,a,b,c), df_train["installation_id"].values, nsamples=5000)

fig, ax = plt.subplots(figsize=(16, 6))
plt.subplot(1, 1, 1)
plt.title('XGB QWK (bayesian opt)')
plt.hist(oof_cohen_scores, bins=100)
plt.show()

print (f'a:{a} b:{b} c:{c} oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')


# In[ ]:


df = pd.DataFrame()
df["accuracy_group"] = df_train["accuracy_group"].values
df["sample"] = df_train["sample"].values
df["installation_id"] = df_train["installation_id"].values
df["y"] = y_oof_lgb
df["y_xgb"] = y_oof_xgb
df["solved"] = y_lgb_solved_oof
df["first"] = y_lgb_first_oof


# In[ ]:


y = (2*0.7*df["y"] + 2*0.3*df["y_xgb"] + 2*3*df["solved"]  + 3*df["first"])/5
rmse_scores,  oof_cohen_scores  = trunc_rmse_qwk_score ( df_train["accuracy_group"].values, y, df_train["installation_id"].values, nsamples=5000)
np.median(rmse_scores),  np.median(oof_cohen_scores)


# In[ ]:


a,b,c = calc_bound (y, df_train["accuracy_group"])

oof_cohen_scores  = trunc_qwk_score ( df_train["accuracy_group"].values, round_prediction (y,a,b,c), df_train["installation_id"].values, nsamples=5000)

fig, ax = plt.subplots(figsize=(16, 6))
plt.subplot(1, 1, 1)
plt.title('QWK (percentile opt)')
plt.hist(oof_cohen_scores, bins=100)
plt.show()

print (f'a:{a} b:{b} c:{c} oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')

a,b,c = qwk3_optimizer ( df_train["accuracy_group"], y)

oof_cohen_scores  = trunc_qwk_score ( df_train["accuracy_group"].values, round_prediction (y,a,b,c), df_train["installation_id"].values, nsamples=5000)

fig, ax = plt.subplots(figsize=(16, 6))
plt.subplot(1, 1, 1)
plt.title(' QWK (bayesian opt)')
plt.hist(oof_cohen_scores, bins=100)
plt.show()

print (f'a:{a} b:{b} c:{c} oof_cohen: {np.median(oof_cohen_scores)} std:{oof_cohen_scores.std()}')


# In[ ]:


del df_train
gc.collect()
psutil.virtual_memory()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ny_test_lgb = predict_lgb (NMODELS, NFOLDS) \ny_test_xgb = predict_xgb (NMODELS, NFOLDS) \ny_test_auc_solved = predict_auc_lgb ( "solved", NMODELS, NFOLDS)\ny_test_auc_first = predict_auc_lgb ( "first", NMODELS, NFOLDS)\n\n\ny_test = (2*0.7*y_test_lgb + 2*0.3*y_test_xgb + 2*3*y_test_auc_solved  + 3*y_test_auc_first)/5\ny_test = round_prediction (y_test,a,b,c)\n\nsubmission = pd.DataFrame()\nsubmission["installation_id"] = test["installation_id"]\nsubmission["accuracy_group"] = y_test.astype(int)\nsubmission.to_csv("submission.csv",index=False)')


# In[ ]:


submission.head()


# In[ ]:


submission.shape 

