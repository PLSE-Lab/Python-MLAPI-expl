#!/usr/bin/env python
# coding: utf-8

# # Data visualization to help find the threshold

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
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
import json

train=pd.read_csv('../input/data-science-bowl-2019/train.csv')
train_labels=pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
submission=pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

# train dataset with only installation_id that are in train_labels:
not_req=(set(train.installation_id.unique()) - set(train_labels.installation_id.unique()))
train_new=~train['installation_id'].isin(not_req)
train.where(train_new,inplace=True)
train.dropna(inplace=True)
train['event_code']=train.event_code.astype(int)

def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
extract_time_features(train)
extract_time_features(test)

# encode title
# make a list with all the unique 'titles' from the train and test set
list_of_user_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))
# make a list with all the unique 'event_code' from the train and test set
list_of_event_code = list(set(train['event_code'].value_counts().index).union(set(test['event_code'].value_counts().index)))
# create a dictionary numerating the titles
activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

# replace the text titles withing the number titles from the dict
train['title'] = train['title'].map(activities_map)
test['title'] = test['title'].map(activities_map)
train_labels['title'] = train_labels['title'].map(activities_map)

win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
# then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
win_code[activities_map['Bird Measurer (Assessment)']] = 4110

# this is the function that convert the raw data into processed features
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
    
    # news features: time spent in each activity
    time_spent_each_act = {actv: 0 for actv in list_of_user_activities}
    event_code_count = {eve: 0 for eve in list_of_event_code}
    last_session_time_sec = 0
    
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy=0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0 
    accumulated_actions = 0
    
    last_accuracy_title = {'acc_' + title[0:4]: -1 for title in assess_titles}
    last_game_time_title = {'lgt_' + title[0:4]: 0 for title in assess_titles}
    ac_game_time_title = {'agt_' + title[0:4]: 0 for title in assess_titles}
    ac_true_attempts_title = {'ata_' + title[0:4]: 0 for title in assess_titles}
    ac_false_attempts_title = {'afa_' + title[0:4]: 0 for title in assess_titles}
    
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    time_play = 0
    title_just_before = 0
    title_assessment_before = 0
    assessment_before_accuracy = 0
    dif2030 = 0
    dif4070 = 0
    dif3010 = 0
    dif3020 = 0
    dif4030 = 0
    dif3110 = 0
    dif4025 = 0
    dif4035 = 0
    dif3120 = 0
    dif2010 = 0
    somme_clip_game_activity = 0
    title_just_before = 0
    actions_dif = 0
    
    # 'Cauldron Filler (Assessment)'
    Cauldron_Filler_misses = 0
    Cauldron_Filler_time = 0
    count_CFA = 0
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        
        # get current session time in seconds
        if session_type != 'Assessment':
            time_spent = int(session['game_time'].iloc[-1] / 1000)
            time_spent_each_act[activities_labels[session_title]] += time_spent
            time_play += time_spent
            
            title_just_before = session_title
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
            features.update(last_game_time_title.copy())
            features.update(ac_game_time_title.copy())
            features.update(ac_true_attempts_title.copy())
            features.update(ac_false_attempts_title.copy())
            
            session_title = session['title'].iloc[0]
            time_spent = int(session['game_time'].iloc[-1] / 1000)
            time_spent_each_act[activities_labels[session_title]] += time_spent
            
            features.update(time_spent_each_act.copy())
            features.update(event_code_count.copy())
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0] 
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            
            session_title_text = activities_labels[session_title]
            ac_true_attempts_title['ata_' + session_title_text[0:4]] += true_attempts
            ac_false_attempts_title['afa_' + session_title_text[0:4]] += false_attempts
            last_game_time_title['lgt_' + session_title_text[0:4]] = session['game_time'].iloc[-1]
            ac_game_time_title['agt_' + session_title_text[0:4]] += session['game_time'].iloc[-1]
            
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text[0:4]] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            
            # assessment_before_accuracy
            features['assessment_before_accuracy'] = assessment_before_accuracy
                     
            if accuracy == 0:
                features['accuracy_group'] = 0
                assessment_before_accuracy = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
                assessment_before_accuracy = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
                assessment_before_accuracy = 2
            else:
                features['accuracy_group'] = 1
                assessment_before_accuracy = 1
                
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            # encode installation_id
            features['installation_id'] = session['installation_id'].iloc[0]
            
            # time play on the app
            features['time_play'] = time_play
            time_play += int(session['game_time'].iloc[-1] / 1000)
            
            # title_assessment_before
            features['title_assessment_before'] = title_assessment_before
            
            # concat (session_title + title_assessment_before) / title_assessment_before is the title of the previous assessment :
            features['title_title_assessment_before'] = int( str(session['title'].iloc[0]) + str(title_assessment_before) )
            title_assessment_before = session['title'].iloc[0]
            
            # concat (session_title + title_just_before) / title_just_before is title game play just before the assessment :
            features['title_title_just_before'] = int( str(session['title'].iloc[0]) + str(title_just_before) )
            title_just_before = session['title'].iloc[0]
            
            
            # 4070 dif
            if features['Assessment'] == 0:
                features['4070_dif'] = features[4070]
                dif4070 = features[4070]
            else:
                features['4070_dif'] = features[4070] - dif4070
                dif4070 = features[4070]
                
            # 2030 dif
            if features['Assessment'] == 0:
                features['2030_dif'] = features[2030]
                dif2030 = features[2030]
            else:
                features['2030_dif'] = features[2030] - dif2030
                dif2030 = features[2030]
                
            # 3010 dif
            if features['Assessment'] == 0:
                features['3010_dif'] = features[3010]
                dif3010 = features[3010]
            else:
                features['3010_dif'] = features[3010] - dif3010
                dif3010 = features[3010]
                
            # 3020 dif
            if features['Assessment'] == 0:
                features['3020_dif'] = features[3020]
                dif3020 = features[3020]
            else:
                features['3020_dif'] = features[3020] - dif3020
                dif3020 = features[3020]
                
            # 4030 dif
            if features['Assessment'] == 0:
                features['4030_dif'] = features[4030]
                dif4030 = features[4030]
            else:
                features['4030_dif'] = features[4030] - dif4030
                dif4030 = features[4030]
                
            # 3110 dif
            if features['Assessment'] == 0:
                features['3110_dif'] = features[3110]
                dif3110 = features[3110]
            else:
                features['3110_dif'] = features[3110] - dif3110
                dif3110 = features[3110]
                
            # 4035 dif
            if features['Assessment'] == 0:
                features['4035_dif'] = features[4035]
                dif4035 = features[4035]
            else:
                features['4035_dif'] = features[4035] - dif4035
                dif4035 = features[4035]
                
            # 4025 dif
            if features['Assessment'] == 0:
                features['4025_dif'] = features[4025]
                dif4025 = features[4025]
            else:
                features['4025_dif'] = features[4025] - dif4025
                dif4025 = features[4025]
                
            # 3120 dif
            if features['Assessment'] == 0:
                features['3120_dif'] = features[3120]
                dif3120 = features[3120]
            else:
                features['3120_dif'] = features[3120] - dif3120
                dif3120 = features[3120]
                
            # 2010 dif
            if features['Assessment'] == 0:
                features['2010_dif'] = features[2010]
                dif2010 = features[2010]
            else:
                features['2010_dif'] = features[2010] - dif2010
                dif2010 = features[2010]
                
                
            # time play assessment
            features['time_play_assessment'] = sum(durations)
            
            # clip+game+activity before assessment
            somme = features['Clip']+features['Game']+features['Activity']
            features['somme_clip_game_activity'] = somme - somme_clip_game_activity
            somme_clip_game_activity = somme
            
            # somme actions 4070 + 2030 + 4030 / accumulated_actions :
            mean_actions = (features[4070] + features[4030] + features[2030]) / (features['accumulated_actions']+1)
            features['mean_actions'] = mean_actions
            
            # actions dif
            if features['Assessment'] == 0:
                features['actions_dif'] = features['accumulated_actions']
                actions_dif = features['accumulated_actions']
            else:
                features['actions_dif'] = features['accumulated_actions'] - actions_dif
                actions_dif = features['accumulated_actions']
            
            # somme actions 4070_dif + 2030_dif + 4030_dif / actions_dif :
            mean_actions = (features['4070_dif'] + features['4030_dif'] + features['2030_dif']) / (features['actions_dif']+1)
            features['mean_actions_dif'] = mean_actions
            
            
            # new features :

            features['Cauldron_Filler_misses'] = Cauldron_Filler_misses
            features['Cauldron_Filler_time'] = Cauldron_Filler_time
            
            if count_CFA == 0:
                if session['title'].iloc[0] == activities_map['Cauldron Filler (Assessment)']:
                    L = session.event_id.unique()
                    if '392e14df' in L:
                        if json.loads(session[session.event_id.isin(['392e14df'])].iloc[0]['event_data'])['correct'] == True:
                            Cauldron_Filler_misses = 2
                        else:
                            Cauldron_Filler_misses = 1       
                        Cauldron_Filler_time= session[session.event_id.isin(['392e14df'])].iloc[0]['game_time'] / 1000
                    else:
                        Cauldron_Filler_misses = -1
                    count_CFA = 1
                    
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
        n_of_event_codes = Counter(session['event_code'])
        
        for key in n_of_event_codes.keys():
            event_code_count[key] += n_of_event_codes[key]

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments[:-1]


# In[ ]:


from tqdm import tqdm_notebook as tqdm
from collections import Counter
# here the get_data function is applyed to each installation_id and added to the compile_data list
compiled_data = []
compiled_data_last = []
# tqdm is the library that draws the status bar below
for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=3614):
    # user_sample is a DataFrame that contains only one installation_id
    L = get_data(user_sample)
    compiled_data += L
    a = get_data(user_sample,test_set=True )
    compiled_data_last.append(a)

# the compiled_data is converted to DataFrame and deleted to save memmory
final_train = pd.DataFrame(compiled_data)
# final_train_last brings together every last assessment of installation_id from train dataset :
final_train_last = pd.DataFrame(compiled_data_last)

pd.set_option('display.max_columns', None)

# final_train_last2 brings together every second last assessment of installation_id from train dataset :
#    equivalent to bring together every last assessment of installation_id from final_train

# add a column index_t helping the separation between last assessment of final_train and others assessment
final_train = final_train.reset_index()
final_train.rename(columns={'index':'index_t'},inplace=True)

# every last assessment of installation_id from final_train
final_train_last2 = final_train.groupby('installation_id', sort=False,as_index=False).last()

# final_train2 = final_train  - final_train_last2

not_req=(set(final_train.index_t.unique()) - set(final_train_last2.index_t.unique()))
final_train2=final_train['index_t'].isin(not_req)
final_train2 = final_train.where(final_train2,try_cast=True)
final_train2.dropna(inplace=True)
final_train2['index_t']=final_train2.index.astype(int)

colonne = list(final_train2)
colonne_float = ['accumulated_accuracy_group','duration_mean','accumulated_accuracy','mean_actions','mean_actions_dif','installation_id']
for name in colonne:
    if name not in colonne_float:
        final_train2[name] = final_train2[name].astype(int)
        
# We do exactly the same to get every third last assessment of installation_id from train dataset:
#    equivalent to bring together every last assessment of installation_id from final_train2

final_train_last3 = final_train2.groupby('installation_id', sort=False,as_index=False).last()

not_req=(set(final_train2.index_t.unique()) - set(final_train_last3.index_t.unique()))
final_train3=final_train2['index_t'].isin(not_req)
final_train3 = final_train2.where(final_train3,try_cast=True)
final_train3.dropna(inplace=True)
final_train3['index_t']=final_train3.index.astype(int)

colonne = list(final_train3)
for name in colonne:
    if name not in colonne_float:
        final_train3[name] = final_train3[name].astype(int)
        
final_train_last4 = final_train3.groupby('installation_id', sort=False,as_index=False).last()
        
not_req=(set(final_train3.index_t.unique()) - set(final_train_last4.index_t.unique()))
final_train4=final_train3['index_t'].isin(not_req)
final_train4 = final_train3.where(final_train4,try_cast=True)
final_train4.dropna(inplace=True)
final_train4['index_t']=final_train4.index.astype(int)

colonne = list(final_train4)
for name in colonne:
    if name not in colonne_float:
        final_train4[name] = final_train4[name].astype(int)
        
final_train = final_train.drop(['index_t'],1)
final_train2 = final_train2.drop(['index_t'],1)
final_train3 = final_train3.drop(['index_t'],1)
final_train4 = final_train4.drop(['index_t'],1)
final_train_last2 = final_train_last2.drop(['index_t'],1)
final_train_last3 = final_train_last3.drop(['index_t'],1)
final_train_last4 = final_train_last4.drop(['index_t'],1)


# **final_train** is the train_label without the last assessment of every installation_id (no data of the test dataset)
# 
# **final_train_last** brings together last assesment of every installation_id of train_label
# *****
# **final_train2** is the train_label without the second and last assessments of every installation_id
# 
# **final_train_last2** brings together second assessment of every installation_id of train_label
# *****
# **final_train3** is the train_label without the third, second and last assessments of every installation_id
# 
# **final_train_last3** brings together third assessment of every installation_id of train_label
# ******
# **final_train4** is the train_label without the fourth,third, second and last assessments of every installation_id
# 
# **final_train_last4** brings together fourth assessment of every installation_id of train_label
# 
# # **final_train_lastx can be see as an example of private test**

# # Accuracy_group distribution of final_train and final_train_last

# In[ ]:


plt.figure(figsize = (30, 8))
plt.subplot(1,2,1)
final_train['accuracy_group'].plot(kind='hist', title="Accuracy_group distribution of final_train")
plt.subplot(1,2,2)
final_train_last['accuracy_group'].plot(kind='hist',colormap='magma', title="Accuracy_group distribution of final_train_last")
plt.show()


# # Percentage distribution

# In[ ]:


table_train = pd.concat([round(final_train['accuracy_group'].value_counts() / len(final_train),2),round(final_train_last['accuracy_group'].value_counts() / len(final_train_last),2)],axis=1)
table_train.columns = ['final_train','final_train_last']
table_train['dif'] = table_train['final_train_last'] - table_train['final_train'] 
table_train['final_train_last without_0_previous_assess'] = round(final_train_last[~final_train_last.Assessment.isin([0])]['accuracy_group'].value_counts() / len(final_train_last[~final_train_last.Assessment.isin([0])]),2)
table_train['dif2'] = table_train['final_train_last without_0_previous_assess'] - table_train['final_train'] 
table_train


# # Look out! The difference for 0 and 3 accuracy is significant

# In[ ]:


print("Distribution in final_train_last:")
print('Cases where sessions have no previous assessment : {}'.format(round(len(final_train_last[final_train_last.Assessment.isin([0])])/len(final_train_last),2)))
print('Cases where sessions have 1 previous assessment : {}'.format(round(len(final_train_last[final_train_last.Assessment.isin([1])])/len(final_train_last),2)))
print('Cases where sessions have 2 or more previous assessments: : {}'.format(round(len(final_train_last[~final_train_last.Assessment.isin([0,1])])/len(final_train_last),2)))


# # Accuracy_group distribution of final_train2 and final_train_last2

# In[ ]:


plt.figure(figsize = (30, 8))
plt.subplot(1,2,1)
final_train2['accuracy_group'].plot(kind='hist', title="Accuracy_group distribution of final_train2")
plt.subplot(1,2,2)
final_train_last2['accuracy_group'].plot(kind='hist',colormap='magma', title="Accuracy_group distribution of final_train_last2")
plt.show()


# In[ ]:


table_train2 = pd.concat([round(final_train2['accuracy_group'].value_counts() / len(final_train2),3),round(final_train_last2['accuracy_group'].value_counts() / len(final_train_last2),3)],axis=1)
table_train2.columns = ['final_train2','final_train_last2']
table_train2['dif'] = table_train2['final_train_last2'] - table_train2['final_train2']
table_train2['final_train_last2 without_0_previous_assess'] = round(final_train_last2[~final_train_last2.Assessment.isin([0])]['accuracy_group'].value_counts() / len(final_train_last2[~final_train_last2.Assessment.isin([0])]),3)
table_train2['dif2'] = table_train2['final_train_last2 without_0_previous_assess'] - table_train2['final_train2'] 
table_train2


# installation_id are the same in **final_train2** and **final_train_last2 without_0_previous_assess** and here the average of accuracy_group improved.

# In[ ]:


print("Distribution in final_train_last2:")
print('Cases where sessions have no previous assessment : {}'.format(round(len(final_train_last2[final_train_last2.Assessment.isin([0])])/len(final_train_last2),2)))
print('Cases where sessions have 1 previous assessment : {}'.format(round(len(final_train_last2[final_train_last2.Assessment.isin([1])])/len(final_train_last2),2)))
print('Cases where sessions have 2 or more previous assessments: : {}'.format(round(len(final_train_last2[~final_train_last2.Assessment.isin([0,1])])/len(final_train_last2),2)))


# # Accuracy_group distribution of final_train3 and final_train_last3

# In[ ]:


plt.figure(figsize = (30, 8))
plt.subplot(1,2,1)
final_train3['accuracy_group'].plot(kind='hist', title="Accuracy_group distribution of final_train3")
plt.subplot(1,2,2)
final_train_last3['accuracy_group'].plot(kind='hist',colormap='magma', title="Accuracy_group distribution of final_train_last3")
plt.show()


# In[ ]:


table_train3 = pd.concat([round(final_train3['accuracy_group'].value_counts() / len(final_train3),3),round(final_train_last3['accuracy_group'].value_counts() / len(final_train_last3),3)],axis=1)
table_train3.columns = ['final_train3','final_train_last3']
table_train3['dif'] = table_train3['final_train_last3'] - table_train3['final_train3']
table_train3['final_train_last3 without_0_previous_assess'] = round(final_train_last3[~final_train_last3.Assessment.isin([0])]['accuracy_group'].value_counts() / len(final_train_last3[~final_train_last3.Assessment.isin([0])]),3)
table_train3['dif2'] = table_train3['final_train_last3 without_0_previous_assess'] - table_train3['final_train3']
table_train3


# In[ ]:


print("Distribution in final_train_last3:")
print('Cases where sessions have no previous assessment : {}'.format(round(len(final_train_last3[final_train_last3.Assessment.isin([0])])/len(final_train_last3),2)))
print('Cases where sessions have 1 previous assessment : {}'.format(round(len(final_train_last3[final_train_last3.Assessment.isin([1])])/len(final_train_last3),2)))
print('Cases where sessions have 2 or more previous assessments: : {}'.format(round(len(final_train_last3[~final_train_last3.Assessment.isin([0,1])])/len(final_train_last3),2)))


# # Accuracy_group distribution of final_train4 and final_train_last4

# In[ ]:


plt.figure(figsize = (30, 8))
plt.subplot(1,2,1)
final_train4['accuracy_group'].plot(kind='hist', title="Accuracy_group distribution of final_train4")
plt.subplot(1,2,2)
final_train_last4['accuracy_group'].plot(kind='hist',colormap='magma', title="Accuracy_group distribution of final_train_last4")
plt.show()


# In[ ]:


table_train4 = pd.concat([round(final_train4['accuracy_group'].value_counts() / len(final_train4),3),round(final_train_last4['accuracy_group'].value_counts() / len(final_train_last4),3)],axis=1)
table_train4.columns = ['final_train4','final_train_last4']
table_train4['dif'] = table_train4['final_train_last4'] - table_train4['final_train4']
table_train4['final_train_last4 without_0_previous_assess'] = round(final_train_last4[~final_train_last4.Assessment.isin([0])]['accuracy_group'].value_counts() / len(final_train_last4[~final_train_last4.Assessment.isin([0])]),3)
table_train4['dif2'] = table_train4['final_train_last4 without_0_previous_assess'] - table_train4['final_train4']
table_train4


# In[ ]:


print("Distribution in final_train_last4:")
print('Cases where sessions have no previous assessment : {}'.format(round(len(final_train_last4[final_train_last4.Assessment.isin([0])])/len(final_train_last4),2)))
print('Cases where sessions have 1 previous assessment : {}'.format(round(len(final_train_last4[final_train_last4.Assessment.isin([1])])/len(final_train_last4),2)))
print('Cases where sessions have 2 or more previous assessments: : {}'.format(round(len(final_train_last4[~final_train_last4.Assessment.isin([0,1])])/len(final_train_last4),2)))


# **Engineering test data**

# **final_test** contains test data which have to be predict.
# 
# **before_test** brings together other data of test dataset.

# In[ ]:


# process test set, the same that was done with the train set
new_test = []
test_to_train = []
for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):
    L = get_data(user_sample)
    a = get_data(user_sample,test_set=True )
    new_test.append(a)
    test_to_train += L
    
final_test = pd.DataFrame(new_test)
# test_to_train contains data that we don't use for final_test:
before_test = pd.DataFrame(test_to_train)


# # Accuracy_group distribution of before_test

# In[ ]:


before_test['accuracy_group'].plot(kind='hist', title="Accuracy_group distribution of before_test")
plt.show()


# The distribution is similar than other final_trainx

# In[ ]:


table_test = pd.DataFrame(round(before_test['accuracy_group'].value_counts() / len(before_test),2))
table_test.columns = ['before_test']
table_test['test'] = 'unknown'
table_test['dif'] = ['d3','d0','d1','d2']
table_test


# We can maybe predict an estimation of d0,d1,d2,d3 with the LB.

# In[ ]:


print("Distribution in before_test:")
print('Cases where sessions have no previous assessment : {}'.format(round(len(before_test[before_test.Assessment.isin([0])])/len(before_test),2)))
print('Cases where sessions have 1 previous assessment : {}'.format(round(len(before_test[before_test.Assessment.isin([1])])/len(before_test),2)))
print('Cases where sessions have 2 previous assessment : {}'.format(round(len(before_test[~before_test.Assessment.isin([0,1])])/len(before_test),2)))


# In[ ]:


print("Distribution in final_test:")
print('Cases where sessions have no previous assessment : {}'.format(round(len(final_test[final_test.Assessment.isin([0])])/len(final_test),2)))
print('Cases where sessions have 1 previous assessment : {}'.format(round(len(final_test[final_test.Assessment.isin([1])])/len(final_test),2)))
print('Cases where sessions have 2 previous assessment : {}'.format(round(len(final_test[~final_test.Assessment.isin([0,1])])/len(final_test),2)))


# # Conclusion
# 
# * It seems in each data visualization that d1 and d2 are close to 0.
# 
# * final_train_last,final_train_last2,final_train_last3,final_train_last4 have almost the same distribution of number of Assessments but the distribution of accuracy_group of** final_train_last is really different from others. **
# 
#  **So it seems not good to use a fix threshold in case of the private test have an accuracy_group distribution more like final_train_last.**
#  
# * Investigate on the difference of distribution between final_train_last and others can maybe help for threshold.
#  
# * If we put aside the distribution of final_train_last, it seems to be interesting for the threshold to give **+(0.005 to 0.02) for group 3 accuracy and -(0.005 to 0.2) for 0 accuracy from train distribution, only for installation_id that have 1 or more previous assessments**. Maybe also -(0.005 to 0.01) for group 1 accuracy

# <font size=4 color='red'> Upvote if you think this kernel was helpful !</font>
