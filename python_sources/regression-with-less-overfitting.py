#!/usr/bin/env python
# coding: utf-8

# <font size=4 color='blue'>2 problems of overfitting that I try to resolve in this kernel:</font>
# 
# > > 
# - use information group by installation_id could lead to overfitting
# > > 
# - use all data of train_label for validation dataset. Indeed, the validation score is low if I only use the last assessment of every installation_id as validation dataset compared to a validation score where I use random validation dataset.

# <font size=4 color='blue'>solutions:</font>
# 
# - I just don't use information group by installation_id but I will use data between an assessment and his previous assessment (e.g: the number of 4070 between two assess)
#  
# 
# - I use specific validation datasets:
# 
#   1. Validation dataset1: last assessment of every installation_id of train_label
#       
#      Train dataset1 = **final_train**: other assessments of train_label
#   
#   2. I do the same thing on **final_train**:
#    
#      Validation dataset2: second last assessment of every installation_id on train_labels
#       
#      Train dataset2 = **final_train2**: other assessments of **final_train**
#       
#   3. I do the same thing on **final_train2**:
#    
#      Validation dataset3: third last assessment of every installation_id on train_labels
#       
#      Train dataset3 = **final_train3**: other assessments of **final_train2**
#       
# 
# **The goal of the point 2** is to only have the last assessments in the validation dataset.
# 
# **You can use these 3 validation datasets to evaluate your model.** The 3 validation datasets are really different. I fit a lot of models with these dataset and I have always a big difference between validation scores(particularly with the validation dataset with last assessment). It would mean that the models are not really good.
# 
# Have a good and same score on the 3 validation scores + good score score on test data would mean your model is less impact by overfitting.

# I will use the **lgb regression model** with parameters of **Andrew Lukyanenko kernel** and also his optimization cutoffs.

# **Importing libraries and data**

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


# In[ ]:


get_ipython().run_cell_magic('time', '', "train=pd.read_csv('../input/data-science-bowl-2019/train.csv')\ntrain_labels=pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')\ntest = pd.read_csv('../input/data-science-bowl-2019/test.csv')\nsubmission=pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')")


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


not_req=(set(train.installation_id.unique()) - set(train_labels.installation_id.unique()))
train_new=~train['installation_id'].isin(not_req)
train.where(train_new,inplace=True)
train.dropna(inplace=True)
train['event_code']=train.event_code.astype(int)


# In[ ]:


def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
extract_time_features(train)
extract_time_features(test)


# In[ ]:


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


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data = pd.concat([train,test],sort=False)

data['installation_id_encoder'] = 0
encode = data[['installation_id']].apply(encoder.fit_transform)
data['installation_id_encoder'] = encode
data.head(10)

train = data[:len(train)]
test = data[len(train):]


# In[ ]:


win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
# then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
win_code[activities_map['Bird Measurer (Assessment)']] = 4110


# In[ ]:


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
    
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    last_game_time_title = {'lgt_' + title: 0 for title in assess_titles}
    ac_game_time_title = {'agt_' + title: 0 for title in assess_titles}
    ac_true_attempts_title = {'ata_' + title: 0 for title in assess_titles}
    ac_false_attempts_title = {'afa_' + title: 0 for title in assess_titles}
    
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
            ac_true_attempts_title['ata_' + session_title_text] += true_attempts
            ac_false_attempts_title['afa_' + session_title_text] += false_attempts
            last_game_time_title['lgt_' + session_title_text] = session['game_time'].iloc[-1]
            ac_game_time_title['agt_' + session_title_text] += session['game_time'].iloc[-1]
            
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
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
            features['installation_id_encoder'] = session['installation_id_encoder'].iloc[0]
            
            # time play on the app
            features['time_play'] = time_play
            time_play += int(session['game_time'].iloc[-1] / 1000)
            
            # title_assessment_before
            features['title_assessment_before'] = title_assessment_before
            
            # concat (session_title + title_assessment_before) / title_assessment_before is the title of the previous assessment :
            features['title*title_assessment_before'] = int( str(session['title'].iloc[0]) + str(title_assessment_before) )
            title_assessment_before = session['title'].iloc[0]
            
            # concat (session_title + title_just_before) / title_just_before is title game play just before the assessment :
            features['title*title_just_before'] = int( str(session['title'].iloc[0]) + str(title_just_before) )
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


# **Engineering train data**

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


# **final_train_last** brings together every last assessment of installation_id from train dataset :

# In[ ]:


final_train_last = pd.DataFrame(compiled_data_last)


# In[ ]:


name_colonne = list(final_train.iloc[:,4:29])


# **final_train** doesn't contain data from **final_train_last**
# 
# And we can now use **final_train** as a train dataset and **final_train_last** as a validation dataset

# In[ ]:


pd.set_option('display.max_columns', None)
final_train


# 4070_dif variable is the difference between the number of 4070 of the assessment and his previous assessment of an installation_id

# # I remove outliers id : 

# In[ ]:


final_train = final_train[~final_train.installation_id_encoder.isin([2572,4351,2901,3064,3759,2012,3804,1406,2055,2096,3766,3347,3994,4209,4444])]
final_train_last = final_train_last[~final_train_last.installation_id_encoder.isin([2572,4351,2901,3064,3759,2012,3804,1406,2055,2096,3766,3347,3994,4209,4444])]


# **final_train_last2** brings together every second last assessment of installation_id from **train** :
#     
#    equivalent to bring together every last assessment of installation_id from **final_train**

# In[ ]:


# add a column index_t helping the separation between last assessment of final_train and others assessment
final_train = final_train.reset_index()
final_train.rename(columns={'index':'index_t'},inplace=True)


# In[ ]:


# every last assessment of installation_id from final_train
final_train_last2 = final_train.groupby('installation_id_encoder', sort=False,as_index=False).last()


# In[ ]:


final_train_last2


# **final_train2**  = **final_train**   -   **final_train_last2** 

# In[ ]:


not_req=(set(final_train.index_t.unique()) - set(final_train_last2.index_t.unique()))
final_train2=final_train['index_t'].isin(not_req)
final_train2 = final_train.where(final_train2,try_cast=True)
final_train2.dropna(inplace=True)
final_train2['index_t']=final_train2.index.astype(int)

colonne = list(final_train2)
colonne_float = ['accumulated_accuracy_group','duration_mean','accumulated_accuracy']
for name in colonne:
    if name not in colonne_float:
        final_train2[name] = final_train2[name].astype(int)


# In[ ]:


final_train2


# We can now use **final_train2** as a train dataset and **final_train_last2** as a validation dataset

# We do exactly the same to get every third last assessment of installation_id from **train**:
# 
#    equivalent to bring together every last assessment of installation_id from **final_train2**

# In[ ]:


final_train_last3 = final_train2.groupby('installation_id_encoder', sort=False,as_index=False).last()


# In[ ]:


not_req=(set(final_train2.index_t.unique()) - set(final_train_last3.index_t.unique()))
final_train3=final_train2['index_t'].isin(not_req)
final_train3 = final_train2.where(final_train3,try_cast=True)
final_train3.dropna(inplace=True)
final_train3['index_t']=final_train3.index.astype(int)

colonne = list(final_train3)
colonne_float = ['accumulated_accuracy_group','duration_mean','accumulated_accuracy']
for name in colonne:
    if name not in colonne_float:
        final_train3[name] = final_train3[name].astype(int)


# We can now use **final_train3** as a train dataset and **final_train_last3** as a validation dataset

# In[ ]:


final_train = final_train.drop(['index_t'],1)
final_train2 = final_train2.drop(['index_t'],1)
final_train3 = final_train3.drop(['index_t'],1)
final_train_last2 = final_train_last2.drop(['index_t'],1)
final_train_last3 = final_train_last3.drop(['index_t'],1)


# **Engineering test data**

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
test_to_train = pd.DataFrame(test_to_train)
final_test.shape


# **test_to_train** contains data that we don't use for final_test, I put them in train dataset:

# In[ ]:


final_train = pd.concat([final_train,test_to_train])
final_train = final_train.reset_index(drop=True)
final_train2 = pd.concat([final_train2,test_to_train])
final_train2 = final_train2.reset_index(drop=True)
final_train3 = pd.concat([final_train3,test_to_train])
final_train3 = final_train3.reset_index(drop=True)


# **Feature selection**

# In[ ]:


keep = ['accuracy_group','session_title','accumulated_accuracy_group','4070_dif','2030_dif',
        'duration_mean','4030_dif','accumulated_uncorrect_attempts','Chow Time','Clip',
        'somme_clip_game_activity','assessment_before_accuracy','accumulated_actions',0,3] + name_colonne
final_train = final_train[keep]
final_train_last = final_train_last[keep]
final_train2 = final_train2[keep]
final_train_last2 = final_train_last2[keep]
final_train3 = final_train3[keep]
final_train_last3 = final_train_last3[keep]
final_test = final_test[keep]


# In[ ]:


print(final_train_last.shape)
print(final_train_last2.shape)
print(final_train_last3.shape)


# In[ ]:


final_train


# **LGBregressor / params from Andrew Lukyanenko kernel**

# In[ ]:


params = {'n_estimators':2000,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.04,
            'feature_fraction': 0.9,
         'max_depth': 15,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'verbose': 100,
            'early_stopping_rounds': 100, 'eval_metric': 'cappa'
            }


# In[ ]:


def lgb_regression(x_train, y_train, x_val, y_val, **kwargs):
    
    models = []
    scores=[]
    categoricals = ['session_title']
        
    train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
    val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)
        
    model = lgb.train(train_set=train_set, valid_sets=[train_set, val_set],**kwargs)
    models.append(model)
        
    pred_val=model.predict(x_val)
    oof = pred_val.reshape(len(x_val))
        
    return models,oof


# **final_train3** as a train data and **final_train_last3** as a validation data :

# In[ ]:


X_train3 = final_train3.drop('accuracy_group',axis=1)
y_train3 = final_train3['accuracy_group'].astype(float)
X_end3 = final_train_last3.drop('accuracy_group',axis=1)
y_end3 = final_train_last3['accuracy_group'].astype(float)

models3,oof3 = lgb_regression(X_train3, y_train3, X_end3, y_end3, params=params, num_boost_round=100000,
                  early_stopping_rounds=500, verbose_eval=40)


# **final_train2** as a train data and **final_train_last2** as a validation data :

# In[ ]:


X_train2 = final_train2.drop('accuracy_group',axis=1)
y_train2 = final_train2['accuracy_group'].astype(float)
X_end2 = final_train_last2.drop('accuracy_group',axis=1)
y_end2 = final_train_last2['accuracy_group'].astype(float)

models2,oof2 = lgb_regression(X_train2, y_train2, X_end2, y_end2, params=params, num_boost_round=40000,
                  early_stopping_rounds=500, verbose_eval=40)


# **final_train** as a train data and **final_train_last** as a validation data :

# In[ ]:


X_train1 = final_train.drop('accuracy_group',axis=1)
y_train1 = final_train['accuracy_group'].astype(float)
X_end1 = final_train_last.drop('accuracy_group',axis=1)
y_end1 = final_train_last['accuracy_group'].astype(float)

models1,oof1 = lgb_regression(X_train1, y_train1, X_end1, y_end1, params=params, num_boost_round=40000,
                  early_stopping_rounds=500, verbose_eval=40)

models_LGBM = models1 + models2 + models3


# We can note that the scores on validation data are really different. **The validation rmse score on final_train_last is quite high.**
# 
# Maybe the model is not really robust considering different model.

# In[ ]:


lgb.plot_importance(models_LGBM[0],max_num_features=20,importance_type='gain')


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

    def _kappa_loss(self, coef, X, y,X2, y2,X3, y3):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions of final_train_last
        :param y: The ground truth labels
        :param X2: The raw predictions of final_train_last2
        :param y2: The ground truth labels
        :param X3: The raw predictions of final_train_last3
        :param y3: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])
        X_p2 = pd.cut(X2, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])
        X_p3 = pd.cut(X3, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])
        
        print("validation score of the last assessment: score1 = {}".format(qwk(y, X_p)))
        
        return ( qwk(y2, X_p2) - qwk(y, X_p) ) + ( qwk(y3, X_p3) - qwk(y, X_p) )  + (0.51 - qwk(y, X_p))*2

    def fit(self, X, y,X2, y2,X3, y3,coef_ini):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y,X2=X2, y2=y2,X3=X3, y3=y3)
        initial_coef = coef_ini
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


# score1 = validation score on final_last_assessment
# 
# score2 = validation score on final_last_assessment2
# 
# score3 = validation score of on final_last_assessment3
# 
# In my previous versions, I noted that the quadratic weighted kappa score on final_train_last is low compared to others and the differences (score2 - score1) and (score3 - score1) seem to be negatively correlated to test score.
# 
# So I modificated the function OptimizedRounder. I try now to reduce the differences between scores. My goal is to have the same score on the 3 validation scores and also a good score (it's why I added (0.49 - qwk(y, X_p))*2 ). 0.49 seem to be the limit of my validation score on final_train_last with this model.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noptR = OptimizedRounder()\noptR.fit(oof1,y_end1,oof2,y_end2,oof3,y_end3,[1.12, 1.83, 2.27])\ncoefficients = optR.coefficients()\nprint(coefficients)')


# In[ ]:


oof3[oof3 <= coefficients[0]] = 0
oof3[np.where(np.logical_and(oof3 > coefficients[0], oof3 <= coefficients[1]))] = 1
oof3[np.where(np.logical_and(oof3 > coefficients[1], oof3 <=coefficients[2]))] = 2
oof3[oof3 > coefficients[2]] = 3
pred3 = np.round(oof3).astype('int')

score3 = qwk(y_end3, pred3)
print("validation score of the third last assessment: score3 = {}".format(score3))


# In[ ]:


oof2[oof2 <= coefficients[0]] = 0
oof2[np.where(np.logical_and(oof2 > coefficients[0], oof2 <= coefficients[1]))] = 1
oof2[np.where(np.logical_and(oof2 > coefficients[1], oof2 <=coefficients[2]))] = 2
oof2[oof2 > coefficients[2]] = 3
pred2 = np.round(oof2).astype('int')

score2 = qwk(y_end2, pred2)
print("validation score of the second last assessment: score2 = {}".format(score2))


# In[ ]:


oof1[oof1 <= coefficients[0]] = 0
oof1[np.where(np.logical_and(oof1 > coefficients[0], oof1 <= coefficients[1]))] = 1
oof1[np.where(np.logical_and(oof1 > coefficients[1], oof1 <=coefficients[2]))] = 2
oof1[oof1 > coefficients[2]] = 3
pred1 = np.round(oof1).astype('int')

score1 = qwk(y_end1, pred1)
print("validation score of the last assessment: score1 = {}".format(score1))


# **Training on all train dataset with seed averaging**

# In[ ]:


params = {'n_estimators':1000,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.04,
            'feature_fraction': 0.9,
          'seed':42,
         'max_depth': 15,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'verbose': 100,
            'early_stopping_rounds': 100, 'eval_metric': 'cappa'
            }
X = pd.concat([X_train1,X_end1])
Y = pd.concat([y_train1,y_end1])
models_all = []
oof_all = np.zeros(len(X))
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train, val in kf.split(X, Y):
    model_all,oof = lgb_regression(X.iloc[train], Y.iloc[train], X.iloc[val], Y.iloc[val], params=params, num_boost_round=40000,
                     early_stopping_rounds=500, verbose_eval=40)
    models_all.append(model_all[0])
    oof_all[val] = oof


# In[ ]:


lgb.plot_importance(model_all[0],max_num_features=20,importance_type='gain')


# In[ ]:


oof_all[oof_all <= coefficients[0]] = 0
oof_all[np.where(np.logical_and(oof_all > coefficients[0], oof_all <= coefficients[1]))] = 1
oof_all[np.where(np.logical_and(oof_all > coefficients[1], oof_all <=coefficients[2]))] = 2
oof_all[oof_all > coefficients[2]] = 3
oof_all = np.round(oof_all).astype('int')

score = qwk(Y, oof_all)
print("validation score with StratifiedKFold_n=5: score = {}".format(score))


# In[ ]:


params = {'n_estimators':1000,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.04,
            'feature_fraction': 0.9,
          'seed':15,
         'max_depth': 15,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'verbose': 100,
            'early_stopping_rounds': 100, 'eval_metric': 'cappa'
            }

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
for train, val in kf.split(X, Y):
    model_all,oof = lgb_regression(X.iloc[train], Y.iloc[train], X.iloc[val], Y.iloc[val], params=params, num_boost_round=40000,
                     early_stopping_rounds=500, verbose_eval=200)
    models_all.append(model_all[0])

params = {'n_estimators':1000,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.04,
            'feature_fraction': 0.9,
          'seed':12,
         'max_depth': 15,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'verbose': 100,
            'early_stopping_rounds': 100, 'eval_metric': 'cappa'
            }

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
for train, val in kf.split(X, Y):
    model_all,oof = lgb_regression(X.iloc[train], Y.iloc[train], X.iloc[val], Y.iloc[val], params=params, num_boost_round=40000,
                     early_stopping_rounds=500, verbose_eval=200)
    models_all.append(model_all[0])
    
params = {'n_estimators':1000,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.04,
            'feature_fraction': 0.9,
          'seed':11,
         'max_depth': 15,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'verbose': 100,
            'early_stopping_rounds': 100, 'eval_metric': 'cappa'
            }

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
for train, val in kf.split(X, Y):
    model_all,oof = lgb_regression(X.iloc[train], Y.iloc[train], X.iloc[val], Y.iloc[val], params=params, num_boost_round=40000,
                     early_stopping_rounds=500, verbose_eval=200)
    models_all.append(model_all[0])


# In[ ]:


X_test = final_test.drop(columns=['accuracy_group'])


# In[ ]:


predictions = []
for model in models_all:
    predictions.append(model.predict(X_test))

L=[]
for i in range (len(predictions[0])):
    mean = []
    for j in range (len(predictions)):
        mean.append(predictions[j][i])
    L.append(np.mean(mean))
    
predictions = np.array(L)

predictions[predictions <= coefficients[0]] = 0
predictions[np.where(np.logical_and(predictions > coefficients[0], predictions <= coefficients[1]))] = 1
predictions[np.where(np.logical_and(predictions > coefficients[1], predictions <=coefficients[2]))] = 2
predictions[predictions > coefficients[2]] = 3

pred = np.round(predictions).astype('int')

sub_LGB_test=pd.DataFrame({'installation_id':submission.installation_id,'accuracy_group':pred})
sub_LGB_test.to_csv('submission.csv',index=False)
sub_LGB_test['accuracy_group'].plot(kind='hist')


# In[ ]:


sub_LGB_test['accuracy_group'].value_counts()


# <font size=4 color='red'> Upvote if you think this kernel was helpful !</font>
