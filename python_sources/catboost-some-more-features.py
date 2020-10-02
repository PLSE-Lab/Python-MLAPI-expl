#!/usr/bin/env python
# coding: utf-8

# ### Update notes  
# 
# I have added some new very simple features to this kernel:  
# 1. How much time was spent in each Title  
# 2. How many actions was registered in each event_code  

# ### Commentator notes  
# 
# This kernel isn't mine, so, if you liked it, upvote this official one made by Massoud Hosseinali: https://www.kaggle.com/mhviraf/a-new-baseline-for-dsb-2019-catboost-model  
# I find this kernel so good and elegant, and also is the top score public on this competition, then I decided to use it as baseline for my attempts, so I have scrutinated the whole notebook and left comments on almost each line.  
# I think this comments can help who wants to use it as baseline aswell, I hope you enjoy the comments and forgive me the english errors.  
#   
# Massoud, if you see any problem doing this, or any error in my comments, tell me please. And congratulations for this kernel.   

# ### Kernel creator notes  
# 
# I had posted my very naive baseline at https://www.kaggle.com/mhviraf/a-baseline-for-dsb-2019. In that kernel I only used the mode label for each Assessment and I thought it should be very easy to beat. This kernel shows how you can beat that baseline by actually applying a model. In this kernel via `get_data()` function, I go over each `installation_id` and try to extract some features based on his/her behavior prior to the assessment. I will then train a `Catboost` classifier on it and make predictions on the test set. Note that the features I made in this kernel are so very basic and you can easily add many more to it. Good luck and happy kaggling. Don't forget to upvote if you found it useful ;)

# In[ ]:


import numpy as np
import pandas as pd
import datetime
from catboost import CatBoostClassifier
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats


# In[ ]:


from sklearn.metrics import confusion_matrix
# this function is the quadratic weighted kappa (the metric used for the competition submission)
def qwk(act,pred,n=4,hist_range=(0,3)):
    
    # Calculate the percent each class was tagged each label
    O = confusion_matrix(act,pred)
    # normalize to sum 1
    O = np.divide(O,np.sum(O))
    
    # create a new matrix of zeroes that match the size of the confusion matrix
    # this matriz looks as a weight matrix that give more weight to the corrects
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # makes a weird matrix that is bigger in the corners top-right and botton-left (= 1)
            W[i][j] = ((i-j)**2)/((n-1)**2)
            
    # make two histograms of the categories real X prediction
    act_hist = np.histogram(act,bins=n,range=hist_range)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
    # multiply the two histograms using outer product
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E)) # normalize to sum 1
    
    # apply the weights to the confusion matrix
    num = np.sum(np.multiply(W,O))
    # apply the weights to the histograms
    den = np.sum(np.multiply(W,E))
    
    return 1-np.divide(num,den)
    


# In[ ]:


train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')


# In[ ]:


# encode title
# make a list with all the unique 'titles' from the train and test set
list_of_user_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))
# make a list with all the unique 'event_code' from the train and test set
list_of_event_code = list(set(train['event_code'].value_counts().index).union(set(test['event_code'].value_counts().index)))
# create a dictionary numerating the titles
activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

# replace the text titles withing the number titles from the dict
train['title'] = train['title'].map(activities_map)
test['title'] = test['title'].map(activities_map)
train_labels['title'] = train_labels['title'].map(activities_map)


# In[ ]:


# I didnt undestud why, but this one makes a dict where the value of each element is 4100 
win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
# then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
win_code[activities_map['Bird Measurer (Assessment)']] = 4110


# In[ ]:


# convert text into datetime
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])


# In[ ]:


train.head()


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
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    
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
    return all_assessments


# In[ ]:


# here the get_data function is applyed to each installation_id and added to the compile_data list
compiled_data = []
# tqdm is the library that draws the status bar below
for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):
    # user_sample is a DataFrame that contains only one installation_id
    compiled_data += get_data(user_sample)


# In[ ]:


# the compiled_data is converted to DataFrame and deleted to save memmory
new_train = pd.DataFrame(compiled_data)
del compiled_data
new_train.shape


# Below are the features I have generated. Note that all of them are **prior** to each event. For example, the first row shows **before** this assessment, the player have watched 3 clips, did 3 activities, played 4 games and solved 0 assessments, so on so forth.

# In[ ]:


pd.set_option('display.max_columns', None)
new_train[:10]


# ## Model

# In[ ]:


# this list comprehension create the list of features that will be used on the input dataset X
# all but accuracy_group, that is the label y
all_features = [x for x in new_train.columns if x not in ['accuracy_group']]
# this cat_feature must be declared to pass later as parameter to fit the model
cat_features = ['session_title']
# here the dataset select the features and split the input ant the labels
X, y = new_train[all_features], new_train['accuracy_group']
del train
X.shape


# In[ ]:


# this function makes the model and sets the parameters
# for configure others parameter consult the documentation below:
# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
def make_classifier(iterations=6000):
    clf = CatBoostClassifier(
                               loss_function='MultiClass',
                                eval_metric="WKappa",
                               task_type="CPU",
                               #learning_rate=0.01,
                               iterations=iterations,
                               od_type="Iter",
                                #depth=4,
                               early_stopping_rounds=500,
                                #l2_leaf_reg=10,
                                #border_count=96,
                               random_seed=42,
                                #use_best_model=use_best_model
                              )
        
    return clf


# In[ ]:


# CV
from sklearn.model_selection import KFold
# oof is an zeroed array of the same size of the input dataset
oof = np.zeros(len(X))
NFOLDS = 5
# here the KFold class is used to split the dataset in 5 diferents training and validation sets
# this technique is used to assure that the model isn't overfitting and can performs aswell in 
# unseen data. More the number of splits/folds, less the test will be impacted by randomness
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)
training_start_time = time()
models = []
for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
    # each iteration of folds.split returns an array of indexes of the new training data and validation data
    start_time = time()
    print(f'Training on fold {fold+1}')
    # creates the model
    clf = make_classifier()
    # fits the model using .loc at the full dataset to select the splits indexes and features used
    clf.fit(X.loc[trn_idx, all_features], y.loc[trn_idx], eval_set=(X.loc[test_idx, all_features], y.loc[test_idx]),
                          use_best_model=True, verbose=500, cat_features=cat_features)
    
    # then, the predictions of each split is inserted into the oof array
    oof[test_idx] = clf.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))
    models.append(clf)
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
    print('____________________________________________________________________________________________\n')
    #break
    
print('-' * 30)
# and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)
print('OOF QWK:', qwk(y, oof))
print('-' * 30)


# In[ ]:


# train model on all data once
#clf = make_classifier()
#clf.fit(X, y, verbose=500, cat_features=cat_features)

del X, y


# In[ ]:


# process test set, the same that was done with the train set
new_test = []
for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):
    a = get_data(user_sample, test_set=True)
    new_test.append(a)
    
X_test = pd.DataFrame(new_test)
del test


# In[ ]:


# make predictions on test set once
predictions = []
for model in models:
    predictions.append(model.predict(X_test))
predictions = np.concatenate(predictions, axis=1)
print(predictions.shape)
predictions = stats.mode(predictions, axis=1)[0].reshape(-1)
print(predictions.shape)
#del X_test


# ## Make submission

# In[ ]:


submission['accuracy_group'] = np.round(predictions).astype('int')
submission.to_csv('submission.csv', index=None)
submission.head()


# In[ ]:


submission['accuracy_group'].plot(kind='hist')


# In[ ]:


train_labels['accuracy_group'].plot(kind='hist')


# In[ ]:


pd.Series(oof).plot(kind='hist')

