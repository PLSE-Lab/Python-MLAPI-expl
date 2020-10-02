#!/usr/bin/env python
# coding: utf-8

# This kernel is totally based on [Catboost - Some more features](https://www.kaggle.com/braquino/catboost-some-more-features) kernel.
# 
# The main purpose of this kernel is to blend catboost and lightgbm.

# ### next steps
#     1. lgbm: use qwk as evaluation metric

# In[ ]:


import numpy as np
import pandas as pd
import datetime
from catboost import CatBoostClassifier
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb


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


get_ipython().run_cell_magic('time', '', "nrows = 100000\nnrows = None\ntrain = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', nrows=nrows)\ntrain_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv', nrows=nrows)\nspecs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv', nrows=nrows)\ntest = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', nrows=nrows)\nsubmission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')")


# In[ ]:


print('Number of rows: {}'.format(train.shape[0]))
keep_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()
train = pd.merge(train, keep_id, on="installation_id", how="inner")
print('Number of rows (after filtering): {}'.format(train.shape[0]))


# In[ ]:


# LabelEncode activities and event codes
# What about OneHotEncoding

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





# In[ ]:


# here the get_data function is applyed to each installation_id and added to the compile_data list
compiled_data = []
# tqdm is the library that draws the status bar below
for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False))):
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


import lightgbm as lgb
import numpy as np

from typing import Tuple, Union

def lgb_classification_qwk(y_true: Union[np.ndarray, list],
                           y_pred: Union[np.ndarray, list],) -> Tuple[str, float, bool]:
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return "qwk", qwk(y_true, y_pred), True


def qwk(y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        max_rat: int = 3) -> float:
    y_true_ = np.asarray(y_true)
    y_pred_ = np.asarray(y_pred)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    uniq_class = np.unique(y_true_)
    for i in uniq_class:
        hist1[int(i)] = len(np.argwhere(y_true_ == i))
        hist2[int(i)] = len(np.argwhere(y_pred_ == i))

    numerator = np.square(y_true_ - y_pred_).sum()

    denominator = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            denominator += hist1[i] * hist2[j] * (i - j) * (i - j)

    denominator /= y_true_.shape[0]
    return 1 - numerator / denominator


# In[ ]:


# this function makes the model and sets the parameters
# for configure others parameter consult the documentation below:
# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import xgboost as xgb

def make_classifier_1(iterations=6000):
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
        random_seed=45,
        #use_best_model=use_best_model,
        verbose=0
    )
        
    return clf

def make_classifier_2():
    params = {
            'learning_rate': 0.005,
            'metric': 'multiclass',
            'objective': 'multiclass',
            'num_classes': 4,
            'num_iterations': 5000,
            # 'feature_fraction': 0.75,
            'early_stopping_rounds': 20,
            # 'subsample': 0.75,
            'n_jobs': -1,
            'seed': 64,
            'verbose': 0
        }
    clf = LGBMClassifier(**params)
    return clf

def make_classifier_3():
    params = {
        'colsample_bytree': 0.8,                 
        'learning_rate': 0.08,
        'max_depth': 10,
        'subsample': 1,
        'objective':'multi:softprob',
        'eval_metric':'mlogloss',
        'min_child_weight':3,
        'gamma':0.25,
        'n_estimators':500
    }
    clf = XGBClassifier(**params)
    return clf


# In[ ]:


get_ipython().run_cell_magic('time', '', "# CV\nfrom sklearn.model_selection import KFold, StratifiedKFold\n# oof is an zeroed array of the same size of the input dataset\n\nNFOLDS = 7\n# here the KFold class is used to split the dataset in 5 diferents training and validation sets\n# this technique is used to assure that the model isn't overfitting and can performs aswell in \n# unseen data. More the number of splits/folds, less the test will be impacted by randomness\n\ntraining_start_time = time()\npredictions_tr = []\nmodels = []\nfor i in range(2):\n    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=345 * i)\n    oof = np.zeros(len(X))\n    for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):\n        # each iteration of folds.split returns an array of indexes of the new training data and validation data\n        start_time = time()\n        print(f'Training on fold {fold+1}')\n        # creates the model\n        for clf_clb in (\n            # make_classifier_3,\n            make_classifier_2,\n            make_classifier_1,\n        ):\n            # fits the model using .loc at the full dataset to select the splits indexes and features used\n            clf = clf_clb()\n            args = (X.loc[trn_idx, all_features], y.loc[trn_idx])\n            kwargs = {\n                'verbose': 0,\n                'eval_set': (X.loc[test_idx, all_features], y.loc[test_idx]),\n                \n            }\n            if clf_clb.__name__.endswith('1'):\n                kw = kwargs.copy()\n                kw.update({\n                    'use_best_model': True,\n                    'cat_features': cat_features,\n                    \n                })\n                clf.fit(*args, **kw)\n            elif clf_clb.__name__.endswith('3'):\n                xgb_train = xgb.DMatrix(X.loc[trn_idx, all_features], y.loc[trn_idx])\n                xgb_eval = xgb.DMatrix(X.loc[test_idx, all_features], y.loc[test_idx])\n                val_X=xgb.DMatrix(X.loc[test_idx, all_features])\n                params = {\n                    'learning_rate': 0.08,\n                    'max_depth': 10,\n                    'subsample': 1,\n                    'objective':'multi:softprob',\n                    'eval_metric':'mlogloss',\n                    'min_child_weight':3,\n                    'gamma':0.25,\n                    'n_estimators':500\n                }\n                xgb_model = xgb.train(params,\n                      xgb_train,\n                      evals=[(xgb_train, 'train'), (xgb_eval, 'val')],\n                      early_stopping_rounds=20,\n                      **params\n                     )\n\n            else:\n                kw = kwargs.copy()\n                clf.fit(*args, **kw)\n\n            # then, the predictions of each split is inserted into the oof array\n            # pr = clf.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))\n            pr = clf.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))\n            if clf_clb.__name__.endswith('3'):\n                pr = np.array(pr).argmax(axis=1)\n            oof[test_idx] = pr[:]\n            models.append(clf)\n    \n    print('-' * 30)\n    # and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)\n    print('OOF QWK:', i, qwk(y, oof))\n    print('-' * 30)")


# In[ ]:


# save predictions (don't ask why)
predictions_tr = []
for model in tqdm(models):
    pr = model.predict(X)
    if len(pr.shape) == 1:
        pr = np.array([[_] for _ in pr])
    predictions_tr.append(pr)

predictions_tr = np.array(predictions_tr).reshape(X.shape[0], len(models))

np.savetxt("predictions_tr.csv", predictions_tr, delimiter=",")


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


if 'accuracy_group' in X_test.columns:
    X_test.drop(['accuracy_group'], axis=1, inplace=True)


# In[ ]:


# make predictions on test set once
predictions = []
for model in models:
    pr = model.predict(X_test)
    if isinstance(pr[0], (int, float, np.int64)):
        pr = [[_] for _ in pr]
    elif 'xgboost' in str(model).lower():
        pr = list(np.array(pr).argmax(axis=1))
    predictions.append(pr)
predictions = np.concatenate(predictions, axis=1)

np.savetxt("predictions_ts.csv", predictions, delimiter=",")

print(predictions.shape)
predictions = stats.mode(predictions, axis=1)[0].reshape(-1)
print(predictions.shape)


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


# In[ ]:




