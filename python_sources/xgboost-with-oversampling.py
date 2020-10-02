#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit,KFold, StratifiedKFold
from time import time
import pickle
import datetime as dt
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import csc_matrix


# In[ ]:


# read data
def init_read_big(train_path, test_path, train_big_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_big = pd.read_csv(train_big_path)
    return train, test, train_big

def init_read_small(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# In[ ]:


def remove_column_big(train, train_big, column_title_big):
    
    #train = train.drop(['Quote_ID'], axis=1)
    train_small = train.copy()
    column_title_small = [A for A in train.columns]
    train_big = train_big[column_title_big]
    train = train_big.copy()
    train.columns = column_title_small
    
    return train
    
    

def data_preprocessing(train, test):
    train = train.drop(['Quote_ID'], axis=1)
    
    y = train.QuoteConversion_Flag.values
    
    train = train.drop(['QuoteConversion_Flag'], axis=1)
    test = test.drop('Quote_ID', axis=1)
    
    
    train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
    train = train.drop('Original_Quote_Date', axis=1)
    
    test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
    test = test.drop('Original_Quote_Date', axis=1)
    
    
    #train['Date_value'] = train['Date'].apply(lambda x: float((x-dt.datetime(2009, 12, 30)).days)+(float((x-dt.datetime(2009, 12, 30)).seconds)/86400))
    #test['Date_value'] = test['Date'].apply(lambda x: float((x-dt.datetime(2009, 12, 30)).days)+(float((x-dt.datetime(2009, 12, 30)).seconds)/86400))
    
    
    train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
    train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
    train['weekday'] = train['Date'].dt.dayofweek

    test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
    test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
    test['weekday'] = test['Date'].dt.dayofweek
    
    # count the number of missing values in each row
    #train['miss_count'] = train.apply(lambda x: len(train.columns)- x.count(), axis=1)
    #test['miss_count'] = test.apply(lambda x: len(test.columns)- x.count(), axis=1)
    
    # drop redundant attributes
    #train = train.drop(['Date','Property_info2','Field_info1'], axis=1)
    #test = test.drop(['Date','Property_info2','Field_info1'], axis=1)
    
    train = train.drop(['Date'], axis=1)
    test = test.drop(['Date'], axis=1)
    
    
    '''
    # use one-hot encoding for categorical attribute
    
    onehot_columns =  [
                'Field_info1',
                'Field_info3',                
                'Coverage_info3',
                'Sales_info4',
                'Geographic_info5']
    
    labelencoder_columns = ['Field_info4','Personal_info1','Property_info1',
                            'Geographic_info4','Personal_info3','Property_info3']
    
    
    for f in labelencoder_columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
        
    train = pd.get_dummies(train, prefix_sep="__",columns=onehot_columns)
    test = pd.get_dummies(test, prefix_sep="__",columns=onehot_columns)
   
    '''
    
    # simplying convert all categorical attributes to number
    for A in train.columns:
        if train[A].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[A].values) + list(test[A].values))
            train[A] = lbl.transform(list(train[A].values))
            test[A] = lbl.transform(list(test[A].values))
    
    
    # fill missing values for Personal_info5
    #train["Personal_info5"] = train["Personal_info5"].fillna(0)
    #test["Personal_info5"] = test["Personal_info5"].fillna(0)
    
    
    train = train.fillna(-999)
    test = test.fillna(-999)
    
    
    
    print("\nPre-processing_big complete!!")
    
    return train, test, y


# In[ ]:


train_path = '/kaggle/input/2019s-uts-data-analytics-assignment-3/Assignment3_TrainingSet.csv'
test_path = '/kaggle/input/2019s-uts-data-analytics-assignment-3/Assignment3_TestSet.csv'
train, test = init_read_small(train_path, test_path)
train, test, y = data_preprocessing(train, test)


# In[ ]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(train, y)


# In[ ]:


# Training on all data
'''


clf = xgb.XGBClassifier(
                            
                        n_estimators=200,
                        min_child_weight = 1.2,
                        max_depth=6,
                        verbosity = 1,
                        n_jobs=8,                                              
                        scale_pos_weight=1.025,
                        tree_method='gpu_exact',
                        objective = 'binary:logistic',
                        predictor='gpu_predictor',
                        colsample_bytree = 0.66,
                        subsample = 1,
                        gamma = 0,
                        learning_rate=0.15,
                        num_parallel_tree = 5,
    
                       )
                        
xgb_model = clf.fit(X_ros, y_ros, eval_metric="auc",verbose=True)

'''


# In[ ]:


n_fold = 5
random_state = 999
#kf =  StratifiedKFold(n_splits = n_fold , shuffle = True, random_state = random_state)

models = []
train_no = 1
training_cycle = 200
repetition = 5
training_sequence = 1

for A in range(repetition):
    
    kf =  StratifiedKFold(n_splits = n_fold , shuffle = True, random_state = random_state+A)
    
    for train_index, val_index in kf.split(X_ros, y_ros):
        train_X = X_ros[train_index]
        val_X = X_ros[val_index]
        train_y = y_ros[train_index]
        val_y = y_ros[val_index]

        print(f'\n\nurrently Training sequence no {training_sequence}')
        clf = xgb.XGBClassifier(

                            n_estimators=training_cycle,
                            min_child_weight = 2,
                            max_depth=6,
                            verbosity = 1,
                            n_jobs=8,                                              
                            scale_pos_weight=1.025,
                            tree_method='gpu_exact',
                            objective = 'binary:logistic',
                            predictor='gpu_predictor',
                            colsample_bytree = 0.66,
                            subsample = 1,
                            gamma = 0,
                            learning_rate=0.15,
                            num_parallel_tree = 5,

                           )


        clf.fit(train_X, train_y, eval_metric="auc", early_stopping_rounds=50,
                    eval_set=[(train_X, train_y), (val_X, val_y)],verbose=True)

        models.append(clf)
        training_sequence += 1


# In[ ]:


'''

X_train, X_valid, y_train, y_valid = train_test_split(X_ros, y_ros, test_size = 0.2,random_state = 999)

time_now = time()
clf = xgb.XGBClassifier(
                            
                        n_estimators=5000,
                        min_child_weight = 1.2,
                        max_depth=6,
                        verbosity = 1,
                        n_jobs=8,                                              
                        scale_pos_weight=1.025,
                        tree_method='gpu_exact',
                        objective = 'binary:logistic',
                        predictor='gpu_predictor',
                        colsample_bytree = 0.66,
                        subsample = 1,
                        gamma = 0,
                        learning_rate=0.15,
                        num_parallel_tree = 5,
    
                       )
                        
xgb_model = clf.fit(X_train, y_train, eval_metric="auc", early_stopping_rounds=100,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=True)

time_new = time()
training_duration = time_new - time_now
'''


# In[ ]:


pred_df = sum([clf.predict(test.values) for clf in models])/(n_fold*repetition)
preds = clf.predict_proba(test.values)[:,1]
for A in range(len(preds)):
    if preds[A] > 0.6:
        preds[A] = int(1)
    else:
        preds[A] = int(0)


# In[ ]:


sum(preds)


# In[ ]:


sample = pd.read_csv('/kaggle/input/2019s-uts-data-analytics-assignment-3/Assignment3_Random_Submission-Kaggle.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('XGB_Oversample.csv', index=False)
print()
print("#########SUBMISSION FILE UPLOADED!!! WELL DONE!!!#########")


# In[ ]:


import pickle
pickle.dump(clf, open("XGBoost_model.dat", "wb"))

