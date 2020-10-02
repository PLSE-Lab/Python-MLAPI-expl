# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""
@author: michael.hartman
"""

import pandas as pd
import numpy as np
import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import time
from datetime import timedelta
    
def get_params():
    plst_list = []

    params = {}
    params["objective"] = 'multi:softprob' 
    params["num_class"] = 12    
    params["eta"] = 0.02
    params["min_child_weight"] = 50
    params["subsample"] = 0.4
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params['nthread'] = 1
    params["max_depth"] = 8
    params["eval_metric"] = 'mlogloss'
    plst = list(params.items())
    plst_list.append(plst) 

    return plst_list
    
train_load_list = [
                  {'name': 'gender_age_train',
                   'types': {'device_id': np.dtype(np.str),
                             'gender': np.dtype(np.str),
                             #'age' : np.dtype(np.int8),
                             'group': np.dtype(np.str)
                             },
                   'na_filter': False,
                   'parse_dates': False,
                   'factorize': ['gender']}]
test_load_list = [
                  {'name': 'gender_age_test',
                   'types': {'device_id': np.dtype(np.str),},
                   'na_filter': False,
                   'parse_dates': False,
                   'factorize': None},]                   
phone_load_list = [
                  {'name': 'phone_brand_device_model',
                   'types': {'device_id': np.dtype(np.str),
                             'phone_brand': np.dtype(np.str),
                             'device_model' : np.dtype(np.str)},
                   'na_filter': False,
                   'parse_dates': False,
                   'factorize': ['phone_brand', 'device_model']},]
event_load_list = [
                  {'name': 'events',
                   'types': {'event_id': np.dtype(np.int64),
                             'device_id': np.dtype(np.str),
                             'timestamp': np.dtype(np.str),
                             'longitude' : np.dtype(np.float64),
                             'latitude': np.dtype(np.float64)},
                   'na_filter': True,
                   'parse_dates': ['timestamp'],
                   'factorize': None},
                  ]
app_event_list = [
                  {'name': 'app_events',
                   'types': {'event_id': np.dtype(np.int64),
                             'app_id': np.dtype(np.str),
                             #'is_installed' : np.dtype(np.str),
                             #'is_active' : np.dtype(np.bool)
                             },
                   'na_filter': True,
                   'parse_dates': False,
                   'factorize': None}]

def get_data(data_load_list, datapath):
    for d in data_load_list:
        print('Loading', d['name'])
        filename = datapath + d['name'] + '.csv'
        df_load = pd.read_csv(filename, usecols=d['types'].keys(), 
                              dtype=d['types'], na_filter=d['na_filter'], 
                              parse_dates=d['parse_dates'])
        if d['factorize'] != None:
            for col in d['factorize']:
                df_load[col], unique = pd.factorize(df_load[col])
                print(col, 'has', unique.size, 'unique values')
    return df_load
    
def get_event_pred(df_event, df_train, df_test):
    # Convert time to number
    df_event['timestamp'] = df_event['timestamp'].astype(np.int64) // 10**9
    df_event['timestamp'] = df_event['timestamp'] - 1462060000
    # Store device id for train
    # merge in labels
    df_train_event = df_train.merge(df_event, how='inner', on='device_id')
    df_train_event.drop_duplicates()
    # Add place to store predictions
    df_train_event['M'] = 0
    df_train_event['F'] = 0
    # Events in test
    df_test_event = df_test.merge(df_event, how='inner', on='device_id')
    df_test_event.drop_duplicates()
    # Add place to store predictions
    df_test_event['M'] = 0
    df_test_event['F'] = 0
    # Prepare data for classifier
    train_event = df_train_event[['timestamp', 'longitude', 'latitude']].values
    train_label = df_train_event['gender'].values
    test_event = df_test_event[['timestamp', 'longitude', 'latitude']].values
    # Prepare splits for train
    folds = 2
    train_folds = np.array_split(df_train['device_id'].values, folds)
    # Predict on location and time for train
    for fold in train_folds:
        clf = GaussianNB()
        fold_mask = df_train_event['device_id'].isin(fold).values
        train_cut = train_event[~fold_mask]
        labels = train_label[~fold_mask]
        pred_cut = train_event[fold_mask]
        # Train the model
        clf.fit(train_cut, labels)
        # Make predictions
        pred = clf.predict_proba(pred_cut)
        df_train_event.loc[fold_mask, ['M','F']] = pred
        pred = clf.predict_proba(test_event)
        df_test_event[['M','F']] += pred
    # Average test predictions
    df_test_event[['M','F']] /= folds
    # Combine results
    df_event_gender = pd.concat((df_train_event[['device_id', 'M','F']], 
                                df_test_event[['device_id', 'M','F']]))
    df_event_gender = df_event_gender.groupby(['device_id'], as_index=False).mean()
    return df_event_gender
    

missing_indicator = np.NAN
xgb_num_rounds = 420

print('Load the data using pandas')
start_time = time.time()
datapath = '../input/'
df_train = get_data(train_load_list, datapath)
df_test = get_data(test_load_list, datapath)
df_phone = get_data(phone_load_list, datapath)
df_event = get_data(event_load_list, datapath)
df_app_event = get_data(app_event_list, datapath)

elapsed = (time.time() - start_time)
print('Data loaded in:', timedelta(seconds=elapsed))

print('Feature engineering')
# Encode labels
le_label = LabelEncoder()
df_train['group'] = le_label.fit_transform(df_train['group'].values)

# Get unique values for phone
df_phone = df_phone.groupby('device_id', as_index=False).first()

# Filter out infrequent phone types
df_phone_count = df_phone.groupby(['phone_brand'])
infrequent = df_phone_count['device_id'].filter(lambda x: len(x) < 30)
df_phone.loc[df_phone['device_id'].isin(infrequent), 'phone_brand'] = -1
df_phone_count = df_phone.groupby(['device_model'])
infrequent = df_phone_count['device_id'].filter(lambda x: len(x) < 30)
df_phone.loc[df_phone['device_id'].isin(infrequent), 'device_model'] = -1

# Get app counts
df_app_event = df_app_event.groupby(['event_id'], as_index=False)
df_app_event = df_app_event['app_id'].count()
df = df_event[['event_id', 'device_id']]
df_device_app = df.merge(df_app_event, how='inner', on='event_id')
del df
df_device_app.drop(['event_id'], axis=1, inplace=True)
df_device_app = df_device_app.groupby(['device_id'], as_index=False)
df_device_app = df_device_app['app_id'].median()
df_device_app.rename(columns={'app_id': 'app_count'}, inplace=True)

# Get count of events
df_event_count = df_event.groupby(['device_id'], as_index=False)
df_event_count = df_event_count['event_id'].count()
df_event_count.rename(columns={'event_id': 'event_count'}, inplace=True)
df_event.drop(['event_id'], axis=1, inplace=True)

# Get Bayesian probability for gender based on time and location
df_event_gender = get_event_pred(df_event, df_train, df_test)
df_train.drop(['gender'], axis=1, inplace=True)

# Merge data
df_train = df_train.merge(df_phone, how='inner', on='device_id')
df_test = df_test.merge(df_phone, how='inner', on='device_id')
df_train = df_train.merge(df_event_count, how='left', on='device_id')
df_test = df_test.merge(df_event_count, how='left', on='device_id')
df_train = df_train.merge(df_device_app, how='left', on='device_id')
df_test = df_test.merge(df_device_app, how='left', on='device_id')
df_train = df_train.merge(df_event_gender, how='left', on='device_id')
df_test = df_test.merge(df_event_gender, how='left', on='device_id')

# Create dataframe to store predictions
df_pred = df_test[['device_id']].copy()
for label in le_label.classes_:
    df_pred[label] = 0
df_pred.set_index('device_id', inplace=True, verify_integrity=True)

labels = df_train['group'].values
df_train.drop(['group'], axis=1, inplace=True)
df_train.set_index('device_id', inplace=True, verify_integrity=True)
df_test.set_index('device_id', inplace=True, verify_integrity=True)

elapsed = (time.time() - start_time)
print('Data prepared in:', timedelta(seconds=elapsed))

plst_list = get_params()
xgbp = plst_list[0]

xgtrain = xgb.DMatrix(df_train, labels, missing=missing_indicator)
xgtest = xgb.DMatrix(df_test, missing=missing_indicator)
model = xgb.train(xgbp, xgtrain, xgb_num_rounds)
test_pred = model.predict(xgtest)
df_pred[df_pred.columns] = test_pred

print('Saving Predictions')
df_pred.to_csv('xgb_submission.csv')

elapsed = (time.time() - start_time)
print('Completed in:', timedelta(seconds=elapsed))