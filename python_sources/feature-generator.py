__author__ = 'brdi'

import datetime
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import zipfile
import time
import shutil
from sklearn.metrics import log_loss

random.seed(2016)

def run_xgb(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 3
    subsample = 0.7
    colsample_bytree = 0.7
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 500
    early_stopping_rounds = 50
    test_size = 0.3

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    score = log_loss(y_valid.tolist(), check)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    test_val = test['device_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table


def calc_timerange(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    return abs((d2 - d1))
    
def read_train_test():
    # Events
    print('Read events...')
    # appLabels = pd.read_csv("../input/app_labels.csv")
    # tp = pd.read_csv('../input/app_events.csv', iterator=True, chunksize=10000)  # gives TextFileReader
    # print('read')
    # for chunk in tp:
    #     print('merged')
    #     print(chunk.shape)
    #     appEvents = pd.merge(chunk, appLabels, on='app_id')
    #     print(appEvents.shape)
    #     #df = pd.concat(appEvents, ignore_index=True)
    # print('done')
    # #print(df.shape)
    # print(appEvents.shape)

    labelCategories = pd.read_csv("../input/label_categories.csv",  dtype={'label_id': np.str})

    
    
    
    events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})#, nrows = 10000)
    appEvents = pd.read_csv("../input/app_events.csv", dtype={'app_id': np.str})#, nrows = 10000)
    appLabels = pd.read_csv("../input/app_labels.csv", dtype={'app_id': np.str, 'label_id': np.str})

    device = pd.DataFrame()
    print(datetime.now())
    #1. count of times used device
    device['count'] = events.groupby(['device_id'])['event_id'].count()
    print('Model features: events shape {}, appEvents shape {}, appLabels shape {}, labelCategories shape {}, device final shape {}'
    .format(events.shape, appEvents.shape, appLabels.shape, labelCategories.shape, device.shape))
    
    #print(appLabels.head(20))#same appId has multiple labelIds
    # appLabels = appLabels[appLabels.app_id == "6058196446775239644"]
    #TODO: add my personal categories as well as booleans for each app, print column name for each app
    appLabels = pd.merge(appLabels, labelCategories, on='label_id')
    appLabels = appLabels.drop('label_id',1).drop_duplicates() #there is no nulls here, this data is perfect
    appLabels['c'] = pd.Series(1, index=appLabels.index)
    appLabels = pd.pivot_table(appLabels,index='app_id',values='c',columns='category')
    appLabels['app_id'] = appLabels.index 
    #print(appLabels.head(20))
    # appEvents = pd.merge(appEvents, appLabels, on='app_id')
    # print(appEvents.head(20))
    # print(appEvents.isnull().sum())


    events = events[events.device_id == "-6401643145415154744"]
    device = device[device.index == "-6401643145415154744"]

    events = pd.merge(events, appEvents, how='left', on='event_id')
    print(events.shape)
    print(events)
    # print(events.isnull().sum()) #sometimes there is no array of apps with the event
    print('done')
    
    #for each event how many total apps, active apps
    #for all events what are all unique apps, total is_installed and total is_active count
    print(events['is_installed'].nunique())
    print(events['app_id'].nunique())
    print(events['is_active'].sum())
    print(events['event_id'].nunique())


    foo = events.groupby(['device_id','app_id'])
    print(foo['is_active'].count())
    print(foo['is_installed'].count())
    print('hmm')

    
    
    #TODO: generate clusters of apps as well as top N apps/app clusters for 7-10
    #create clusters in label Categories 
    #print(labelCategories.category.unique())
    #print(labelCategories.shape)
    #print(labelCategories.category.value_counts())
	#11. Labels amongst these top apps - what type do they often use
	#TODO: get just the single set of all apps associated with the user. not per event
	
	#TODO: how many times an app was reopened. top isActive repeated
	
    # print(events.groupby(['device_id','category']).nlargest(5))
    #eventsCategoryDescribed2 = events.groupby(['device_id'])['category'].value_counts().unstack()
    #print(eventsCategoryDescribed2)

    
    #TODO: 6. for each app installed, what is the active/inactive rate as well as # of times that app has shown up

    #6. Number of active apps per device - How many apps do they use
    eventsActiveApps = events[events.is_active == 1.0]
    bar = eventsActiveApps.groupby(['device_id','app_id'])
    print(bar['is_active'].count())
    print(bar['is_installed'].count())
    device['activeApps'] = eventsActiveApps.groupby(['device_id','app_id'])['is_active'].count()
    print(device.head(10))
    #7. Number of installed apps per device - how many apps do they have
    device['installedApps'] = events.groupby(['device_id','app_id'])['is_installed'].count()
    device = device.sort_values('count', ascending=False)
    print(device.head(10))
    events = events.sort_values('app_id', ascending=False)

    print(events[events.app_id == 8693964245073640147])
    print('the values we actually want') #for this app it is 34 installed,and 32/34
    
    
    #8. Top label across all apps - what type do they often use
    #this works just commenting out for now
    # eventsCategoryDescribed = events.groupby(['device_id'])['category'].describe().unstack()
    # #device['topApp'] = events.groupby(['device_id'])['category'].nth(0)
    # device['topAppLabel'] = eventsCategoryDescribed['top']
    # print(device.head(10))
    # device['topAppLabel'].fillna("missing", inplace=True)
    # print(device.head(10))
    # device = map_column(device, 'topAppLabel')

    # #9. top 5 apps used - what are they often using
    # device['topAppLabelCount'] = eventsCategoryDescribed['freq']
    
    # #10. Number of unique labels (how diverse are they)
    # device['uniqueLabelCount'] = eventsCategoryDescribed['unique']

    
    #2. mean/std of lat long excluding 0s to see how much they travel and general location
    #TODO: need to replace NaN std with the overall average of people like them
    #TODO: need to replace NaN mean with... remove probably, or just 0
    print('Generate lat/long calculations...')
    eventsLat = events[events.latitude !=0]
    eventsLat = eventsLat.groupby(['device_id'])['latitude']
    # events['latitude_mean'] = eventsLat.transform(np.mean)
    # events['latitude_std'] = eventsLat.transform(np.std)
    device['latitude_mean'] = eventsLat.mean()
    device['latitude_std'] = eventsLat.std()
    eventsLon = events[events.longitude !=0]
    eventsLon = eventsLon.groupby(['device_id'])['longitude']
    # events['longitude_mean'] = eventsLon.transform(np.mean)
    # events['longitude_std'] = eventsLon.transform(np.std)
    device['longitude_mean'] = eventsLon.mean()
    device['longitude_std'] = eventsLon.std()
    
    #3. Do dotw/hour as rows
    #TODO: cluster in periods of 4 hour clusters as well as m-thur than fri-sat
    #FIX: should just be count of events not multiple if we joined table with appEvents
    print('Generate dotw/hour calculations...')
    eventsDatetime = pd.to_datetime(events.timestamp, format='%Y-%m-%d %H:%M:%S')
    events['dotw'] = eventsDatetime.dt.dayofweek
    events['hour'] = eventsDatetime.dt.hour
    device[['mo','tu','we','th','fr','sa','su']] = events.groupby(['device_id'])['dotw'].value_counts().unstack().fillna(0.)
    #device[['h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23']] = events.groupby(['device_id'])['hour'].value_counts().unstack().fillna(0.)

    device['device_id'] = device.index
    device.sort_values(by='device_id',ascending=True)
    print(device.head(5))
    print(device.shape)
    # device.drop_duplicates('device_id', keep='first', inplace=True)

    #4. Max-min timestamp for the device - how long have they had the phone
    #TODO: meh might not be worth doing this one because only a short range overall
    #TODO: average gap between phone pickups, std deviation/get how many clusters of pickups
    
	
    #this kinda way works too
    # grouped = events.groupby(['device_id'])
    # print(grouped['latitude'].agg({'latitude_mean' : np.mean, 'latitude_std' : np.std}))
    # print(grouped['longitude'].agg({'longitude_mean' : np.mean, 'longitude_std' : np.std}))
    # print(grouped.head(5))

    #testing a single id
    # print(events.head(5))
    # print(events[events.device_id == "-1007186911744386922"])
    # # events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
    # # print(events_small.head(5))
    # dummy = events[events.device_id == "29182687948017175"]
    # print(dummy.head(5))
    # print(dummy['timestamp'].describe())
    # #4. Max-min timestamp for the device - how long have they had the phone
    # mostRecentTime = events['timestamp'].max()
    # oldestTime = events['timestamp'].min()
    # print(mostRecentTime)
    # print(oldestTime)
    # print('earliest timestamp')
    # mostRecentTime = dummy['timestamp'].max()
    # oldestTime = dummy['timestamp'].min()
    # print(mostRecentTime)
    # print(oldestTime)
    # print(calc_timerange(mostRecentTime,oldestTime))
    # #find what is correlated about the time variables, because otherwise useless
    # print(dummy.head())
    # #time of day, day of week, distribution of timestamps
    # print(dummy['hour'].describe())
    # print(dummy['dotw'].describe())
    # print(dummy['hour'].value_counts()) #what popular hour they use phone
    # print(dummy['dotw'].value_counts()) #what popular/least day they use phone

    
    

    

    # Phone brand
    print('Read brands...')
    pbd = pd.read_csv("../input/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')
    pbd.sort_values(by='device_id', ascending=True)
    print(pbd[pbd['device_id'] == '-100015673884079572'])
    print(pbd.head(5))
    print(pbd.shape)

    # Train
    print('Read train...')
    train = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str})
    train.sort_values(by='device_id',ascending=True)
    print(train[train['device_id'] =='-8260683887967679142'])
    train = map_column(train, 'group')
    train = map_column(train, 'gender')
    # train = train.drop(['age'], axis=1)
    # train = train.drop(['gender'], axis=1)
    print(train.shape)
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    print(train.shape)
    
    train = pd.merge(train, device, how='inner', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)
    train = train.sort_values('activeApps', ascending=False)
    print(train.head(200))
    # train.to_csv("train1.csv", sep='\t')
    print(train[train.columns[:]].corr()['age'].sort_values())
    print(train[train.columns[:]].corr()['gender'].sort_values())
    #FIX:women use their phone more yet the count is pos correlated with men
    #FIX:younger use their phone more yet the count is pos correlated with age
    #TODO: theres definitely a huge prediction we should make on low count but huge number of apps. they have very similar patterns, phone brand etc.
    #TODO:cluster device model/phone brand to price and popular and have that be a feature
    print(train[train.columns[:]].corr()['device_model'].sort_values())
    
    #TODO: split each train by groupId and see charts of what they are like

    print(train.head(5))
    print(train.shape)

    # Test
    print('Read test...')
    test = pd.read_csv("../input/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, device, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)

    # Features
    features = list(test.columns.values)
    features.remove('device_id')
    print(train[train.columns[:]].corr()['group'].sort_values())
    return train, test, features

train, test, features = read_train_test()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
# test_prediction, score = run_xgb(train, test, features, 'group')
# print("LS: {}".format(round(score, 5)))
# create_submission(score, test, test_prediction)
