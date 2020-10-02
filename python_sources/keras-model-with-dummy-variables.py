#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Inspirational Notebooks
#https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
#https://www.kaggle.com/shep312/single-generalised-lightgbm-lb-0-9686/code
#
import pandas as pd
from IPython.display import display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import gc
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import np_utils
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler

os.environ['OMP_NUM_THREADS'] = '4'  # Number of threads on the Kaggle server

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

n_rows_to_train = 3000000
n_rows_to_test = 1000000

dummycol = ['ip_cat','app','os']
notcol = ['click_id','is_attributed']


def load_talking_train_big_data(n_rows):
    data = pd.read_csv("D:/Kaggle/TalkingData/train.csv", dtype=dtypes, nrows=n_rows, usecols=train_cols)   
    return data

def load_test_model_data(n_rows,skiprows):
    train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    data_test = pd.read_csv("D:/Kaggle/TalkingData/train.csv",
                            dtype=dtypes, nrows=n_rows, skiprows = range(1,skiprows), usecols=train_cols)
    return data_test

def load_test_submission_data():
    data_test = pd.read_csv("D:/Kaggle/TalkingData/test.csv", dtype=dtypes, usecols=test_cols)
    return data_test

def get_features(td):
    
    GROUP_AGG_FLAG = True
    GROUP_NEXT_CLICKS_FLAG = True
    HISTORY_CLICKS_FLAG = True

    td['click_time'] = pd.to_datetime(td.click_time)
    td['hour'] = td.click_time.dt.hour.astype('uint8')
    td['day'] = td.click_time.dt.day.astype('uint8')
    
    #add a section of day
    day_section = 0
    for start_time, end_time in zip([0,6,12,18],[6,12,18,24]):
        td.loc[(td['hour'] >= start_time) & (td['hour'] < end_time), 'day_section'] = day_section
        day_section +=1
    td['day_section'] = td['day_section'].astype('uint8')
    gc.collect()
    
    if(GROUP_AGG_FLAG==True):
        
        GROUPBY_AGGREGATIONS = [

            # V1 - GroupBy Features #
            #########################    
            # Count, for ip
            {'groupby': ['ip'], 'select': 'channel', 'agg': 'count'},
            # Count, for ip-day-hour
            {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
            # Count, for ip-app
            {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
            # Count, for ip-device
            {'groupby': ['ip', 'device'], 'select': 'channel', 'agg': 'count'},
            # Count, for ip-device-os
            {'groupby': ['ip', 'device','os'], 'select': 'channel', 'agg': 'count'},

            # V2 - GroupBy Features #
            #########################
            # Average clicks on app by distinct users; is it an app they return to?
            {'groupby': ['app'], 
             'select': 'ip', 
             'agg': lambda x: float(len(x)) / len(x.unique()), 
             'agg_name': 'AvgViewPerDistinct'
            },
            # How popular is the app or channel?
            {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
            {'groupby': ['channel'], 'select': 'app', 'agg': 'count'}
        ]

        for spec in GROUPBY_AGGREGATIONS:
            agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
            print("Grouping by {}, and aggregating {} with {}".format(spec['groupby'], spec['select'], agg_name))        
            list_features = list(set(spec['groupby']+[spec['select']]))
            name_feature = "{}_{}_{}".format('_'.join(spec['groupby']),agg_name,spec['select'])

            new_feature = td[list_features].groupby(spec['groupby'])[spec['select']].agg(spec['agg']).                reset_index().rename(index=str,columns={spec['select']:name_feature}).fillna(0)
            gc.collect()
            td = td.merge(new_feature,on=spec['groupby'],how="left")
            del(new_feature)
            gc.collect()
        
        #add category of ip
        td.loc[td['ip_count_channel'] == 1, 'ip_cat'] = 0
        cat = 1
        for start, end in zip([2,5,10,15,25,50],[5,10,15,25,50,100]):
            td.loc[(td['ip_count_channel'] >= start) & (td['ip_count_channel'] < end) , 'ip_cat'] = cat
            cat +=1
        td.loc[td['ip_count_channel'] >= 100, 'ip_cat'] = cat
        td['ip_cat'] = td['ip_cat'].astype('uint8')
        gc.collect()

    if(GROUP_NEXT_CLICKS_FLAG == True):  
        
        #add features of time till next click
        GROUP_BY_NEXT_CLICKS = [
            {'groupby': ['ip']},
            {'groupby': ['ip', 'app']},
            {'groupby': ['ip', 'channel']},
        ]

        for spec in GROUP_BY_NEXT_CLICKS:
            new_feature = '{}_nextClick'.format('_'.join(spec['groupby']))    
            # Unique list of features to select
            list_features = spec['groupby'] + ['click_time']
            td[new_feature] = td[list_features].groupby(spec['groupby']).click_time.                transform(lambda x: x.diff().shift(-1)).dt.seconds.fillna(0)
            gc.collect()
            
    if(HISTORY_CLICKS_FLAG==True):
        
        # add features of prev and next clicks
        HISTORY_CLICKS = {
            'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
            'ip_device_os_clicks': ['ip', 'device', 'os'],
            'app_clicks': ['ip', 'app']
        }

        for fname, fset in HISTORY_CLICKS.items():
            # prev clicks
            td['prev_'+fname] = td.groupby(fset).cumcount().rename('prev_'+fname)
            # next clicks
            td['future_'+fname] = td.iloc[::-1].groupby(fset).cumcount().rename('future_'+fname).iloc[::-1]
        gc.collect()
    del(td['ip'])
    gc.collect()
    
    return td

def convert_preds(raw_preds):
    preds = []
    for p in raw_preds:
        preds.append(1 - p[0])
    return preds

def check_model(model,scaler):
    test_check = load_test_model_data(n_rows_to_test,n_rows_to_train)
    print("test data uploaded")
    test_check = get_features(test_check)
    y_test = test_check['is_attributed']
    y_test = np_utils.to_categorical(y_test, 2)
    
    X_test = test_check.drop('is_attributed', axis=1).select_dtypes(include=[np.number])
    del(test_check)
    gc.collect()
    
    scalercol = [col for col in X_test.columns if col not in (dummycol+notcol)]
    
    for col_title in dummycol:
        dummy = pd.get_dummies(X_test[X_test[col_title].isin(uniq_dict[col_title])][col_title],
                               columns=[col_title], prefix=col_title)
        
        if(len(uniq_dict[col_title]) > dummy.shape[1]):
            for num in uniq_dict[col_title]:
                if('_'.join([col_title,num.astype('str')]) not in dummy.columns):
                    dummy['_'.join([col_title,num.astype('str')])] = 0
        X_test = pd.concat([dummy,X_test],axis=1)
        
        del(dummy)
        gc.collect()
        
    X_test[scalercol] = scaler.transform(X_test[scalercol])
    
    del(scaler)
    gc.collect()
        
    scores_test = model.evaluate(X_test, y_test, batch_size=100, verbose=0)
    print("Result on test data: {:.2f}".format(scores_test[1]*100))
    
    del(y_test,X_test)
    gc.collect()


def make_submission(model_data,scaler):
    test_data = load_test_submission_data()
    print("Submission-data uploaded")
    test_data = get_features(test_data).select_dtypes(include=[np.number])
    
    scalercol = [col for col in test_data.columns if col not in (dummycol+notcol)]
    
    for col_title in dummycol:
        dummy = pd.get_dummies(test_data[test_data[col_title].isin(uniq_dict[col_title])][col_title],
                               columns=[col_title], prefix=col_title)
        if(len(uniq_dict[col_title]) > dummy.shape[1]):
            for num in uniq_dict[col_title]:
                if('_'.join([col_title,num.astype('str')]) not in dummy.columns):
                    dummy['_'.join([col_title,num.astype('str')])] = 0
        test_data = pd.concat([dummy,test_data],axis=1)
        del(dummy)
        gc.collect() 
        
    test_data[scalercol] = scaler.transform(test_data[scalercol])
    submit = pd.read_csv("D:/Kaggle/TalkingData/test.csv", dtype='int', usecols=['click_id'],nrows=500000)
    gc.collect()
    submit['is_attributed'] = convert_preds(model_data.predict_proba(test_data, batch_size=100))
    del(test_data)
    gc.collect()
    submit.to_csv('D:/Kaggle/TalkingData/result_keras.csv', index=False)
    print("submission-file is created")

    
train_data = load_talking_train_big_data(n_rows_to_train)
print("Train data uploaded. Count of lines: {}".format(train_data.shape[0]))
train_data = get_features(train_data)
y = train_data['is_attributed']
y = np_utils.to_categorical(y, 2)
X = train_data.drop('is_attributed', axis=1).select_dtypes(include=[np.number])
del(train_data)
gc.collect()

scalercol = [col for col in X.columns if col not in (dummycol+notcol)]
uniq_dict = {col:[] for col in dummycol}

#add dummy-variables for selected features
for col_title in dummycol:
    uniq_dict[col_title] = pd.unique(X[col_title].values)
    dummy = pd.get_dummies(X[col_title], columns=[col_title], prefix=col_title)
    X = pd.concat([dummy,X],axis=1)
    del(dummy)
    gc.collect()
    
#scale other features
scaler = MinMaxScaler()
X[scalercol] = scaler.fit_transform(X[scalercol])

model = Sequential()
model.add(Dense(100,input_dim=X.shape[1],activation="relu", kernel_initializer="normal"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(2, activation="softmax"))
#sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
ada = optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss="categorical_crossentropy", optimizer=ada, metrics=["accuracy"])

model.fit(X, y, batch_size=100, epochs=3, validation_split=0.15)

print("Model trained")
scores_train = model.evaluate(X, y, batch_size=100, verbose=0)
print("Result on train data: {:.2f}".format(scores_train[1]*100))

del(X,y)
gc.collect()

#check_model(model,scaler)
make_submission(model,scaler)


# In[ ]:




