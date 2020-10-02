#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time 
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test_id = pd.read_csv('../input/test_id.csv')
print(' shape of train=', train.shape, 'shape of test = ', test_id.shape)


# In[ ]:


def print_time(start, end):
    
    minutes = int((end-start)/60)
    seconds = int(end-start - minutes*60)
    return "%s mins and %s secs"%(minutes,seconds)

start = time.time()

train_not_outlier = train[(np.abs(stats.zscore(train[['MAX_USER','BANDWIDTH_TOTAL']])) < 3).all(axis=1)]
print("delete %s lines from %s lines"%(train.shape[0]-train_not_outlier.shape[0], train.shape[0]))
train_test_df = train_not_outlier.append(test_id,sort=False)
train_test_df = train_test_df.drop(['id'], axis=1)
print(train_test_df.shape)
train_test_df.head(n=5)

def since1(x):
    """
    x is a str of type yyyy-mm-dd
    return the number of day from 2017-10-01"""
    x = x.split('-')
    day_of_year = (int(x[0])-2017)*365.25
    day_of_month = (int(x[1])-10)*365.25/12
    days = (int(x[2])-1)
    return int(day_of_year+day_of_month+days)

def since2(x):
    """
    return the days from the first day of data: 2017-10-01"""
    return (datetime.strptime(x, "%Y-%m-%d")-datetime.strptime('2017-10-01', "%Y-%m-%d")).days

def day_of_week(x):
    """
    x is a str of date, type yyyy-mm-dd
    return if x is a week day or not"""
    day = datetime.strptime(x, "%Y-%m-%d").weekday()
    # day has value from 0 to 6: correspond to monday-sunday
    return day
    
def process_server_name2(name):
    
    keys = name[-6:]
    keys = keys.split('_')
    number = int(keys[1])
    return number

def discretizer_server_number(x):
    """
    x is a server name, which is in fact a number of type abcd where a is 1, 2, or 3
    There are 573 server name, which is a real challenge to make it to one hot vector.
    We will consider it as a continous feature, to discretizer it and then use one hot vector for this 
    new feature"""
    pass


def data_processing(df):
    
    df['year'] = df['UPDATE_TIME'].map(lambda x:int(x.split('-')[0]))
    df['month'] = df['UPDATE_TIME'].map(lambda x:int(x.split('-')[1]))
    df['day'] = df['UPDATE_TIME'].map(lambda x:day_of_week(x))
    df['zone'] = df['ZONE_CODE'].map(lambda x:int(x[-2:]))
    df['server_number1'] = df.apply(lambda x: 0 if x['zone'] !=1 else (process_server_name2(x['SERVER_NAME'])), axis =1)
    df['server_number2'] = df.apply(lambda x: 0 if x['zone']!=2 else (process_server_name2(x['SERVER_NAME'])), axis =1)
    df['server_number3'] = df.apply(lambda x: 0 if x['zone']!=3 else (process_server_name2(x['SERVER_NAME'])), axis =1)
    df['since'] = df['UPDATE_TIME'].map(lambda x: since1(x))
    #max_since = df['since'].max()
    #df['since'] = df['since'].map(lambda x: x/max_since)
    #df['sinceSquared'] = df['since'].map(lambda x: x*x)
    #df['sinceVsMonth'] = df.apply(lambda x: x['since']*x[''])
    #df['hourSquared'] = df['HOUR_ID'].map(lambda x: x*x/(23*23))
    #df['daySquared'] = df['day'].map(lambda x: x*x/36)
    #df['day_hour'] = df.apply(lambda x: x['hourSquared']*x['daySquared'], axis=1)
    print('df shape after processing=', df.shape)
    for var in df:
        print(var, end=' ')
    print()
    print('Done of data processing!')
    return df


def normalization(df):
    """
    df is a data frame from data_processing
    """
    #list_features_to_get_dummies = ['year', 'zone', 'HOUR_ID']
    df['day'] = df['day']/6
    df['HOUR_ID'] = df['HOUR_ID']/23
    df['zone'] = df['zone']/df['zone'].max()
    df['year'] = df['year']/df['year'].max()
    #print('done of creating new features for year, zone  and hour')
    df['month'] = df['month']/12
    #server_cat = pd.cut(df['server_number'], N, labels = ['server_'+str(i) for i in range(N)])
    #dummies = pd.get_dummies(server_cat)
    #df = pd.concat([df, dummies], axis=1, sort=False)
    #server_number_max = df['server_number'].max()
    df['server_number1'] = df['server_number1']/df['server_number1'].max() 
    df['server_number2'] = df['server_number2']/df['server_number2'].max() 
    df['server_number3'] = df['server_number3']/df['server_number3'].max() 

    #df['server_numberSquared'] = df['server_number'].map(lambda x: x*x)
    #print('good for discretizering continous feature')
    #df['is_week_day'] = df['UPDATE_TIME'].map(lambda x: is_week_day(x))
    numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    print('before treatment, shape of df=', df.shape)
    df = df.select_dtypes(include=numerics) # we drop all features that don't have dtypes as above
    # we need to normalize feature here
    print('shape of df=',df.shape)
    #print(df.dtypes)
    return df

def make_interaction(df):
    """
    We create interaction features from existing features
    df is the return of one_hot_features method"""
    product_attributes = [['HOUR_ID', 'HOUR_ID'], ['HOUR_ID', 'year'], ['HOUR_ID', 'month'], 
                      ['HOUR_ID', 'server_number1'], ['HOUR_ID', 'since'], ['year', 'since'], 
                      ['server_number1', 'since'], ['since', 'since']]
    quotes_attributes = [['HOUR_ID', 'HOUR_ID'], ['HOUR_ID', 'year'], ['HOUR_ID', 'month'], ['HOUR_ID', 'day'],
                     ['HOUR_ID', 'zone'], ['HOUR_ID', 'server_number1'], ['HOUR_ID', 'server_number2'], 
                     ['HOUR_ID', 'server_number3'], ['HOUR_ID', 'since'], ['year', 'zone'], ['year', 'server_number2'], 
                     ['server_number1', 'server_number1'], ['since', 'year'], ['since', 'month'], ['since', 'day'], 
                     ['since', 'zone'], ['since', 'server_number2'], ['since', 'server_number3'], ['since', 'since']]
    print('Make interaction between attributes')
    print('before the process, shape of df=', df.shape)
    for attrs in product_attributes:
        var1 = attrs[0]
        var2 = attrs[1]
        product_name = str(var1)+'*'+str(var2)
        df[product_name] = df[var1]*df[var2]
    for attrs in quotes_attributes:
        var1 = attrs[0]
        var2 = attrs[1]
        quote_name = str(var1)+'_OVER_'+str(var2)
        df[quote_name] = df[var1]/(df[var2]+1)
    print('after processing, shape of df=', df.shape)
    print('delete attributes which are too correlated with other...')
    list_attr = list(df.columns)
    list_to_delete = []
    list_done = []
    threshold = 0.975
    for i in range(len(list_attr)-1):
        for j in range(i+1, len(list_attr)):
            var1 = list_attr[i]
            var2 = list_attr[j]
            t = np.corrcoef(df[var1], df[var2])[1][0]
            if t>threshold and (var2 not in list_to_delete):
                    list_to_delete.append(var2)
    print('deleting %s features...'%len(list_to_delete))
    df = df.drop(list_to_delete, axis=1)
    print('after all processing, df has %s features'%df.shape[1])            
            
    return df
            
train_test_df = data_processing(train_test_df)
print('This  first step is ok')
train_test_df = normalization(train_test_df)
train_test_df = make_interaction(train_test_df)
print('This second step is ok')
train_df = train_test_df[:train_not_outlier.shape[0]]
test_df = train_test_df[train_not_outlier.shape[0]:]
test_df = test_df.drop(['BANDWIDTH_TOTAL', 'MAX_USER'], axis =1)


target_bandwidth = train_df.BANDWIDTH_TOTAL
median_band = train_df.BANDWIDTH_TOTAL.median()
target_bandwidth01 = target_bandwidth.map(lambda x: 1 if x>median_band else 0)
target_max_user = train_df.MAX_USER
X_train, X_test, Y_train_bandwidth01, Y_test_bandwidth01, _, Y_test_band= train_test_split(train_df.drop(['BANDWIDTH_TOTAL', 'MAX_USER'], axis=1), target_bandwidth01,                   target_bandwidth, 
                 test_size = 0.3, \
                random_state = 99)
X_test,X_cross,  Y_test_band01, _,_,Y_cross_band = train_test_split(X_test, Y_test_bandwidth01,                                                                                                 Y_test_band,
                                                                                         test_size=0.5,\
                                                                                         random_state=99)

assert X_train.shape[1] == X_test.shape[1] # same number of features
assert X_train.shape[0] == len(Y_train_bandwidth01) # same numbers of rows
#assert X_train.shape[0] == len(Y_train_max_user)
assert X_test.shape[0] == len(Y_test_band01)
#assert X_test.shape[0] == len(Y_test_user)

print('total times=', print_time(start, time.time()))


# # Training

# In[ ]:


start = time.time()
num_boost_round = 15000
params = {}
params['learning_rate']= 0.2
#params['learning_rate']= 0.4
#params['learning_rate'] = 0.5
#params['learning_rate'] = 0.6

params['boosting']='gbdt'
params['objective']='binary'
params['metric']='binary_logloss'
#params['max_leaf'] = 60
#params['lambda_l1'] = 1
#params['lambda_l2'] = 1
params['is_training_metric']= True
params['seed'] = 42

model_bandwidth = lgb.train(params, train_set=lgb.Dataset(X_train, label=Y_train_bandwidth01), num_boost_round= num_boost_round,
                  valid_sets=[lgb.Dataset(X_train, label=Y_train_bandwidth01), lgb.Dataset(X_test, label=Y_test_band01)],
                  verbose_eval=500, early_stopping_rounds=500)
print("training time:", print_time(start, time.time()))


# In[ ]:





# # Evaluate model

# In[ ]:


def mape(a, b): 
    mask = a != 0
    return 100*(np.fabs(a - b)/a)[mask].mean()

threshold = 0.6

#pred_user = model_max_user.predict(X_cross)
pred_band = model_bandwidth.predict(X_cross)

trucate_band = lambda x: 2*median_band if x> threshold else 0
#trucate_user = lambda x: x-20 if x>20 else 0

ban_f = np.vectorize(trucate_band)
#user_f = np.vectorize(trucate_user)

pred_band_trucate = ban_f(pred_band)
#pred_user_trucate = user_f(pred_user)


error_band = np.round(mape(Y_cross_band,pred_band_trucate), 2)
#error_user = np.round(mape(Y_cross_user, pred_user_trucate), 2)
print('BAND error of model binary on test set =%s'%error_band)
#print('error of model binary on test set =%s'%error_user)

#print('error of model binary on test set =%s'%(0.8*error_band+0.2*error_user))
#print('*'*50)


# In[ ]:


print('get some other threshold for band on test set')
threshold_list = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for threshold in threshold_list:
    trucate_band = lambda x: 2*median_band if x> threshold else 0
    ban_f = np.vectorize(trucate_band)
    pred_band_trucate = ban_f(pred_band)
    error_band = mape(Y_cross_band,pred_band_trucate )
    print('With threshold = %s, error =%.2f'%(threshold, error_band))


# # Prediction and save to file

# In[ ]:


sample_submission_df = pd.read_csv('../input/sample_submission.csv')
#test_df = test_df.drop(['BANDWIDTH_TOTAL', 'MAX_USER'], axis=1)
assert sample_submission_df.shape[0] == test_df.shape[0] # for sure that test df have the same lines of sample_submission_df
assert test_df.shape[1] == X_train.shape[1]

predict_band = model_bandwidth.predict(test_df)
assert len(predict_band)== sample_submission_df.shape[0]




sample_submission_df['bandwidth'] = predict_band
sample_submission_df['bandwidth'] = sample_submission_df['bandwidth'].map(lambda x: 2*median_band if x>0.6 else 0)
sample_submission_df['bandwidth'] = sample_submission_df['bandwidth'].map(lambda x: "%.2f" % x)
#sample_submission_df['max_user'] = predict_user
#sample_submission_df['original_band_user'] = sample_submission_df.apply(lambda x: str(x['bandwidth'])+ ' ' + str(x['max_user']), axis=1)
#sample_submission_df['trucate_original_by_median'] = sample_submission_df.apply(lambda x: str(x['bandwidth'])+ ' ' + str(x['max_user']), axis=1)

#sample_submission_df.head()
sample_submission_df['label'] = sample_submission_df['bandwidth']
sample_submission_df = sample_submission_df.drop(['label'], axis=1)
sample_submission_df.head()

sample_submission_df.to_csv(path_or_buf='prediction_V2_3V3.csv',index=False)
prediction_df = pd.read_csv('prediction_V2_3V3.csv')
prediction_df.head()


# ## Save original prediction: of probas from 0 to 1 of band

# In[ ]:


sample_submission_df = pd.read_csv('../input/sample_submission.csv')
#test_df = test_df.drop(['BANDWIDTH_TOTAL', 'MAX_USER'], axis=1)
assert sample_submission_df.shape[0] == test_df.shape[0] # for sure that test df have the same lines of sample_submission_df
assert test_df.shape[1] == X_train.shape[1]

predict_band = model_bandwidth.predict(test_df)
sample_submission_df['bandwidth'] = predict_band
sample_submission_df = sample_submission_df.drop(['label'], axis=1)
sample_submission_df.to_csv(path_or_buf='original_predictionV2_3V3.csv',index=False)
prediction_df = pd.read_csv('original_predictionV2_3V3.csv')
prediction_df.head()

