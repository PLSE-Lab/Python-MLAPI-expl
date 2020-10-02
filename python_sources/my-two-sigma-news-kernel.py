#!/usr/bin/env python
# coding: utf-8

# ```
# from kaggle.competitions import twosigmanews
# env = twosigmanews.make_env()
# 
# (market_train_df, news_train_df) = env.get_training_data()
# train_my_model(market_train_df, news_train_df)
# 
# for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
#   predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
#   env.predict(predictions_df)
#   
# env.write_submission_file()
# ```
# Note that `train_my_model` and `make_my_predictions` are functions you need to write for the above example to work.

# In[ ]:


def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
from kaggle.competitions import twosigmanews
import json
import re
import time
import datetime
import multiprocessing
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

master_s_time = time.time()
env = twosigmanews.make_env()


# In[ ]:


MY_LOG = open('my_log.txt', 'w+')
MY_LOG.write('start_time: {0}'.format(datetime.datetime.now()))
market_train_df, news_train_df = env.get_training_data()
features = ['day', 'totalNumArticles','numAssetArticles',
            'portionOfAssetArticles','relevance','negSentiment',
            'neutralSentiment', 'posSentiment', 'noveltyCount24H',
            'volumeCount24H']
input_features = list(set(features + ['open', 'close', 'returnsOpenPrevMktres1']) - {'day'})


# In[ ]:


MY_MAX_TIME = pd.Timestamp('2007-06-01 22:00:00+0000', tz='UTC')
market_train_df = market_train_df.loc[market_train_df['time'] < MY_MAX_TIME].copy()
news_train_df = news_train_df.loc[news_train_df['time'] < MY_MAX_TIME].copy()


# In[ ]:


def output(s): 
    print(s)
    MY_LOG.write(s)


# In[ ]:


def get_next_trading_date(df, current_date, date_col='time'):
    return df.loc[df[date_col] > current_date, date_col].min()

def get_prev_trading_date(df, current_date, date_col='time'):
    return df.loc[df[date_col] < current_date, date_col].max()


# In[ ]:


def add_day_feature_to_market_data(market_df, asset_codes): 
    market_df['day'] = np.NaN
    market_df.sort_values('time')
    for i, asset_code in enumerate(asset_codes):
        market_df.loc[market_df['assetCode'] == asset_code, 'day'] = list(range(1, len(market_df.loc[market_df['assetCode'] == asset_code]) + 1))
        


# In[ ]:


def add_curr_day_news_features(market_df, news_df, asset_name, ind):
    market_df.loc[ind, 'totalNumArticles'] = len(news_df)
    news_df = news_df.loc[news_df['assetName'] == asset_name].copy()
    market_df.loc[ind, 'numAssetArticles'] = len(news_df)
    if market_df.loc[ind, 'numAssetArticles'] == 0: 
        market_df.loc[ind, ['portionOfAssetArticles',
                            'relevance',
                            'negSentiment',
                            'neutralSentiment', 
                            'posSentiment', 
                            'noveltyCount24H', 
                            'volumeCount24H']] = 0
    else: 
        market_df.loc[ind, 'portionOfAssetArticles'] = market_df.loc[ind, 'numAssetArticles'] / market_df.loc[ind, 'totalNumArticles']
        market_df.loc[ind, 'relevance'] = news_df['relevance'].sum() / market_df.loc[ind,'numAssetArticles']
        market_df.loc[ind, 'negSentiment'] = news_df['sentimentNegative'].sum() / market_df.loc[ind,'numAssetArticles']
        market_df.loc[ind, 'neutralSentiment'] = news_df['sentimentNeutral'].sum() / market_df.loc[ind,'numAssetArticles']
        market_df.loc[ind, 'posSentiment'] = news_df['sentimentPositive'].sum() / market_df.loc[ind, 'numAssetArticles']
        market_df.loc[ind, 'noveltyCount24H'] = news_df['noveltyCount24H'].sum() / market_df.loc[ind,'numAssetArticles']
        market_df.loc[ind, 'volumeCount24H'] = news_df['volumeCounts24H'].sum() / market_df.loc[ind,'numAssetArticles']
                  
def add_news_features(market_df, news_df, asset_codes): 
    s_time = time.time()
    name = multiprocessing.current_process().name
    for i, asset_code in enumerate(asset_codes): 
        if i % 100 == 0: 
            e_time = timedelta(seconds=(time.time() - s_time))
            name = multiprocessing.current_process().name
            print('On assetCode: {0}/{1} | Thread {2}'.format(i,len(asset_codes), name))
        tmp_market_df = market_df.loc[market_df['assetCode'] == asset_code].copy()
        tmp_market_df.sort_values(by=['day'], inplace=True)
        prev_date = news_df['time'].min()
        curr_date = None
        counter = 0
        for ind, row in tmp_market_df.iterrows():
            if row['day'] == 1: 
                prev_date = row['time']
                continue
            else: 
                curr_date = row['time']
                tmp_news_df = news_df.loc[(news_df['time'] > prev_date) &
                                          (news_df['time'] < curr_date)].copy()
                # Assigns values to market_df by ref
                add_curr_day_news_features(market_df, tmp_news_df, row['assetName'], ind)
                prev_date = curr_date


# In[ ]:


def worker(from_date, to_date, min_date, l): 
    multiprocessing.current_process().name += ' |' + str(from_date) + ' - ' + str(to_date) 
    name = multiprocessing.current_process().name
    print('indexing...')
    s_time = time.time()
    tmp_market_df = market_train_df.loc[(market_train_df['time'] >= from_date) &
                                        (market_train_df['time'] < to_date)].copy()
    if from_date > min_date:
        from_date = get_prev_trading_date(market_train_df, from_date)
    tmp_news_df = news_train_df.loc[(news_train_df['time'] >= from_date) &
                                    (news_train_df['time'] < to_date)].copy()
    asset_codes = tmp_market_df['assetCode'].unique().tolist()
    asset_names = tmp_news_df['assetName'].unique().tolist()
    e_time = time.time() - s_time
    print('Took: {0} | thread {1}'.format(str(timedelta(seconds=e_time)), name))
    print('add_news_feature_to_market_data...')
    s_time = time.time()
    add_news_features(tmp_market_df, tmp_news_df, asset_codes)
    e_time = time.time() - s_time
    print('Took: {0} | thread {1}'.format(str(timedelta(seconds=e_time)), name))
    l.append(tmp_market_df)
    print('appended to list')

def callback(tmp_market_df): 
    print('callback fn...')
    name = multiprocessing.current_process().name
    s_time = time.time()
    inds = tmp_market_df.index
    market_train_df.loc[inds, features] = tmp_market_df.loc[inds, features]
    e_time = time.time() - s_time
    print('Took: {0} | thread {1}'.format(str(timedelta(seconds=e_time)), name))
    
def add_features_cols(): 
    for feature in features: 
        market_train_df[feature] = np.NaN
    news_train_df['day'] = np.NaN

def add_features_to_training_data(): 
    MONTHS_RANGE = 1
    NUM_PROCESSES = 4
    with multiprocessing.Manager() as manager:
        min_date = market_train_df['time'].min()
        prev_date = min_date
        curr_date = prev_date + relativedelta(months=MONTHS_RANGE)
        max_date = market_train_df['time'].max()
        p = multiprocessing.Pool(processes=NUM_PROCESSES)
        l = manager.list()
        process_counter = 1
        while prev_date < max_date: 
            p.apply_async(worker, args=(prev_date, curr_date, min_date, l))
            prev_date = curr_date
            curr_date += relativedelta(months=MONTHS_RANGE)
            process_counter += 1
        p.close()
        p.join()
        
        for tmp_market_df in l: 
            callback(tmp_market_df)


# 
# * `day` column is used when calculating 10-day prev market return

# In[ ]:


def add_large_return_features(market_train_df,
                              num_std=1, target='returnsOpenNextMktres10'):
    market_train_df['large_pos_target'] = 0
    market_train_df['large_neg_target'] = 0
    mean = market_train_df[target].mean()
    std = market_train_df[target].std()
    market_train_df.loc[market_train_df[target] > mean + std * num_std, 'large_pos_target'] = 1
    market_train_df.loc[market_train_df[target] < mean - std * num_std, 'large_neg_target'] = 1


# In[ ]:


def add_pos_neg_return_features(market_train_df, target='returnsOpenNextMktres10'):
    market_train_df['pos_target'] = 0
    market_train_df['neg_target'] = 0
    market_train_df.loc[market_train_df[target] > 0, 'pos_target'] = 1
    market_train_df.loc[market_train_df[target] < 0, 'neg_target'] = 1


# In[ ]:


def train_models(enhanced_market_df, asset_codes):
    models = dict.fromkeys(asset_codes)
    for i, asset_code in enumerate(asset_codes):
        if i % 100 == 0: 
            print("Training assetCode: {0}/{1}".format(i, len(asset_codes)))
        X = market_train_df.loc[market_train_df['assetCode'] == asset_code, input_features]
        y_pos = market_train_df.loc[market_train_df['assetCode'] == asset_code, 'pos_target']
        y_neg = market_train_df.loc[market_train_df['assetCode'] == asset_code, 'neg_target']
        if len(y_pos.unique()) < 2: 
            print('{0} has less than 2 y_pos classes '.format(asset_code))
            continue
        if len(y_neg.unique()) < 2: 
            print('{0} has less than 2 y_neg classe'.format(asset_code))
            continue
        pos_return_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                                   multi_class='multinomial', n_jobs=-1).fit(X, y_pos)
        neg_return_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                                   multi_class='multinomial', n_jobs=-1).fit(X, y_neg)
        models[asset_code] = {
            'pos_return_classifier': pos_return_classifier,
            'neg_return_classifier': neg_return_classifier
        }
    return models


# In[ ]:


def fill_market_df(market_df, asset_codes):
    for asset_code in asset_codes: 
        market_train_df.loc[market_train_df['assetCode'] == asset_code,
                            'returnsOpenPrevMktres1'] = market_train_df.loc[market_train_df['assetCode'] == asset_code,
                                                                            'returnsOpenPrevMktres1'].interpolate(method='linear')
    market_train_df.dropna(subset=input_features, inplace=True)
    return market_train_df


# In[ ]:


def fill_and_train_worker(asset_codes, l):
    market_df = market_train_df.loc[market_train_df['assetCode'].isin(asset_codes)].copy()
    filled_market_df = fill_market_df(market_df, asset_codes)
    models = train_models(filled_market_df, asset_codes)
    l.append(models)
    
def fill_and_train_models(market_train_df, asset_codes): 
    div_asset_code = round(len(asset_codes) / 4)
    NUM_PROCESSES = 4
    models = {}
    with multiprocessing.Manager() as manager: 
        p = multiprocessing.Pool(processes=NUM_PROCESSES)
        l = manager.list()
        for i in range(NUM_PROCESSES):
            if i == (NUM_PROCESSES - 1): 
                portion_asset_codes = asset_codes[(i * div_asset_code):]
            else: 
                portion_asset_codes = asset_codes[(i * div_asset_code): ((i+1) * div_asset_code)]
            p.apply_async(fill_and_train_worker, args=(portion_asset_codes, l))
        p.close()
        p.join()
        models = {**l[0], **l[1], **l[2], **l[3]}
        
    return models


# In[ ]:


def train_my_model(market_train_df, news_train_df):
    asset_codes = market_train_df['assetCode'].unique().tolist()
    asset_names = market_train_df['assetName'].unique().tolist()
    add_features_cols()
    
    output('add_day_feature_to_market_data... |  start_time: {0}'.format(datetime.datetime.now()))
    s_time = time.time()
    add_day_feature_to_market_data(market_train_df, asset_codes)
    e_time = time.time() - s_time
    output('Took: {0} to add_day_feature_to_market_data'.format(str(timedelta(seconds=e_time))))
    
    output('add_features_to_market_data... | start_time: {0}'.format(datetime.datetime.now()))
    s_time = time.time()
    add_features_to_training_data()
    e_time = time.time() - s_time
    output('Finished adding features to training data. Took: {0}'.format(str(timedelta(seconds=e_time))))
    output('End time: {0}'.format(datetime.datetime.now()))
    output('add_large_return_features... |  start_time: {0}'.format(datetime.datetime.now()))
    s_time = time.time()
    add_large_return_features(market_train_df, num_std=1)
    e_time = time.time() - s_time
    output('Took: {0} to add large return features'.format(str(timedelta(seconds=e_time))))
    output('add_pos_neg_return_features... |  start_time: {0}'.format(datetime.datetime.now()))
    s_time = time.time()
    add_pos_neg_return_features(market_train_df)
    e_time = time.time() - s_time
    output('Took: {0} to add pos_neg_return features'.format(str(timedelta(seconds=e_time))))
    output('fill_market_df & Train... |  start_time: {0}'.format(datetime.datetime.now()))
    s_time = time.time()
    models = fill_and_train_models(market_train_df, asset_codes)
    #filled_market_df = fill_market_df(market_train_df, asset_codes) # parralelize
    #e_time = time.time() - s_time
    #print('Took: {0}'.format(str(timedelta(seconds=e_time))))
    #print('Train Models...')
    #s_time = time.time()
    #models = train_models(filled_market_df, asset_codes) # parralelize
    e_time = time.time() - s_time
    output('Took: {0} to train models'.format(str(timedelta(seconds=e_time))))
    output('done! train_my_model()')
    return models


# In[ ]:


models = train_my_model(market_train_df, news_train_df)


# In[ ]:


class tmp_classifier():
    def predict_proba(X): 
        return [[0, 0]]


# In[ ]:


for asset_code, classifiers in models.items(): 
    if classifiers is None: 
        models[asset_code] = {
            'pos_return_classifier': tmp_classifier, 
            'neg_return_classifier': tmp_classifier
        }


# In[ ]:


def make_my_predictions(market_obs_df, news_obs_df, predictions_template_df): 
    global market_train_df
    output('On date {0} |  current_time: {1}'.format(market_obs_df['time'].iloc[0], datetime.datetime.now()))
    output('{0} | {1} | {2}'.format(len(predictions_template_df), len(market_obs_df), len(news_obs_df)))
    for feature in features: 
        market_obs_df[feature] = np.NaN
    for ind, row in market_obs_df.iterrows():
        #market_obs_df.loc[ind, 'day'] = market_train_df.loc[market_train_df['assetCode'] == row['assetCode'],
        #                                                    'day'].max() + 1
        add_curr_day_news_features(market_obs_df, news_obs_df, row['assetName'], ind)
        market_obs_df.loc[ind, 'returnsOpenPrevMktres1'] = market_obs_df.loc[ind, 'returnsOpenPrevMktres1'] if not np.isnan(market_obs_df.loc[ind, 'returnsOpenPrevMktres1']) else 0
        X = market_obs_df.loc[[ind], input_features]
        pred_ind = predictions_template_df.loc[predictions_template_df['assetCode'] == row['assetCode']].index
        if row['assetCode'] in models: 
            predictions_template_df.loc[pred_ind, 'confidenceValue'] = models[row['assetCode']]['pos_return_classifier'].predict_proba(X)[0][0]
        else: 
            predictions_template_df.loc[pred_ind, 'confidenceValue'] = 0
    return predictions_template_df
    # when I'm ready to add Moving avgs    
    # market_obs_missing_cols = ['neg_target', 'large_neg_target', 'larget_pos_target', 'pos_target',
    #                           'returnsOpenNextMktres10', 'universe']
    # for missing_col in market_obs_missing_cols: 
    #     market_obs_df[missing_col] = np.NaN
    # market_train_df = pd.concat([market_train_df, market_obs_df])
    
    # change to ffill with prev day data
    # market_obs_df['returnsOpenPrevMktres1'] = market_obs_df['returnsOpenPrevMktres1'].fillna(0)
    # for ind, row in predictions_template_df.iterrows(): 
    #    X = market_obs_df.loc[market_obs_df['assetCode'] == row['assetCode'], input_features]
    #    if row['assetCode'] in models: 
    #        predictions_template_df.loc[ind, 'confidenceValue'] = models[row['assetCode']]['pos_return_classifier'].predict_proba(X)[0][0]
    #    else: 
    #        predictions_template_df.loc[ind, 'confidenceValue'] = 0
    # return predictions_template_df
    


# In[ ]:


days = env.get_prediction_days()
#market_obs_df, news_obs_df, predictions_template_df = next(days)


# In[ ]:


#predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
#predictions_df
#env.predict(predictions_df)


# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
    env.predict(predictions_df)
    
env.write_submission_file()


# In[ ]:


e_time = time.time() - master_s_time
print('Total time: {0}'.format(e_time))
MY_LOG.close()


# In[ ]:




