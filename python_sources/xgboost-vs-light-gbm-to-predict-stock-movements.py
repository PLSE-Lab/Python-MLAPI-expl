#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import os
import time
from datetime import date, datetime
import matplotlib.patches as mpatches
from math import log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb 
import xgboost as xgb 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


from multiprocessing import Pool

def create_lag(df_code, n_lag=[3, 7, 14, ], shift_size=1):
    code = df_code['assetCode'].unique()

    for col in return_features:
        for window in n_lag:
            rolled = df_code[col].shift(shift_size).rolling(window=window)
            lag_mean = rolled.mean()
            lag_max = rolled.max()
            lag_min = rolled.min()
            lag_std = rolled.std()
            df_code['%s_lag_%s_mean' % (col, window)] = lag_mean
            df_code['%s_lag_%s_max' % (col, window)] = lag_max
            df_code['%s_lag_%s_min' % (col, window)] = lag_min

    return df_code.fillna(-1)



def generate_lag_features(df, n_lag=[3, 7, 14]):
    
    
    features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
                'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
                'returnsOpenNextMktres10', 'universe']

    assetCodes = df['assetCode'].unique()
    print(assetCodes)
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time', 'assetCode'] + return_features]
                for df_code in df_codes]
    print('total %s df' % len(df_codes))

    pool = Pool(4)
    all_df = pool.map(create_lag, df_codes)

    new_df = pd.concat(all_df)
    new_df.drop(return_features, axis=1, inplace=True)
    pool.close()

    return new_df

def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

def data_prep(market_train):
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    market_train = market_train.dropna(axis=0)
    return market_train

def exp_loss(p, y):
    y = y.get_label()
    grad = -y * (1.0 - 1.0 / (1.0 + np.exp(-y * p)))
    hess = -(np.exp(y * p) * (y * p - 1) - 1) / ((np.exp(y * p) + 1)**2)
    return grad, hess


# In[ ]:


market_train_df['time'] = market_train_df['time'].dt.date
market_train_df = market_train_df.loc[market_train_df['time'] >= date(
    2010, 1, 1)] #recession data is anomalous

return_features = ['returnsClosePrevMktres10',
                   'returnsClosePrevRaw10', 'open', 'close']
#making lag features
n_lag = [3, 7, 14]
new_df = generate_lag_features(market_train_df, n_lag=n_lag)
market_train_df = pd.merge(market_train_df, new_df,
                           how='left', on=['time', 'assetCode'])

#fixing null values
market_train_df = mis_impute(market_train_df)
#encoding assetCode
market_train_df = data_prep(market_train_df)


# In[ ]:


market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()
market_train_df.sort_values('price_diff')[:10]
#looking at big stock changes


# In[ ]:


#changing outlier features to mean
market_train_df['close_to_open'] =  np.abs(market_train_df['close'] / market_train_df['open'])
market_train_df['assetName_mean_open'] = market_train_df.groupby('assetName')['open'].transform('mean')
market_train_df['assetName_mean_close'] = market_train_df.groupby('assetName')['close'].transform('mean')

for i, row in market_train_df.loc[market_train_df['close_to_open'] >= 2].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['assetName_mean_open']
    else:
        market_train_df.iloc[i,4] = row['assetName_mean_close']
        
for i, row in market_train_df.loc[market_train_df['close_to_open'] <= 0.5].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['assetName_mean_open']
    else:
        market_train_df.iloc[i,4] = row['assetName_mean_close']


# In[ ]:


#more feature engineering
market_train_df['returnsOpenPrevRaw1_to_volume'] = market_train_df['returnsOpenPrevRaw1'] / market_train_df['volume']
market_train_df['close_to_open'] = market_train_df['close'] / market_train_df['open']
news_train_df['sentence_word_count'] =  news_train_df['wordCount'] / news_train_df['sentenceCount']
news_train_df['time'] = news_train_df.time.dt.hour
news_train_df['sourceTimestamp']= news_train_df.sourceTimestamp.dt.hour
news_train_df['firstCreated'] = news_train_df.firstCreated.dt.date
news_train_df['assetCodesLen'] = news_train_df['assetCodes'].map(lambda x: len(eval(x)))
news_train_df['assetCodes'] = news_train_df['assetCodes'].map(lambda x: list(eval(x))[0])
news_train_df['headlineLen'] = news_train_df['headline'].apply(lambda x: len(x))
news_train_df['assetCodesLen'] = news_train_df['assetCodes'].apply(lambda x: len(x))


# In[ ]:


lbl = {k: v for v, k in enumerate(news_train_df['headlineTag'].unique())}
news_train_df['headlineTagT'] = news_train_df['headlineTag'].map(lbl)
kcol = ['firstCreated', 'assetCodes']


# In[ ]:


#merging market and news
news_train_df = news_train_df.groupby(kcol, as_index=False).mean()
agg_market_df = pd.merge(market_train_df, news_train_df, how='left', left_on=['time', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])


# In[ ]:


#freeing memory and dropping na values
del market_train_df
del news_train_df
#lbl = {k: v for v, k in enumerate(market_df['assetCode'].unique())}
#market_df['assetCodeT'] = market_df['assetCode'].map(lbl)
agg_market_df = agg_market_df.dropna(axis=0)


# In[ ]:


#dropping features
agg_market_df.drop(['price_diff', 'assetName_mean_open', 'assetName_mean_close'], axis=1, inplace=True)
up = agg_market_df.returnsOpenNextMktres10 >= 0

fcol = [c for c in agg_market_df.columns if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'assetCodeT',
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider',
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

#making labels
X = agg_market_df[fcol].values
up = up.values
r = agg_market_df.returnsOpenNextMktres10.values

# Scaling of X values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
assert X.shape[0] == up.shape[0] == r.shape[0]


# In[ ]:


#creates train test split with nsamples and specified random state
def sample_ttsplit(nsamples, rng):
    r_sample = r[:nsamples]
    X_sample = X[:nsamples]
    up_sample = up[:nsamples]

    X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X_sample, up_sample, r_sample, random_state=rng)
    return X_train, X_test, up_train, up_test, r_train, r_test


# In[ ]:


#returns execution time, accuracy, roc, and the model itself for either lgbm or xgbm
def boost_test(X_train, X_test, y_train, y_test, xg, model, params):
    #The data is stored in a DMatrix object 
    #label is used to define our outcome variable
    if xg == True:
        dtrain=model.DMatrix(X_train,label=y_train, feature_names=agg_market_df[fcol].columns)
        dtest=model.DMatrix(X_test, feature_names=agg_market_df[fcol].columns)
    else:
        dtrain=model.Dataset(X_train,label=y_train, feature_name=list(agg_market_df[fcol].columns))

    num_round=50
    start = datetime.now() 
    fit=model.train(params,dtrain,num_round) 
    stop = datetime.now()
    #Execution time of the model 
    execution_time = stop-start 
    
    if xg == True:
        ypred=fit.predict(dtest) 
    else:
        ypred=fit.predict(X_test)
    
    max = len(ypred)
    for i in range(0,max): 
        if ypred[i]>=.5:       # setting threshold to .5 
           ypred[i]= True 
        else: 
           ypred[i]= False 

    accuracy = accuracy_score(y_test,ypred) 
    auc =  roc_auc_score(y_test,ypred)
    return execution_time, accuracy, auc, fit


# In[ ]:


X_train, X_test, up_train, up_test, r_train, r_test = sample_ttsplit(100, 99)


# In[ ]:


#small model for plotting
params = {'max_depth':7, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
a, b, c, xgbfit = boost_test(X_train, X_test, up_train, up_test, True, xgb, params)


# In[ ]:


#small model for plotting
param = {'num_leaves':150, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']
a, b, c, lgbfit = boost_test(X_train, X_test, up_train, up_test, False, lgb, param)


# In[ ]:


#lgb tree
plt.figure()
lgb.plotting.plot_tree(lgbfit, figsize=(100, 100))
fig = plt.gcf()
fig.set_size_inches(150, 100)
plt.show()


# In[ ]:


#xgb tree
plt.figure()
xgb.plot_tree(xgbfit)
fig = plt.gcf()
fig.set_size_inches(150, 100)
plt.show()


# In[ ]:


#computing accuracy, roc, execution time for several max_depths, cross validation included
score_dict = {}
for md in [4, 7, 11, 15, 19]:
    xgb_time, xgb_acc, xgb_auc = 0, 0, 0
    lgb_time, lgb_acc, lgb_auc = 0, 0, 0
    for i in [14, 50, 80]:
        X_train, X_test, up_train, up_test, r_train, r_test = sample_ttsplit(10000, i)
        lgbparams = {'num_leaves':2**(4*log(md)), 'objective':'binary','max_depth':md,'learning_rate':.05,'max_bin':200}
        lgbparams['metric'] = ['auc', 'binary_logloss']
        xgbparams = {'max_depth':md, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
        xgbtime, xgbacc, xgbauc, xgbfit = boost_test(X_train, X_test, up_train, up_test, True, xgb, params)
        lgbtime, lgbacc, lgbauc, lgbfit = boost_test(X_train, X_test, up_train, up_test, False, lgb, param)
        xgb_time += xgbtime.total_seconds()
        xgb_acc += xgbacc
        xgb_auc += xgbauc
        lgb_time += lgbtime.total_seconds()
        lgb_acc += lgbacc
        lgb_auc += lgbauc
    xgb_time /= 3
    xgb_acc /= 3
    xgb_auc /= 3
    lgb_time /= 3
    lgb_acc /= 3
    lgb_auc /= 3
    score_dict[md] = {}
    score_dict[md]['xgb'] = [xgb_time, xgb_acc, xgb_auc]
    score_dict[md]['lgb'] = [lgb_time, lgb_acc, lgb_auc]

score_dict


# In[ ]:


plt.figure()
for (i, col) in enumerate(['r', 'g']):
    xgb_vec = [score_dict[4]['xgb'][i + 1], score_dict[7]['xgb'][i + 1], score_dict[11]['xgb'][i + 1], score_dict[15]['xgb'][i+1], score_dict[19]['xgb'][i+1]]
    lgb_vec = [score_dict[4]['lgb'][i + 1], score_dict[7]['lgb'][i + 1], score_dict[11]['lgb'][i + 1], score_dict[15]['lgb'][i+1], score_dict[19]['lgb'][i+1]]
    plt.plot([4, 7, 11, 15, 19], xgb_vec, color = col)
    plt.plot([4, 7, 11, 15, 19], lgb_vec, color = col, linestyle = '--')
plt.xlabel("max depth")
plt.title("max depth vs accuracy and roc_auc score (10000 samples)")
plt.ylabel("score")
plt.legend(['xgb roc', 'lgb roc', 'xgb accuracy', 'lgb accuracy'])
plt.show()

plt.figure()
xgb_vec = [score_dict[4]['xgb'][0], score_dict[7]['xgb'][0], score_dict[11]['xgb'][0], score_dict[15]['xgb'][0], score_dict[19]['xgb'][0]]
lgb_vec = [score_dict[4]['lgb'][0], score_dict[7]['lgb'][0], score_dict[11]['lgb'][0], score_dict[15]['lgb'][0], score_dict[19]['lgb'][0]]
plt.plot([4, 7, 11, 15, 19], xgb_vec, color = 'r')
plt.plot([4, 7, 11, 15, 19], lgb_vec, color = 'g')
plt.xlabel("max depth")
plt.title("max depth vs computational time (10000 samples)")
plt.ylabel("seconds")
plt.legend(['xgb', 'lgb'])
plt.show()


# In[ ]:


#a larger sample split
X_train, X_test, up_train, up_test, r_train, r_test = sample_ttsplit(200000, i)
md = 10
lgbparams = {'num_leaves':2**(4*log(md)), 'objective':'binary','max_depth':md,'learning_rate':.05,'max_bin':200}
lgbparams['metric'] = ['auc', 'binary_logloss']
xgbparams = {'max_depth':md, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
xgbtime, xgbacc, xgbauc, xgbfit = boost_test(X_train, X_test, up_train, up_test, True, xgb, params)
lgbtime, lgbacc, lgbauc, lgbfit = boost_test(X_train, X_test, up_train, up_test, False, lgb, param)


# In[ ]:


#xgb feature importances
X_importance = X_test

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(xgbfit)
shap_values = explainer.shap_values(pd.DataFrame(X_importance, columns = agg_market_df[fcol].columns))
shap.summary_plot(shap_values, X_importance, agg_market_df[fcol].columns)
shap.summary_plot(shap_values, X_importance, agg_market_df[fcol].columns, plot_type='bar')


# In[ ]:


#lgb feature importances
X_importance = X_test

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(lgbfit)
shap_values = explainer.shap_values(X_importance)
shap.summary_plot(shap_values, X_importance, agg_market_df[fcol].columns)
shap.summary_plot(shap_values, X_importance, agg_market_df[fcol].columns, plot_type='bar')

