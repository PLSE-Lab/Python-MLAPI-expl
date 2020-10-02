#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## This is Two Merge Kernel of

# https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data/
# https://www.kaggle.com/the1owl/my-two-sigma-cents-only/


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
import warnings
warnings.simplefilter("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train = market_train_df.tail(2_000_000)
market_train1 = market_train_df.tail(2_000_000)
market_train_df = market_train_df.tail(2_000_000)
news_train = news_train_df.tail(6_000_000)
news_train1 = news_train_df.tail(6_000_000)
news_train_df = news_train_df.tail(6_000_000)


# In[ ]:


def data_prep(market_train1,news_train1):
    market_train1.time = market_train1.time.dt.date
    news_train1.time = news_train1.time.dt.hour
    news_train1.sourceTimestamp= news_train1.sourceTimestamp.dt.hour
    news_train1.firstCreated = news_train1.firstCreated.dt.date
    news_train1['assetCodesLen'] = news_train1['assetCodes'].map(lambda x: len(eval(x)))
    news_train1['assetCodes'] = news_train1['assetCodes'].map(lambda x: list(eval(x))[0])
    kcol = ['firstCreated', 'assetCodes']
    news_train1 = news_train1.groupby(kcol, as_index=False).mean()
    market_train1 = pd.merge(market_train1, news_train1, how='left', left_on=['time', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])
    lbl = {k: v for v, k in enumerate(market_train1['assetCode'].unique())}
    market_train1['assetCodeT'] = market_train1['assetCode'].map(lbl)
    
    
    market_train1 = market_train1.dropna(axis=0)
    
    return market_train1

market_train1 = data_prep(market_train1,news_train1)


# In[ ]:


up = market_train.returnsOpenNextMktres10 >= 0
fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 
                                             'assetName', 'audiences', 'firstCreated', 'headline',
                                             'headlineTag', 'marketCommentary', 'provider', 'returnsOpenNextMktres10',
                                             'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

# We still need the returns for model tuning
X = market_train[fcol].values
up = up.values
r = market_train.returnsOpenNextMktres10.values

# Scaling of X values
# It is good to keep these scaling values for later
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)

# Sanity check
assert X.shape[0] == up.shape[0] == r.shape[0]

from sklearn import model_selection
X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.25, random_state=42)


# In[ ]:


from xgboost import XGBClassifier
import time

xgb_up = XGBClassifier(n_jobs=4,n_estimators=200,max_depth=8,eta=0.1)


# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import accuracy_score\nprint('Fitting Up')\nxgb_up.fit(X_train,up_train,verbose=True)")


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(xgb_up.predict(X_test),up_test)


# In[ ]:


# Feature Engineering

news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean', 'std'],
    'wordCount': ['min', 'max', 'mean', 'std'],
    'sentenceCount': ['min', 'max', 'mean', 'std'],
    'companyCount': ['min', 'max', 'mean', 'std'],
    'marketCommentary': ['min', 'max', 'mean', 'std'],
    'relevance': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
    'sentimentWordCount': ['min', 'max', 'mean', 'std'],
    'noveltyCount12H': ['min', 'max', 'mean', 'std'],
    'noveltyCount24H': ['min', 'max', 'mean', 'std'],
    'noveltyCount3D': ['min', 'max', 'mean', 'std'],
    'noveltyCount5D': ['min', 'max', 'mean', 'std'],
    'noveltyCount7D': ['min', 'max', 'mean', 'std'],
    'volumeCounts12H': ['min', 'max', 'mean', 'std'],
    'volumeCounts24H': ['min', 'max', 'mean', 'std'],
    'volumeCounts3D': ['min', 'max', 'mean', 'std'],
    'volumeCounts5D': ['min', 'max', 'mean', 'std'],
    'volumeCounts7D': ['min', 'max', 'mean', 'std']
}


# # Join Two File

# In[ ]:


def join_market_news(market_train_df, news_train_df):
    # Expand assetCodes
    assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
    assetCodes_index = news_train_df.index.repeat( news_train_df['assetCodes'].apply(len) )

    assert len(assetCodes_index) == len(assetCodes_expanded)
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

    # Create expandaded news (will repeat every assetCodes' row)
    news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
    news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))

    # Free memory
    del news_train_df, df_assetCodes

    # Aggregate numerical news features
    news_train_df_aggregated = news_train_df_expanded.groupby(['time', 'assetCode']).agg(news_cols_agg)
    
    # Convert to float32 to save memory
    news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)

    # Free memory
    del news_train_df_expanded

    # Flat columns
    news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]

    # Join with train
    market_train_df = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])

    # Free memory
    del news_train_df_aggregated
    
    return market_train_df


# In[ ]:


def get_xy(market_train_df, news_train_df, le=None):
    x, le = get_x(market_train_df, news_train_df)
    y = market_train_df['returnsOpenNextMktres10'].clip(-1, 1)
    return x, y, le


def label_encode(series, min_count):
    vc = series.value_counts()
    le = {c:i for i, c in enumerate(vc.index[vc >= min_count])}
    return le


def get_x(market_train_df, news_train_df, le=None):
    # Split date into before and after 22h (the time used in train data)
    # E.g: 2007-03-07 23:26:39+00:00 -> 2007-03-08 00:00:00+00:00 (next day)
    #      2009-02-25 21:00:50+00:00 -> 2009-02-25 00:00:00+00:00 (current day)
    news_train_df['time'] = (news_train_df['time'] - np.timedelta64(22,'h')).dt.ceil('1D')

    # Round time of market_train_df to 0h of curret day
    market_train_df['time'] = market_train_df['time'].dt.floor('1D')

    # Fix asset codes (str -> list)
    news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")    
    
    # Join market and news
    x = join_market_news(market_train_df, news_train_df)
    
    # If not label-encoder... encode assetCode
    if le is None:
        le_assetCode = label_encode(x['assetCode'], min_count=10)
        le_assetName = label_encode(x['assetName'], min_count=5)
    else:
        # 'unpack' label encoders
        le_assetCode, le_assetName = le
        
    x['assetCode'] = x['assetCode'].map(le_assetCode).fillna(-1).astype(int)
    x['assetName'] = x['assetName'].map(le_assetName).fillna(-1).astype(int)
    
    try:
        x.drop(columns=['returnsOpenNextMktres10'], inplace=True)
    except:
        pass
    try:
        x.drop(columns=['universe'], inplace=True)
    except:
        pass
    x['dayofweek'], x['month'] = x.time.dt.dayofweek, x.time.dt.month
    x.drop(columns='time', inplace=True)
#    x.fillna(-1000,inplace=True)

    # Fix some mixed-type columns
    for bogus_col in ['marketCommentary_min', 'marketCommentary_max']:
        x[bogus_col] = x[bogus_col].astype(float)
    
    return x, (le_assetCode, le_assetName)


# In[ ]:


X, y, le = get_xy(market_train_df, news_train_df)
X.shape, y.shape


# In[ ]:


# Save universe data for latter use
universe = market_train_df['universe']
time = market_train_df['time']


# In[ ]:


n_train = int(X.shape[0] * 0.8)

X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_valid, y_valid = X.iloc[n_train:], y.iloc[n_train:]


# In[ ]:


# For valid data, keep only those with universe > 0. This will help calculate the metric
u_valid = (universe.iloc[n_train:] > 0)
t_valid = time.iloc[n_train:]

X_valid = X_valid[u_valid]
y_valid = y_valid[u_valid]
t_valid = t_valid[u_valid]


# In[ ]:


# Creat
train_cols = X.columns.tolist()
categorical_cols = ['assetCode', 'assetName', 'dayofweek', 'month']

# Note: y data is expected to be a pandas Series, as we will use its group_by function in `sigma_score`
dtrain = lgb.Dataset(X_train.values, y_train, feature_name=train_cols, categorical_feature=categorical_cols, free_raw_data=False)
dvalid = lgb.Dataset(X_valid.values, y_valid, feature_name=train_cols, categorical_feature=categorical_cols, free_raw_data=False)


# In[ ]:


# We will 'inject' an extra parameter in order to have access to df_valid['time'] inside sigma_score without globals
dvalid.params = {
    'extra_time': t_valid.factorize()[0]
}


# # Light GBM

# In[ ]:


lgb_params = dict(
    objective = 'regression_l1',
    learning_rate = 0.1,
    num_leaves = 3,
    max_depth = -1,
    min_data_in_leaf = 1000,
#     min_sum_hessian_in_leaf = 1000,
    bagging_fraction = 0.5,
    bagging_freq = 2,
    feature_fraction = 0.75,
    lambda_l1 = 0.0,
    lambda_l2 = 0.0,
    metric = 'None', # This will ignore the loss objetive and use sigma_score instead,
    seed = 42 # Change for better luck! :)
)

def sigma_score(preds, valid_data):
    df_time = valid_data.params['extra_time']
    labels = valid_data.get_label()
    
#    assert len(labels) == len(df_time)

    x_t = preds * labels #  * df_valid['universe'] -> Here we take out the 'universe' term because we already keep only those equals to 1.
    
    # Here we take advantage of the fact that `labels` (used to calculate `x_t`)
    # is a pd.Series and call `group_by`
    x_t_sum = x_t.groupby(df_time).sum()
    score = x_t_sum.mean() / x_t_sum.std()

    return 'sigma_score', score, True

evals_result = {}
m = lgb.train(lgb_params, dtrain, num_boost_round=1000, valid_sets=(dvalid,), valid_names=('valid',), verbose_eval=50,
              early_stopping_rounds=200, feval=sigma_score, evals_result=evals_result)


df_result = pd.DataFrame(evals_result['valid'])


# In[ ]:


ax = df_result.plot(figsize=(12, 8))
ax.scatter(df_result['sigma_score'].idxmax(), df_result['sigma_score'].max(), marker='+', color='red')

num_boost_round, valid_score = df_result['sigma_score'].idxmax()+1, df_result['sigma_score'].max()
print(f'Best score was {valid_score:.5f} on round {num_boost_round}')


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(14, 5))
lgb.plot_importance(m, ax=ax[0])
lgb.plot_importance(m, ax=ax[1], importance_type='gain')
fig.tight_layout()


# In[ ]:


# Train full model
dtrain_full = lgb.Dataset(X, y, feature_name=train_cols, categorical_feature=categorical_cols)
model = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round)


# # ExtraTreeRegressor

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import *
from kaggle.competitions import twosigmanews

col = [c for c in market_train if c not in ['assetCode', 'assetName', 'time', 'returnsOpenNextMktres10', 'universe']]
for c in col:
    market_train[c] = market_train[c].fillna(0.0)
#LabelEncode assetCode, add news based on split assetCodes and dates, aggregate news results
etr = ensemble.ExtraTreesRegressor(n_jobs=-1)
etr.fit(market_train[col], market_train['returnsOpenNextMktres10'])


# In[ ]:


def make_predictions(predictions_template_df, market_obs_df, news_obs_df, le):
    x, _ = get_x(market_obs_df, news_obs_df, le)
    predictions_template_df.confidenceValue = np.clip(model.predict(x), -1, 1)


# In[ ]:


days = env.get_prediction_days()


n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    ########Light GBM
    make_predictions(predictions_template_df, market_obs_df, news_obs_df, le) # LightGBM Prediction Function Submission
    ########### XGB
    market_obs_df1, news_obs_df1, predictions_template_df1 = market_obs_df, news_obs_df, predictions_template_df
     n_days +=1
    print(n_days,end=' ')
    t = time.time()
    market_obs_df1 = data_prep(market_obs_df1, news_obs_df1)
    market_obs_df1 = market_obs_df1[market_obs_df1.assetCode.isin(predictions_template_df1.assetCode)]
    X_live = market_obs_df1[fcol].values
    X_live = 1 - ((maxs - X_live) / rng)
    prep_time += time.time() - t
    
    t = time.time()
    lp = xgb_up.predict_proba(X_live)
    prediction_time += time.time() -t
    t = time.time()
    confidence = 2* lp[:,1] -1
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df1 = predictions_template_df1.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    ########EXTRA TREE REGRESSOR
    for c in col:
        market_obs_df[c] = market_obs_df[c].fillna(0.0)
    market_obs_df['confidenceValue'] = etr.predict(market_obs_df[col]).clip(-1.0, 1.0)
    sub = market_obs_df[['assetCode','confidenceValue']] # Extratreeregressor Function Submission
    predictions_template_df.confidenceValue = predictions_template_df.confidenceValue * 0.34 + sub["confidenceValue"] * 0.33 + predictions_template_df1.confidenceValue * 0.33 # Blending
    env.predict(predictions_template_df)
env.write_submission_file()
print('Done!')

