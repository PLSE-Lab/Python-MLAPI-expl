#!/usr/bin/env python
# coding: utf-8

# ## A simple model - using the market data
# You just can't trust media sources these days.

# ![](https://meme.xyz/uploads/posts/t/l-21806-you-are-fake-news.jpg )

# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


import xgboost as xgb
import numpy as np
import pandas as pd


# In[ ]:


def get_xy(market_train_df):
    x = get_x(market_train_df)
    y = market_train_df['returnsOpenNextMktres10']
    return x, y

def get_x(market_train_df):
    x = market_train_df.drop(columns=['assetCode', 'assetName'])
    try:
        x.drop(columns=['returnsOpenNextMktres10'], inplace=True)
    except:
        pass
    try:
        x.drop(columns=['universe'], inplace=True)
    except:
        pass
    x['dayofweek'], x['month']  = market_train_df.time.dt.dayofweek, market_train_df.time.dt.month
    x.drop(columns='time', inplace=True)
    x.fillna(-1000,inplace=True)
    return x


# In[ ]:


x, y = get_xy(market_train_df)

data = xgb.DMatrix(x, label=y, feature_names=x.columns)

train_cols = x.columns


# In[ ]:


m = xgb.train({}, data, )


# This model gets 0.06071 on public leaderboard

# # Feature Importance
# Let's see what actually makes the difference

# In[ ]:


xgb.plot_importance(m)


# ### It looks like a lot of these are redundant - and the time factors seem pretty unhelpful. So let's make a new model that removes them

# In[ ]:


best_cols = ['volume', 'close', 'open','returnsClosePrevRaw1','returnsClosePrevMktres10']


# In[ ]:


def get_xy(market_train_df):
    x = get_x(market_train_df)
    y = market_train_df['returnsOpenNextMktres10']
    return x, y

def get_x(market_train_df):
    x = market_train_df[best_cols]
    x.fillna(-1000,inplace=True)
    return x


# In[ ]:


x, y = get_xy(market_train_df)

data = xgb.DMatrix(x, label=y, feature_names=x.columns)

train_cols = x.columns


# In[ ]:


m = xgb.train({}, data, )


# In[ ]:


xgb.plot_importance(m)


# In[ ]:


def make_predictions(predictions_template_df, market_obs_df):
    x = get_x(market_obs_df)
    x = x[train_cols]
    data = xgb.DMatrix(x)
    predictions_template_df.confidenceValue = np.clip(m.predict(data),-1,1)


# In[ ]:


days = env.get_prediction_days()

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_predictions(predictions_template_df, market_obs_df)
    env.predict(predictions_template_df)
print('Done!')


# In[ ]:


env.write_submission_file()


# # Submission time!!

# Next up, a result including the news
