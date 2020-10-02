#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import *
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train, news_train) = env.get_training_data()


# In[ ]:


market_train.time = market_train.time.dt.date
news_train.time = news_train.time.dt.hour
news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
news_train.firstCreated = news_train.firstCreated.dt.date
news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))
news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
kcol = ['firstCreated', 'assetCodes']
news_train = news_train.groupby(kcol, as_index=False).mean()
market_train = pd.merge(market_train, news_train, how='left', left_on=['time', 'assetCode'], right_on=['firstCreated', 'assetCodes'])
lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe']]
#for c in fcol:
    #market_train[c] = market_train[c].fillna(0.0)


# In[ ]:


#Add a Metric Function
#etr = ensemble.ExtraTreesRegressor(n_jobs=-1)
#etr.fit(market_train[fcol], market_train['returnsOpenNextMktres10'])
#print(metrics.mean_squared_error(market_train['returnsOpenNextMktres10'], etr.predict(market_train[fcol])))


# In[ ]:


import lightgbm as lgb

x1, x2, y1, y2 = model_selection.train_test_split(market_train[fcol], market_train['returnsOpenNextMktres10'], test_size=0.25, random_state=99)

def lgb_rmse(preds, y): #Update to Competition Metric
    y = np.array(list(y.get_label()))
    score = np.sqrt(metrics.mean_squared_error(y, preds))
    return 'RMSE', score, False

params = {'learning_rate': 0.2, 'max_depth': 6, 'boosting': 'gbdt', 'objective': 'regression', 'seed': 2018}
lgb_model = lgb.train(params, lgb.Dataset(x1, label=y1), 500, lgb.Dataset(x2, label=y2), verbose_eval=10, early_stopping_rounds=20) #feval=lgb_rmse


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({'imp': lgb_model.feature_importance(importance_type='gain'), 'col':fcol})
df = df.sort_values(['imp','col'], ascending=[True, False])
_ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))
#plt.savefig('lgb_gain.png')

df = pd.DataFrame({'imp': lgb_model.feature_importance(importance_type='split'), 'col':fcol})
df = df.sort_values(['imp','col'], ascending=[True, False])
_ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))
#plt.savefig('lgb_split.png')


# In[ ]:


for (market_test, news_test, sub) in env.get_prediction_days():
    market_test.time = market_test.time.dt.date
    news_test.time = news_test.time.dt.hour
    news_test.sourceTimestamp= news_test.sourceTimestamp.dt.hour
    news_test.firstCreated = news_test.firstCreated.dt.date
    news_test['assetCodesLen'] = news_test['assetCodes'].map(lambda x: len(eval(x)))
    news_test['assetCodes'] = news_test['assetCodes'].map(lambda x: list(eval(x))[0])
    kcol = ['firstCreated', 'assetCodes']
    col = [c for c in news_test.columns if c not in ['sourceId', 'headline', 'provider', 'subjects', 'audiences', 'headlineTag', 'marketCommentary', 'assetName', 'assetCodesLen']]
    news_test = news_test.groupby(kcol, as_index=False).mean()
    market_test = pd.merge(market_test, news_test, how='left', left_on=['time', 'assetCode'], right_on=['firstCreated', 'assetCodes'])
    market_test['assetCodeT'] = market_test['assetCode'].map(lambda x: lbl[x] if x in lbl else 0)
    fcol = [c for c in market_test if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe']]
    #for c in fcol:
        #market_test[c] = market_test[c].fillna(0.0)
    #market_test['confidenceValue'] = etr.predict(market_test[fcol]).clip(-1.0, 1.0)
    market_test['confidenceValue'] = lgb_model.predict(market_test[fcol], num_iteration=lgb_model.best_iteration).clip(-1.0, 1.0)
    sub = market_test[['assetCode','confidenceValue']]
    env.predict(sub)
env.write_submission_file()

