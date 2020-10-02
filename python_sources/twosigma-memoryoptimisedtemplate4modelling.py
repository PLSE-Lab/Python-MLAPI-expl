#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, time, os, gc
from datetime import datetime, date 

from sklearn import *
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import mean_squared_error
#from sklearn import model_selection

from multiprocessing import Pool
import seaborn as sns, matplotlib, matplotlib.pyplot as plt
#print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


from kaggle.competitions import twosigmanews
if 'env' not in globals():
    env = twosigmanews.make_env()


# # Script Parameters

# In[ ]:


inputMarketObservationFilter = ["volume", "close", "open", 
                                "returnsClosePrevRaw1", "returnsOpenPrevRaw1", 
                                "returnsClosePrevMktres1", "returnsOpenPrevMktres1", 
                                "returnsClosePrevRaw10", "returnsOpenPrevRaw10", 
                                "returnsClosePrevMktres10", "returnsOpenPrevMktres10"]

inputNewsObservationFilter = ["relevance", "sentimentNegative", "sentimentNeutral", "sentimentPositive"]

paramLagFeatures = ['returnsClosePrevMktres10','returnsClosePrevMktres1']
paramLagFrequencies = [3,7,14]


# # Preload Functions
# Purpose: Save Memory

# In[ ]:


def PreloadMarketTrainingRaw():
    mandatoryColumns =  ["time", "assetCode", "universe", 'returnsOpenNextMktres10']
    returnColumns = mandatoryColumns + inputMarketObservationFilter
    marketRaw = env.get_training_data()[0][returnColumns]

    #marketRaw['time'] = marketRaw.time.dt.date
    marketRaw['volume'] = pd.to_numeric(marketRaw.volume, errors='coerce', downcast='integer')
    marketRaw['universe'] = pd.to_numeric(marketRaw.universe, errors='coerce', downcast='integer')
    real = {c: 'float16' for c in marketRaw.columns if c not in ['assetCode', 'time', "volume", "universe"]}
    return marketRaw.astype(real)

def PreloadNewsTrainingRaw():
    mandatoryColumns =  ["time", "assetCodes"]
    returnColumns = mandatoryColumns + inputNewsObservationFilter
    newsRaw = env.get_training_data()[1][returnColumns]
    #newsRaw['time'] = newsRaw.time.dt.date
    return newsRaw


# ## TEST
# nraw = PreloadNewsTrainingRaw()
# nraw.head()
# nraw.info()
# mraw = PreloadMarketTrainingRaw()
# mraw.head()
# mraw.info()

# # Data Preparation Functions for Training

# ## Market Processing

# In[ ]:


def ConsolidateMarket(inputMarket):
    mandatoryColumns =  ["time", "assetCode"]
    cols = mandatoryColumns + inputMarketObservationFilter
    
    # append target when available
    if 'returnsOpenNextMktres10' in inputMarket.columns:
        cols = cols + ['returnsOpenNextMktres10']
    
    output = inputMarket[cols].reset_index()
    output['time'] = output.time.dt.date
    
    output['returnsClose'] = (output['close'] / output['open'])-1
    #output['volume'] = pd.to_numeric(output.volume, errors='coerce', downcast='integer')
    #output['returnsClose'] = pd.to_numeric(output.returnsClose, errors='coerce', downcast='float')
    output.dropna(axis=0, inplace=True)
   
    dropColumns = ["close", "open", "index"]
    output.drop(dropColumns, axis=1, inplace=True)

    aggregations = ['mean']
    gp = output.groupby(['assetCode', 'time']).agg(aggregations)
    gp.columns = pd.Index(["{}".format(e[0]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    
    real = {c: 'float16' for c in output.columns if c not in ['assetCode', 'time', "universe", "volume"]}
    return gp.astype(real)


# ### TEST
# 
# cMarket = ConsolidateMarket(PreloadMarketTrainingRaw())
# cMarket.head()
# cMarket.info()
# 

# ## NEWS Processing

# In[ ]:


# helper function to decompose assetcodes
def MetaBuildNewsAssetCodeIndex(inputNews):
    codes = []
    indexes = []
    for i, values in inputNews['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    output = pd.DataFrame({'ID': indexes, 'assetCode': codes})
    output["ID"] = pd.to_numeric(output["ID"], errors='coerce', downcast='integer')
    del codes, indexes
    gc.collect()
    return output


# denormalising assetcodes into assetCode column which serves as foreign key to market assetCode
def MetaBuildIndexedNews(inputNews):
    inputNews['ID'] = inputNews.index.copy()
    inputNews['assetCodes'] = inputNews['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    # Merge news on unstacked assets
    output = MetaBuildNewsAssetCodeIndex(inputNews).merge(inputNews, how='left', on='ID')
    output.drop(['ID', 'assetCodes'], axis=1, inplace=True)
    return output


## Comine multiple news reports for same assets on same day.
def MetaGroupByDay(inputNews):
    aggregations = ['mean']
    gp = inputNews.groupby(['assetCode', 'time']).agg(aggregations)
    gp.columns = pd.Index(["{}".format(e[0]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    # Set datatype to float16
    real = {c: 'float16' for c in gp.columns if c not in ['assetCode', 'time', 'volume']}
    return gp.astype(real)


# ### Test
# ac = MetaBuildNewsAssetCodeIndex(cNews)
# ac.head()
# ac.info()
# 
# acidx = MetaBuildNewsAssetCodeIndex(cNews)
# acidx
# 
# newsByDay = MetaGroupByDay(cMarket)
# newsByDay.tail()
# newsByDay.info()

# In[ ]:


# putting it all together
def ConsolidateNews(inputNews):
    mandatoryColumns =  ["time", "assetCodes"]
    cols = mandatoryColumns + inputNewsObservationFilter
    
    output = inputNews[cols].reset_index()
    output['time'] = output.time.dt.date
    
    output["SentimentCoefficient"] = (output.sentimentPositive - output.sentimentNegative) * (1-output.sentimentNeutral)

    
    dropColumns = ["index", "sentimentPositive", "sentimentNegative", "sentimentNeutral"]
    output.drop(dropColumns, axis=1, inplace=True)
    
    idxn = MetaBuildIndexedNews(output)
    output = MetaGroupByDay(idxn)
    
    return output


# ### Test
# cNews = ConsolidateNews(PreloadNewsTrainingRaw())
# cNews.head()
# cNews.info()
# del cMarket, cNews
# gc.collect()

# In[ ]:


def ConsolidateMarketNews(inputMarketRaw, inputNewsRaw):   
        return ConsolidateMarket(inputMarketRaw).merge(ConsolidateNews(inputNewsRaw), how='left', on=['time','assetCode']).fillna(0)


# ## Test
# marketNews = ConsolidateMarketNews(PreloadMarketTrainingRaw(), PreloadNewsTrainingRaw())
# marketNews.head()
# marketNews.info()
# 

# # Training

# ## Ingest Variations

# ### Variation 1: template processing using Sentiment
# 
# def V1Ingest(inputMarketRaw,inputNewsRaw):
#     cmn = ConsolidateMarketNews(inputMarketRaw, inputNewsRaw)
#     
#     ## enter transformationlogic here 
#     
#     ## adjust derived features to transformation logic
#     derivedFeatures = ['returnsClose', 'returnsOpenPrevMktres10', 'returnsClosePrevMktres10', 'SentimentCoefficient']
#     
#     outputFeatureSet = cmn[derivedFeatures]
#     outputTag = [] # target label for training, assetCode for prediction
#     
#     if 'returnsOpenNextMktres10' in cmn.columns:
#         outputTag = (cmn.returnsOpenNextMktres10 >= 0).astype('int8')
#     else:
#         outputTag = cmn.assetCode
#     
#     return outputFeatureSet, outputTag
# 
# # Training
# featureSet, target = V1Ingest(PreloadMarketTrainingRaw(), PreloadNewsTrainingRaw())
# 
# #featureSet.head()
# #target.describe()
# ## best parameters found.
# V1Model = LGBMClassifier(
#     objective='binary',
#     boosting='gbdt',
#     learning_rate = 0.05,
#     max_depth = 8,
#     num_leaves = 80,
#     n_estimators = 400,
#     bagging_fraction = 0.8,
#     feature_fraction = 0.9)
#     #reg_alpha = 0.2,
#     #reg_lambda = 0.4)
#     
# V1Model.fit(featureSet, target)

# In[ ]:


### Variation 2: template processing using Sentiment

def V2Ingest(inputMarketRaw,inputNewsRaw):
    cmn = ConsolidateMarketNews(inputMarketRaw, inputNewsRaw)
    
    ## enter transformationlogic here 
    cmn["SentimentWeighted"] = cmn.relevance * cmn.SentimentCoefficient
    ## adjust derived features to transformation logic
    
    derivedFeatures = ['returnsClose', 'returnsOpenPrevMktres10', 'SentimentWeighted']
    
    outputFeatureSet = cmn[derivedFeatures]
    outputTag = [] # target label for training, assetCode for prediction
    
    if 'returnsOpenNextMktres10' in cmn.columns:
        outputTag = (cmn.returnsOpenNextMktres10 >= 0).astype('int8')
    else:
        outputTag = cmn.assetCode
    
    return outputFeatureSet, outputTag

# Training
featureSet, target = V2Ingest(PreloadMarketTrainingRaw(), PreloadNewsTrainingRaw())

#featureSet.head()
#target.describe()
## best parameters found.
V2Model = LGBMClassifier(
    objective='binary',
    boosting='gbdt',
    learning_rate = 0.05,
    max_depth = 8,
    num_leaves = 80,
    n_estimators = 400,
    bagging_fraction = 0.8,
    feature_fraction = 0.9)
    #reg_alpha = 0.2,
    #reg_lambda = 0.4)
    
V2Model.fit(featureSet, target)


# # Prediction
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "days = env.get_prediction_days()\nn_days = 0\n\nfor (market_obs_df, news_obs_df, predictions_template_df) in days:\n    n_days += 1\n    print(n_days,end=' ')\n    \n    # adjust block to select VxIngest and models\n    featureSet, assetCode = V2Ingest(market_obs_df, news_obs_df)\n    preds = V2Model.predict_proba(featureSet)[:, 1] * 2 - 1\n    \n    sub = pd.DataFrame({'assetCode': assetCode, 'confidence': preds})\n    predictions_template_df = predictions_template_df.merge(sub, how='left').drop(\n        'confidenceValue', axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})\n\n    env.predict(predictions_template_df)\n\nprint('Prediction Complete!')")


# In[ ]:


env.write_submission_file()
print("submission written")

