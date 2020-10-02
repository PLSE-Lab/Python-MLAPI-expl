#!/usr/bin/env python
# coding: utf-8

# ### Xgboost with only few years data
# * Building the model using only data from 2013 onwards, just checking how much the recent years data will help in making the prediction.
# <br/>
#     * **Train:**  2013 to 2015
#     * **Test:** 2016
# 
# * I'm using both marketing and news data, with no major feature engineering.
# * Using Xgboost for modeling, building it as an regression problem. For final submission, converting the final results to +1/-1 based on their score.
# <br/>
# <br/>
# Note: This is just a simple model, there is lot of scope for improvement. This is my first public kernel, share your thoughts in the comments.
# 

# In[ ]:


#### Load packages
import pandas as pd
import numpy as np
import gc

#### Read the training data
from kaggle.competitions import twosigmanews

# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train_df.head()


# Dropping the rows with missing values

# In[ ]:


print(" ---------------- Before removing missing values")
print(market_train_df.isna().sum())

# Remove missing values
market_train_df.dropna(inplace= True)

print(" \n ---------------- After removing missing values ")
print(market_train_df.isna().sum())


# Taking marketing data from 2013 onwards

# In[ ]:


market_train_df_1 = market_train_df[market_train_df.time.dt.year >= 2013]


# In[ ]:


print("No. of observations (2013 to 2016): ", market_train_df_1.shape[0])
print(market_train_df_1.time.dt.year.value_counts())


# In[ ]:


del market_train_df
gc.collect()


# Taking news data from 2013 onwards

# In[ ]:


#Subset news data
news_train_df_1 = news_train_df[news_train_df.time.dt.year >= 2013].copy()


# In[ ]:


del news_train_df
gc.collect()


# In[ ]:


news_train_df_1.time.dt.year.value_counts()


# In[ ]:


news_train_df_1.head()


# In[ ]:


news_train_df_1.info()


# Selecting time, assetName and all the numeric variables

# In[ ]:


news_var = ['time','assetName', 'bodySize','companyCount','sentenceCount','wordCount',
                                  'firstMentionSentence','relevance','sentimentClass','sentimentNegative',
                                   'sentimentNeutral','sentimentPositive','sentimentWordCount','noveltyCount12H',
                                   'noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D',
                                   'volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D'
                                  ]


# In[ ]:


news_train_df_1 = news_train_df_1[news_var]

news_train_df_1['date'] = news_train_df_1.time.dt.date


# In[ ]:


news_train_df_1.groupby(['date','assetName']).size().head(15)


# 
# In a single day there can be multiple articles about an asset, so taking mean values of variables for now

# In[ ]:


#Group to get day & assetName level data 
news_train_df_grp = news_train_df_1.groupby(['date','assetName']).mean().reset_index()


# Merging  market and news data

# In[ ]:


market_train_df_1['date'] = market_train_df_1.time.dt.date

market_train_df_1 = pd.merge(market_train_df_1,news_train_df_grp,how='left',on = ['assetName','date'])
market_train_df_1.head()


# In[ ]:


market_train_df_1.isna().sum()/market_train_df_1.shape[0]


# In[ ]:


#Fill 0 for NA's in News data
market_train_df_1.fillna(0,inplace=True)


# In[ ]:


del news_train_df_grp
gc.collect()


# In[ ]:


#Find the correlations
corr_1 = market_train_df_1.corr()

print(corr_1['returnsOpenNextMktres10'].sort_values(ascending = False))
del corr_1


# In[ ]:


#Removing rows with universe 0
market_train_df_1 = market_train_df_1[market_train_df_1.universe == 1].copy()


# In[ ]:


# Train/test split
id_train = market_train_df_1.time.dt.year != 2016
id_test = market_train_df_1.time.dt.year == 2016

dep_var = 'returnsOpenNextMktres10'
ind_var = ['volume', 'close', 'open', 'returnsClosePrevRaw1',
       'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
       'returnsOpenPrevMktres1', 'returnsClosePrevRaw10',
       'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
       'returnsOpenPrevMktres10','bodySize', 'companyCount', 'sentenceCount', 'wordCount',
       'firstMentionSentence', 'relevance', 'sentimentClass',
       'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
       'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
       'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
       'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
       'volumeCounts7D']

df_train = market_train_df_1.loc[id_train,ind_var]
df_test = market_train_df_1.loc[id_test,ind_var]

print("{0} training rows and {1} testing rows".format(df_train.shape[0],df_test.shape[0]))


y_train = market_train_df_1.loc[id_train,dep_var]
y_test = market_train_df_1.loc[id_test,dep_var]


# In[ ]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error

#-------------- XGboost (untuned)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)

xg_reg.fit(df_train,y_train)


# In[ ]:


import matplotlib.pyplot as plt
xgb.plot_importance(xg_reg,max_num_features = 15)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

pred_train = xg_reg.predict(df_train)
rms_train = sqrt(mean_squared_error(y_train, pred_train))

pred_test = xg_reg.predict(df_test)
rms_test = sqrt(mean_squared_error(y_test, pred_test))

print('Train RMSE: {0} Test RMSE: {1}'.format(rms_train,rms_test))


# Checking the competition metric

# In[ ]:


pred_test_df = market_train_df_1.loc[id_test,['time','assetCode','universe','returnsOpenNextMktres10']]
pred_test_df['dayofyear'] = pred_test_df.time.dt.dayofyear
pred_test_df['confidence'] = [1 if pred >=0 else -1 for pred in pred_test]
pred_test_df['score'] = pred_test_df.universe * pred_test_df.returnsOpenNextMktres10 * pred_test_df.confidence
print(pred_test_df.confidence.value_counts())

score_1 = pred_test_df.groupby(['dayofyear']).score.sum()
score_2 = score_1.mean()/ score_1.std()
print("\n Competition Score: ",np.round(score_2,4))


# #### Submisson creation

# In[ ]:


# You can only iterate through a result from `get_prediction_days()` once 
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()


# In[ ]:


def make_predictions(market_obs_df,news_obs_df,predictions_df,ind_var,news_var,xg_reg):
    
    #Process news data
    news_obs_df = news_obs_df.loc[:,news_var]
    news_obs_df['date'] = news_obs_df.time.dt.date
    news_train_df_grp = news_obs_df.groupby(['date','assetName']).mean().reset_index()
    
    #Merge the market and news data
    market_obs_df['date'] = market_obs_df.time.dt.date
    market_obs_df = pd.merge(market_obs_df,news_train_df_grp,how='left',on = ['assetName','date'])

    #Fill 0 for NA's in News data
    market_obs_df.fillna(0,inplace = True)
    test = market_obs_df.loc[:,ind_var]
    predictions_df.confidenceValue = [1 if  pred >=0 else -1 for pred in xg_reg.predict(test)]


# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_predictions(market_obs_df,news_obs_df,predictions_template_df,ind_var,news_var,xg_reg)
    env.predict(predictions_template_df)
print('Done!')


# Create the submission file

# In[ ]:


env.write_submission_file()

