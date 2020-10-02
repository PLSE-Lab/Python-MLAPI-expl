#!/usr/bin/env python
# coding: utf-8

# # Rolling with the Features
# 
# In a [previous kernel](https://www.kaggle.com/donkeys/quest-for-market-data-outliers-meaning), I explored the outliers in the market data, and the meaning of all those Mktres columns. As I could not quite figure what these features mean, I wanted to try to calculate my own.
# 
# A slightly more interesting twist is that not all data is accessible at once. Rather, the training data is given, but for the test data, one has to read a line at a time and any custom features need to be build based on having access to the new data one line at a time.
# 
# Here we go.

# ## Given description of the market data
# 
# The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:
# 
# * Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
# * Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.
# * Returns can be calculated over any arbitrary interval. Provided here are 1 day and 10 day horizons.
# * Returns are tagged with 'Prev' if they are backwards looking in time, or 'Next' if forwards looking.
# 

# ## Market data columns
# 
# Within the marketdata, you will find the following columns:
# 
# * __time(datetime64[ns, UTC])__ - the current time (in marketdata, all rows are taken at 22:00 UTC)
# * __assetCode(object)__ - a unique id of an asset
# * __assetName(category)__ - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
# * __universe(float64)__ - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
# * __volume(float64)__ - trading volume in shares for the day
# * __close(float64)__ - the close price for the day (not adjusted for splits or dividends)
# * __open(float64)__ - the open price for the day (not adjusted for splits or dividends)
# * __returnsClosePrevRaw1(float64)__ - see returns explanation above
# * __returnsOpenPrevRaw1(float64)__ - see returns explanation above
# * __returnsClosePrevMktres1(float64)__ - see returns explanation above
# * __returnsOpenPrevMktres1(float64)__ - see returns explanation above
# * __returnsClosePrevRaw10(float64)__ - see returns explanation above
# * __returnsOpenPrevRaw10(float64)__ - see returns explanation above
# * __returnsClosePrevMktres10(float64)__ - see returns explanation above
# * __returnsOpenPrevMktres10(float64)__ - see returns explanation above
# * __returnsOpenNextMktres10(float64)__ - 10 day, market-residualized return. This is the target variable used in competition scoring. The market data has been filtered such that returnsOpenNextMktres10 is always not null.

# # The code

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


from kaggle.competitions import twosigmanews
#   You  can  only    call    make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


market_df, news_df = env.get_training_data()


# In[ ]:


market_df.head()


# In[ ]:


market_df.describe()


# In[ ]:


market_df.isnull().sum()


# In[ ]:


market_df.nunique()


# The following drops the outliers identified in my [previous kernel](https://www.kaggle.com/donkeys/quest-for-market-data-outliers-meaning). But it only drops the outlier row, leaving all the tails it causes in market residual and raw columns that reflect back in time. Which is why I wanted to recalculate my own versions to get rid of all those outlier impacts.

# In[ ]:


outlier_indices = [3845015, 3845309, 3845467, 3845835, 3846067, 3846276, 3846636, 
                   50031, 92477, 206676, 459234, 132779, 50374, 276388, 3845946, 
                   616236, 3846151, 49062, 588960, 165718, 25574, 555738, 56387, 
                   1127598, 49050, 50332, 49850, 49531, 627577, 503021, 520681, 
                   471405, 242868, 3264667, 120158, 617101, 133218, 132360, 132809, 
                   133089, 133602, 7273, 194569, 133345, 459489, 132386, 132342, 
                   551857, 133587, 132817, 88284, 133478, 549498, 132821, 193015, 177298]
market_df.drop(outlier_indices, inplace=True)


# From all the data, just pick the columns needed to build my own features:

# In[ ]:


mt_df = market_df[["time", "assetCode", "assetName", "volume", "open", "close"]]#, "open_change_p1"]]


# Pull out the prediction target so I can drop the rest of the original dataframe and save memory:

# In[ ]:


#possibly another target could be made from self-created features, but for now..
y = market_df["returnsOpenNextMktres10"]
up = market_df["returnsOpenNextMktres10"] > 0


# Now delete the original dataframe to save memory:

# In[ ]:


del market_df


# Verify it looks good:

# In[ ]:


mt_df.head()


# In[ ]:


mt_df.columns.values


# Preprocess dataframes to add my own features for percent change in one day and percent change over 10 days. And 10 day exponential weighted moving average.

# In[ ]:


def preprocess(df):
    #quiet the setting with copy warning for this method or the log will be full of it
    #since this is called for every row in test set
    orig_setting = pd.options.mode.chained_assignment
    
    pd.options.mode.chained_assignment = None
    #mt_df.sort_values("assetCode").groupby('assetCode', as_index=False).pct_change().head(10)
    #needs apply due to pandas bug in 23.0: https://github.com/pandas-dev/pandas/issues/21200
    #percentage change over 1 day
    df['pct_chg_open1'] = df.groupby('assetCode')['open'].apply(lambda x: x.pct_change())
    df['pct_chg_close1'] = df.groupby('assetCode')['close'].apply(lambda x: x.pct_change())
    #percentage change over 10 days
    df['pct_chg_open10'] = df.groupby('assetCode')['open'].apply(lambda x: x.pct_change(periods=10))
    df['pct_chg_close10'] = df.groupby('assetCode')['close'].apply(lambda x: x.pct_change(periods=10))
    #https://stackoverflow.com/questions/37924377/does-pandas-calculate-ewm-wrong?noredirect=1&lq=1

    #if running in rolling model, drop the old average columns to create new ones.
    #otherwise it will cause naming issues with overlapping column names..
    if 'avg_pct_open1' in df.columns:
        #if there is one, there are all
        df.drop('avg_pct_open1', axis=1, inplace=True)
        df.drop('avg_pct_close1', axis=1, inplace=True)
        df.drop('avg_pct_open10', axis=1, inplace=True)
        df.drop('avg_pct_close10', axis=1, inplace=True)

    #average changes over 1 and 10 days
    avg_open1 = df.groupby('time')['pct_chg_open1'].mean().reset_index()
    avg_open1.rename(columns={'pct_chg_open1': 'avg_pct_open1'}, inplace=True)
    df = df.merge(avg_open1, how='left', on=['time'])

    avg_close1 = df.groupby('time')['pct_chg_close1'].mean().reset_index()
    avg_close1.rename(columns={'pct_chg_close1': 'avg_pct_close1'}, inplace=True)
    df = df.merge(avg_close1, how='left', on=['time'])

    avg_open10 = df.groupby('time')['pct_chg_open10'].mean().reset_index()
    avg_open10.rename(columns={'pct_chg_open10': 'avg_pct_open10'}, inplace=True)
    df = df.merge(avg_open10, how='left', on=['time'])
    
    avg_close10 = df.groupby('time')['pct_chg_close10'].mean().reset_index()
    avg_close10.rename(columns={'pct_chg_close10': 'avg_pct_close10'}, inplace=True)
    df = df.merge(avg_close10, how='left', on=['time'])

    df["close_ewma_10"] = df.groupby('assetName')['pct_chg_close1'].apply(lambda x: x.ewm(span=10).mean())
    df["open_ewma_10"] = df.groupby('assetName')['pct_chg_open1'].apply(lambda x: x.ewm(span=10).mean())
    
    pd.options.mode.chained_assignment = orig_setting

    return df


# Process the initial training data:

# In[ ]:


mt_df = preprocess(mt_df)


# In[ ]:


mt_df[mt_df["assetCode"]=="A.N"].head(11)


# In[ ]:


X_cols = [col for col in mt_df.columns if col not in ["time", "assetCode", "assetName"]]
X_cols


# In[ ]:


X_cols.append("time")
X = mt_df[X_cols]


# The usual train-test split. Not sure if time should be counted or not. Since the features are on a single line, maybe not. For other types of models more likely so.

# In[ ]:


from sklearn import model_selection

X_train, X_test, y_train, y_test, u_train, u_test = model_selection.train_test_split(X, y, up, test_size=0.25, random_state=99)


# In[ ]:


y.head()


# In[ ]:


y = (y > 0).astype(int)


# In[ ]:


y.head()


# In[ ]:


X_train


# In[ ]:


X_train.drop("time", axis=1, inplace=True)
X_test.drop("time", axis=1, inplace=True)


# In[ ]:


print(X_train.shape, X_test.shape)
print(u_train.shape, u_test.shape)


# In[ ]:


X_cols = [col for col in X_cols if "time" not in col]


# Just a simple classifier to see the rolling works:

# In[ ]:


from catboost import CatBoostClassifier
import time

cat_up = CatBoostClassifier(thread_count=4, 
                            #n_estimators=400, 
                            max_depth=10, 
                            eta=0.1, 
                            loss_function='Logloss', 
                            random_seed = 64738,
                            iterations=1000,
                            verbose=10)

t = time.time()
print('Fitting Up')
cat_up.fit(X_train, u_train)#, cat_features) 
print(f'cat Done, time = {time.time() - t}')


# Find the last date in the dataset.

# In[ ]:


mt_df.tail(1)["time"][0]


# Slice the dataframe to get the last 30 days of data.  Use this later as basis to build the custom features for test data.

# In[ ]:


last_day = mt_df.tail(1)["time"][0]
past_offset = pd.Timedelta(30, unit='d')
future_offset = pd.Timedelta(1, unit='d')
date1 = last_day - past_offset
date2 = last_day + future_offset
roller = mt_df[mt_df['time'] > date1]


# In[ ]:


roller.shape


# In[ ]:


roller.head()


# Now try the tricks on test data that is provided a day at a time:

# In[ ]:


days = env.get_prediction_days()


# In[ ]:


n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    drop_cols = [col for col in market_obs_df.columns if col not in roller.columns]
    market_obs_df = market_obs_df.drop(drop_cols, axis=1)

    #add the new rows to the current set of lines used to build the new features
    roller3 = pd.concat([roller, market_obs_df]).reset_index()
    roller3.drop("index", axis=1, inplace=True)
    roller3.columns
    
    #Create the custom features for the new data (using the old to calculate it)
    roller4 = preprocess(roller3)
    #now slice only the data for the new day/rows along with custom features just added
    last_day = roller4.tail(1)["time"][0]
    roller5 = roller4[roller4['time'] >= last_day]
    #and add it to the current set of rows as basis to calculate the next date in this loop
    roller = pd.concat([roller, roller5])

    #slice the roller to keep it at 30 days worth of data and save memory
    past_offset = pd.Timedelta(30, unit='d')
    date1 = last_day - past_offset
    roller = roller[roller['time'] > date1]

    t = time.time()
    # discard assets that are not scored
    roller5 = roller5[roller5.assetCode.isin(predictions_template_df.assetCode)]
    X_market_obs = roller5[X_cols]
    prep_time += time.time() - t
    
    t = time.time()
    #make predictions for the new data received for the new date
    market_prediction = cat_up.predict_proba(X_market_obs)[:,1]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':roller5['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')


# Just print some final info:

# In[ ]:


roller.head()


# In[ ]:


df_feats = pd.DataFrame()
df_feats["names"] = X_cols
df_feats["weights"] = cat_up.feature_importances_
df_feats.sort_values(by="weights")


# That's all folks. I am not sure how to verify this all works as intended, and the custom features are correctly calculated for the new test data. But I think the general concept should work. 
# 
# This version also does not seem to score all that well but then I could use it as a basis at least to try other customized versions. For example, calculate rise/fall of an asset in relation to the others in the same period. This should make it closer to the idea of the "Mketres" columns in the original dataset.
# 
# If you have some ideas, or otherwise improvement suggestions, or anything..

# In[ ]:




