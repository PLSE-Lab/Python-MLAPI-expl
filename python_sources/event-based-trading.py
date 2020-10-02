#!/usr/bin/env python
# coding: utf-8

# ## In-depth Introduction
# First let's import the module and create an environment.

# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()


# In[ ]:


import pandas as pd
from ast import literal_eval
import sklearn.model_selection as skm
import lightgbm as lgb
import numpy as np
import itertools as itr

import plotly.offline as pyo
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go

import matplotlib.pyplot as plt

init_notebook_mode(connected=True)
(market_train_df, news_train_df) = env.get_training_data()
sampling = False
SERIES_LEN = 31
org_cols = market_train_df.columns.values
org_market_train_df = market_train_df
org_news_train_df = news_train_df


# ## **`get_training_data`** function
# 
# Returns the training data DataFrames as a tuple of:
# * `market_train_df`: DataFrame with market training data
# * `news_train_df`: DataFrame with news training data
# 
# These DataFrames contain all market and news data from February 2007 to December 2016.  See the [competition's Data tab](https://www.kaggle.com/c/two-sigma-financial-news/data) for more information on what columns are included in each DataFrame.

# In[ ]:


def add_missing_rows(in_df):
    print("-----------------------------------")
    print("Rows before adding missing values:", in_df.shape[0])
    asset_code_list = list(in_df["assetCode"].unique())
    time_list = list(in_df["time"].unique())
    dic_asseCode_sno = {k: v for v, k in enumerate(asset_code_list)}
    expected_data = list(itr.product(asset_code_list, time_list))
    expected_df = pd.DataFrame(expected_data, columns=["assetCode","expectedTime"])
    expected_df["assetCodeT"] = expected_df["assetCode"].map(dic_asseCode_sno)
    in_df["assetCodeT"] = in_df["assetCode"].map(dic_asseCode_sno)
    expected_df = expected_df.sort_values(["expectedTime"])
    expected_df = expected_df.drop(["assetCode"], axis=1)
    in_df = in_df.sort_values(["time"])
    merged_df = pd.merge_asof(expected_df, in_df, left_on=["expectedTime"], right_on=["time"], direction="nearest", by="assetCodeT", tolerance=pd.Timedelta(str(SERIES_LEN) + " days"))
    missing_rows= merged_df[merged_df["time"] != merged_df["expectedTime"]]
    generated_rows = missing_rows[~missing_rows["time"].isna()]
    print(missing_rows.shape[0], generated_rows.shape[0])
    merged_df["time"] = merged_df["expectedTime"]
    print("Rows after adding missing values:", merged_df.shape[0])
    return merged_df[org_cols]


# In[ ]:


#Handle null values
def remove_rows_with_nulls(in_df):
    print("-----------------------------------")
    in_df_cols = in_df.columns.values
    print("Total Rows before removing nulls: ", in_df.shape[0])
    for col in in_df_cols:
        df_col = in_df[[col]].values
        num_na = df_col[pd.isna(df_col)].shape[0]
        #print("Col: {0} Num NA: {1}".format(col, num_na))
    in_df = in_df.dropna(axis=0)
    print("Total Rows after removing nulls: ", in_df.shape[0])
    return in_df


# In[ ]:


def remove_assets_with_missing_rows(in_df):  
   print("-----------------------------------")
   time_list = list(in_df["time"].unique())
   dic_time_sno = {k: v for v, k in enumerate(time_list)}
   print("Total Rows before asset with missing rows: ", in_df.shape[0])

   grouped_df = in_df[["time","assetCode"]].groupby(["assetCode"],as_index=False).agg({"time":{"min_time":"min", "max_time":"max","num_rows":"count"}})
   grouped_df.columns = ["assetCode","min_time","max_time","num_rows"]

   grouped_df["min_timeT"] = grouped_df["min_time"].map(dic_time_sno)
   grouped_df["max_timeT"] = grouped_df["max_time"].map(dic_time_sno)
   grouped_df["expected_rows"] = grouped_df["max_timeT"] - grouped_df["min_timeT"] + 1
   grouped_df["num_missing_rows"] = grouped_df["expected_rows"] - grouped_df["num_rows"]
   print(grouped_df.head(2).T)
   grouped_df_missing = grouped_df[grouped_df["num_missing_rows"]>0]
   missing_asset = list(grouped_df_missing["assetCode"])
   in_df = in_df[~in_df["assetCode"].isin(missing_asset)]
   print("Total Rows after removing asset with missing rows: ", in_df.shape[0])
   return in_df[org_cols]


# In[ ]:


#Keep rows for each asset so tht we can have fixed length series data
def remove_rows_extra_from_series_len(in_df):
    print("-----------------------------------")
    time_list = list(in_df["time"].unique())
    dic_time_sno = {k: v for v, k in enumerate(time_list)}
    print("Total Rows before remove_rows_extra_from_series_len: ", in_df.shape[0])
    in_df["timeT"] = in_df["time"].map(dic_time_sno)
    print(in_df.shape)

    grouped_df = in_df[["time","assetCode"]].groupby(["assetCode"],as_index=False).agg({"time":{"min_time":"min", "max_time":"max","num_rows":"count"}})
    grouped_df.columns = ["assetCode","min_time","max_time","num_rows"]

    grouped_df["min_timeT"] = grouped_df["min_time"].map(dic_time_sno)
    grouped_df["max_timeT"] = grouped_df["max_time"].map(dic_time_sno)
    grouped_df["expected_rows"] = grouped_df["max_timeT"] - grouped_df["min_timeT"] + 1
    grouped_df["del_timeT"] = grouped_df["min_timeT"] + (grouped_df["num_rows"] - (SERIES_LEN * (grouped_df["num_rows"] // SERIES_LEN)))

    in_df = pd.merge(in_df, grouped_df, how="inner", left_on=["assetCode"], right_on=["assetCode"])
    in_df = in_df[in_df["timeT"] >= in_df["del_timeT"]]
    print("Total Rows after remove_rows_extra_from_series_len: ", in_df.shape[0])
    return in_df[org_cols]


# In[ ]:


def get_merged_df(in_market_data, in_news_data):
    print("-----------------------------------")
    print("Rows before merging",in_market_data.shape[0])
    in_market_data['mktDate'] = in_market_data['time'].dt.date #strftime('%Y-%m-%d %H:%M:%S').str.slice(0,10)
    in_news_data["newsDate"] = in_news_data["firstCreated"].dt.date #strftime('%Y-%m-%d %H:%M:%S').str.slice(0,10)
    
    #Extract asset code for news records
    in_news_data["assetCode_new"] = in_news_data["assetCodes"].map(lambda x: list(eval(x))[0])
    in_news_data["assetCode_len"] = in_news_data["assetCodes"].map(lambda x: len(list(eval(x))))
    
    in_news_data_final = in_news_data
    if 1 == 2:
        numAssetCodes = in_news_data["assetCode_len"].max()
        additionalData = in_news_data
        colNames = list(in_news_data_final.columns.values)
        for i in range(1,numAssetCodes):
            additionalData = additionalData[additionalData["assetCode_len"] > i]
            additionalData["assetCode_new"] =  additionalData["assetCodes"].map(lambda x: list(eval(x))[i] if len(list(eval(x)))>i else "")
            unexpectedData = additionalData[additionalData["assetCode_new"] == ""].head(5)
            print(unexpectedData[["assetCodes","assetCode_new","assetCode_len"]])
            in_news_data_final = in_news_data_final.append(additionalData[colNames])

    in_news_data_grouped = in_news_data_final.groupby(["newsDate","assetCode_new"], as_index=False).mean()
    df_merged = pd.merge(in_market_data,in_news_data_grouped, how="left",left_on=["mktDate","assetCode"], right_on=["newsDate","assetCode_new"])
    dic_assetCode_sno = {k: v for v, k in enumerate(df_merged['assetCode'].unique())}
    dic_sno_assetCode = {v: k for v, k in enumerate(df_merged['assetCode'].unique())}
    date_list = list(df_merged["time"].dt.date.sort_values().unique())
    dic_date_sno = {k: v for v, k in enumerate(date_list)}
    dic_sno_date = {v: k for v, k in enumerate(date_list)}
    df_merged['assetCodeT'] = df_merged['assetCode'].map(dic_assetCode_sno)
    df_merged['mktDateT'] = df_merged['mktDate'].map(dic_date_sno)
    print("Rows after merging",df_merged.shape[0])
    return df_merged, dic_sno_assetCode, dic_sno_date


# In[ ]:


def get_x_y(merged_df, dic_sno_assetCode, dic_sno_date):
    print("-----------------------------------")
    feature_cols = ["volume","close","open", "returnsClosePrevRaw1","returnsOpenPrevRaw1","returnsClosePrevMktres1","returnsOpenPrevMktres1", 
                    "returnsClosePrevRaw10", "returnsOpenPrevRaw10","returnsClosePrevMktres10","returnsOpenPrevMktres10",
                   "urgency","marketCommentary","relevance","sentimentClass","sentimentNegative","sentimentNeutral","sentimentPositive",
                   "noveltyCount12H","noveltyCount24H", "noveltyCount3D","noveltyCount5D","noveltyCount7D",
                   "volumeCounts12H","volumeCounts24H","volumeCounts3D","volumeCounts5D","volumeCounts7D","returnsOpenNextMktres10","assetCodeT", "mktDateT"]
    df_feature = merged_df[feature_cols]
    df_feature = df_feature.fillna(0)
    num_rows = df_feature.shape[0]/SERIES_LEN
    arr_data = np.array(np.split(df_feature[feature_cols].values, num_rows))
    max_date = merged_df["mktDate"].max()
    min_date = merged_df["mktDate"].min()
    df_lstm = pd.DataFrame({"data":list(arr_data)})
    df_lstm["x_data"] = df_lstm["data"].apply(lambda x: np.array(x[:-1,:-2]))
    df_lstm["y_data"] = df_lstm["data"].apply(lambda x: x[-1,-3])
    df_lstm["assetCodeT"] = df_lstm["data"].apply(lambda x: x[-1,-2])
    df_lstm["mktDateT"] = df_lstm["data"].apply(lambda x: x[-1,-1])
    df_lstm["assetCode"] = df_lstm["assetCodeT"].map(dic_sno_assetCode)
    df_lstm["mktDate"] = df_lstm["mktDateT"].map(dic_sno_date)
    train_x = np.array(df_lstm["x_data"])
    train_y = np.array(df_lstm["y_data"].apply(lambda x: 1 if x>0 else 0))

    x_data = []
    for x in train_x:
        x_data.append(x)
    train_x = np.array(x_data)
    train_y = train_y.reshape(train_y.shape[0],1)
    min_y = df_lstm["y_data"].min()
    max_y = df_lstm["y_data"].max()
    return train_x, train_y, df_lstm


# In[ ]:


market_train_df = org_market_train_df
news_train_df = org_news_train_df
pred_market_train_df = market_train_df
market_train_df = add_missing_rows(market_train_df)
market_train_df = remove_rows_with_nulls(market_train_df)
news_train_df = remove_rows_with_nulls(news_train_df)
market_train_df = remove_assets_with_missing_rows(market_train_df)
market_train_df = remove_rows_extra_from_series_len(market_train_df)
merged_df, dic_sno_assetCode, dic_sno_date = get_merged_df(market_train_df, news_train_df)
train_x, train_y, df_lstm = get_x_y(merged_df, dic_sno_assetCode, dic_sno_date)


# In[ ]:


import keras
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                                            patience=1, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

verbose, epochs, batch_size = 1, 50, 16
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)


# In[ ]:


def make_random_predictions_lstm(market_obs_df, news_obs_df, predictions_df):
    global pred_market_train_df
    global news_train_df
    print("-------------------Start Predition---------------")
    obs_time = market_obs_df["time"][0]
    obs_date = market_obs_df["time"].dt.date[0]
    all_time_list = np.array(pred_market_train_df["time"].sort_values().unique())
    all_date_list = np.array(pred_market_train_df["time"].dt.date.sort_values().unique())
    prv_time_list = all_time_list[all_time_list < obs_time]
    prv_date_list = all_date_list[all_date_list < obs_date]
    x_time_list = list(prv_time_list[-SERIES_LEN+1:])
    x_date_list = list(prv_date_list[-SERIES_LEN+1:])
    print(obs_time)
    
    obs_asset_list = list(market_obs_df["assetCode"].unique())
    import itertools as itr
    expected_data = list(itr.product(obs_asset_list, x_time_list))
    expected_df = pd.DataFrame(expected_data, columns=["assetCode","expectedTime"])
    
    prv_market_df = pred_market_train_df[((pred_market_train_df["assetCode"].isin(obs_asset_list)) & (pred_market_train_df["time"].isin(x_time_list)))]
    prv_news_df = news_train_df[((news_train_df["assetCode_new"].isin(obs_asset_list)) & (news_train_df["firstCreated"].dt.date.isin(x_date_list)))]
    
    dic_assetCode_sno = {k: v for v, k in enumerate(obs_asset_list)}
    dic_sno_assetCode = {v: k for v, k in enumerate(obs_asset_list)}
    
    expected_df["assetCodeT"] = expected_df['assetCode'].map(dic_assetCode_sno)
    prv_market_df["assetCodeT"] = prv_market_df['assetCode'].map(dic_assetCode_sno)
    
    expected_df = expected_df.drop(["assetCode"], axis=1)
    
    expected_df = expected_df.sort_values(["expectedTime","assetCodeT"])
    prv_market_df = prv_market_df.sort_values(["time", "assetCodeT"])
    
    prv_market_df = remove_rows_with_nulls(prv_market_df)
    expected_df = remove_rows_with_nulls(expected_df)
        
    mkt_actual_df = pd.merge_asof(expected_df, prv_market_df, left_on=["expectedTime"], right_on=["time"], by='assetCodeT', direction="nearest", tolerance=pd.Timedelta(str(SERIES_LEN) + " days"))
    mkt_actual_df["time"] = mkt_actual_df["expectedTime"]
    
    final_market_df = market_obs_df.append(mkt_actual_df[org_cols])
    final_news_df = news_obs_df.append(prv_news_df[news_obs_df.columns])
    
    merged_obs_df, dic_sno_assetCode, dic_sno_date = get_merged_df(final_market_df, final_news_df)
    train_x, train_y, df_lstm = get_x_y(merged_obs_df, dic_sno_assetCode, dic_sno_date)
    pred = model.predict(train_x)
    
    df_lstm["pred"] = pred
    
    #min_pred = np.min(np.array([pred[pred>0.0].min(), pred[pred<0.0].max()*(-1.0)]))
    
    predictions_df.set_index("assetCode")
    df_lstm.set_index("assetCode")
    final_df = predictions_df.join(df_lstm[["pred"]], how="left")
    final_df["pred"] = final_df["pred"]*2-1
    predictions_df.reset_index()
    df_lstm.reset_index()
    
    predictions_df.confidenceValue = final_df["pred"]
    
    predictions_df = predictions_df.fillna(0)
    news_train_df = news_train_df.append(news_obs_df)
    market_obs_df["universe"]=1
    market_obs_df_final = pd.merge(market_obs_df, df_lstm[["assetCode","pred"]],how="left", left_on = ["assetCode"], right_on=["assetCode"])
    market_obs_df_final["returnsOpenNextMktres10"] = market_obs_df_final["pred"].fillna(0)
    pred_market_train_df = pred_market_train_df.append(market_obs_df_final[org_cols])
    pred_market_train_df = pred_market_train_df[org_cols]
    


# ## `get_prediction_days` function
# 
# Generator which loops through each "prediction day" (trading day) and provides all market and news observations which occurred since the last data you've received.  Once you call **`predict`** to make your future predictions, you can continue on to the next prediction day.
# 
# Yields:
# * While there are more prediction day(s) and `predict` was called successfully since the last yield, yields a tuple of:
#     * `market_observations_df`: DataFrame with market observations for the next prediction day.
#     * `news_observations_df`: DataFrame with news observations for the next prediction day.
#     * `predictions_template_df`: DataFrame with `assetCode` and `confidenceValue` columns, prefilled with `confidenceValue = 0`, to be filled in and passed back to the `predict` function.
# * If `predict` has not been called since the last yield, yields `None`.

# In[ ]:


# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()


# Note that we'll get an error if we try to continue on to the next prediction day without making our predictions for the current day.

# ### **`predict`** function
# Stores your predictions for the current prediction day.  Expects the same format as you saw in `predictions_template_df` returned from `get_prediction_days`.
# 
# Args:
# * `predictions_df`: DataFrame which must have the following columns:
#     * `assetCode`: The market asset.
#     * `confidenceValue`: Your confidence whether the asset will increase or decrease in 10 trading days.  All values must be in the range `[-1.0, 1.0]`.
# 
# The `predictions_df` you send **must** contain the exact set of rows which were given to you in the `predictions_template_df` returned from `get_prediction_days`.  The `predict` function does not validate this, but if you are missing any `assetCode`s or add any extraneous `assetCode`s, then your submission will fail.

# Let's make random predictions for the first day:

# In[ ]:


if 1==1: 
    market_obs_df, news_obs_df, predictions_template_df = next(days)
    make_random_predictions_lstm(market_obs_df, news_obs_df,predictions_template_df)
    env.predict(predictions_template_df) 


# In[ ]:





# Now we can continue on to the next prediction day and make another round of random predictions for it:

# ## Main Loop
# Let's loop through all the days and make our random predictions.  The `days` generator (returned from `get_prediction_days`) will simply stop returning values once you've reached the end.

# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions_lstm(market_obs_df, news_obs_df,predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')


# ## **`write_submission_file`** function
# 
# Writes your predictions to a CSV file (`submission.csv`) in the current working directory.

# In[ ]:


env.write_submission_file()


# In[ ]:


# We've got a submission file!
import os
print([filename for filename in os.listdir('.') if '.csv' in filename])


# As indicated by the helper message, calling `write_submission_file` on its own does **not** make a submission to the competition.  It merely tells the module to write the `submission.csv` file as part of the Kernel's output.  To make a submission to the competition, you'll have to **Commit** your Kernel and find the generated `submission.csv` file in that Kernel Version's Output tab (note this is _outside_ of the Kernel Editor), then click "Submit to Competition".  When we re-run your Kernel during Stage Two, we will run the Kernel Version (generated when you hit "Commit") linked to your chosen Submission.

# ## Restart the Kernel to run your code again
# In order to combat cheating, you are only allowed to call `make_env` or iterate through `get_prediction_days` once per Kernel run.  However, while you're iterating on your model it's reasonable to try something out, change the model a bit, and try it again.  Unfortunately, if you try to simply re-run the code, or even refresh the browser page, you'll still be running on the same Kernel execution session you had been running before, and the `twosigmanews` module will still throw errors.  To get around this, you need to explicitly restart your Kernel execution session, which you can do by pressing the Restart button in the Kernel Editor's bottom Console tab:
# ![Restart button](https://i.imgur.com/hudu8jF.png)

# In[ ]:


import numpy as np
def make_random_predictions(market_obs_df, news_obs_df, predictions_df):
    print (market_obs_df.shape, news_obs_df.shape, predictions_df.shape)
    print(predictions_df.head(5))
    merged_obs_df = Get_Merged_DF(market_obs_df, news_obs_df)
    merged_obs_df = merged_obs_df[feature_cols]
    merged_obs_df = merged_obs_df.fillna(0)
    x_obs_data = np.array(merged_obs_df[feature_cols])
    print(x_obs_data.shape)
    x_obs_data = 1 - ((maxs - x_obs_data) / rng)
    obs_pred = clf.predict(x_obs_data)
    print(obs_pred.shape)
    print(predictions_df.shape)
    print((obs_pred * 2 - 1).shape)
    predictions_df.confidenceValue = obs_pred * 2 - 1
    predictions_df = predictions_df.fillna(0)
if 1==2:
    market_train_df_sample.head(1)
    market_train_df_sample.tail(1)
    news_train_df_sample.head(1)
    news_train_df_sample.tail(1)
    news_train_df_sample.head(1) 
    df_merged.head(1)
    df_merged.tail(1)
if 1==2:
    all_cols = feature_cols.append(label_col)
    df_merged_feature = df_merged[feature_cols]
    df_merged_feature = df_merged_feature.fillna(0)

    #df_merged_feature = df_merged_feature.dropna(axis=0)

    feature_data = np.array(df_merged_feature)
    mins = np.min(feature_data, axis=0)
    maxs = np.max(feature_data, axis=0)
    rng = maxs - mins
    feature_data = 1 - ((maxs - feature_data) / rng)
    label_data = df_merged[label_col] >= 0
if 1==2:
    label_data = np.array(label_data)
    maxs = maxs.reshape(1,-1)
    rng = rng.reshape(1,-1)
    print(maxs.shape)
    print(rng.shape)
    print(feature_data.shape)
    x_train, x_valid, y_train, y_valid = skm.train_test_split(np.array(feature_data), np.array(label_data), test_size=0.2)
    print(x_train.shape)
    print(x_valid.shape)
    print(y_train.shape)
    print(y_valid.shape)
    d_train = lgb.Dataset(x_train, label=y_train)
    params = {}
    params['learning_rate'] = 0.001
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 300
    params['min_data'] = 50
    params['max_depth'] = 10

    clf = lgb.train(params, d_train, 1000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




