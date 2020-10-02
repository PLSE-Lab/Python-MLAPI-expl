#!/usr/bin/env python
# coding: utf-8

# # Attempt to generate news features
# 1. In V22 mkt cap taken as mean market cap for the asset
# 2. In V23 mkt cap taken as mean for asset and time
# 3. In V24 
#     a. sen SR mean taken instead of sum as asset and time level and also at the time of rolling in asset function. 
#     b. Also for asset sensr also we are multiplying by market cap
#     c. removing news for days that is in the same direction as prevailing sentiment
#  4. In V25 rollback of change in V24
# 

# In[ ]:


import pandas as pd
from ast import literal_eval
import sklearn.model_selection as skm
import sklearn.model_selection as skp
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import lightgbm as lgb
import numpy as np
import itertools as itr
import datetime as dt
import nltk

import plotly.offline as pyo
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sea
import gc
import sys
import multiprocessing as mp
import six
from abc import ABCMeta
from sklearn.base import BaseEstimator, ClassifierMixin
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import math

init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


#market_train_df = market_train_df[market_train_df["assetName"]=="Apple Inc"]
#news_train_df = news_train_df[news_train_df["assetName"] == "Apple Inc"]
print(market_train_df.shape)
print(news_train_df.shape)


# In[ ]:


def fix_time_col(in_mkt_df, in_news_df):
    if "datetime" in str(in_mkt_df["time"].dtype):
        in_mkt_df["time"] = in_mkt_df["time"].dt.date
        in_news_df["time"] = in_news_df["time"].dt.date

def clean_assetName_col(in_mkt_df, in_news_df):
    in_mkt_df["assetName"] = in_mkt_df["assetName"].str.lower()
    in_mkt_df["assetName"] = in_mkt_df["assetName"].str.strip()
    in_mkt_df["assetName"] = in_mkt_df["assetName"].str.replace("inc","")
    in_mkt_df["assetName"] = in_mkt_df["assetName"].str.replace("llc","")
    in_mkt_df["assetName"] = in_mkt_df["assetName"].str.replace("ltd","")
    in_mkt_df["assetName"] = in_mkt_df["assetName"].str.strip()

    in_news_df["assetName"] = in_news_df["assetName"].str.lower()
    in_news_df["assetName"] = in_news_df["assetName"].str.strip()
    in_news_df["assetName"] = in_news_df["assetName"].str.replace("inc","")
    in_news_df["assetName"] = in_news_df["assetName"].str.replace("llc","")
    in_news_df["assetName"] = in_news_df["assetName"].str.replace("ltd","")
    in_news_df["assetName"] = in_news_df["assetName"].str.strip()


# In[ ]:


def get_final_assetName(row ):
    min_mkt_assetNameSNo = row["min_mkt_assetNameSNo"]
    max_mkt_assetNameSNo = row["max_mkt_assetNameSNo"]
    min_assetNameSNo = row["min_assetNameSNo"]
    max_assetNameSNo = row["max_assetNameSNo"]
    if ((not math.isnan(min_mkt_assetNameSNo)) & (math.isnan(max_mkt_assetNameSNo))):
        return min_assetNameSNo
    elif ((not math.isnan(max_mkt_assetNameSNo)) & (math.isnan(min_mkt_assetNameSNo))):
        return max_assetNameSNo
    else:
        return 0
    
    
def get_replace_assetName(row):
    min_mkt_assetNameSNo = row["min_mkt_assetNameSNo"]
    max_mkt_assetNameSNo = row["max_mkt_assetNameSNo"]
    min_assetNameSNo = row["min_assetNameSNo"]
    max_assetNameSNo = row["max_assetNameSNo"]
    if ((math.isnan(min_mkt_assetNameSNo)) & (not math.isnan(max_mkt_assetNameSNo))):
        return min_assetNameSNo
    elif ((math.isnan(max_mkt_assetNameSNo)) & (not math.isnan(min_mkt_assetNameSNo))):
        return max_assetNameSNo
    else:
        return 0

dic_assetName={}
dic_assetNameSNo={}
dic_assetCodes={}
def set_assetNameCodes_SNo(in_mkt_df, in_news_df,dic_assetName, dic_assetNameSNo, dic_assetCodes):
    set_asset = set(in_news_df["assetName"].unique()) | set(in_mkt_df["assetName"].unique())
    set_codes = set(in_news_df["assetCodes"].unique())
    list_asset = list(set_asset)
    list_codes = list(set_codes)
    
    if len(dic_assetName) == 0:
        dic_assetName= {k:v for v,k in enumerate(list_asset)}
        dic_assetNameSNo= {v:k for v,k in enumerate(list_asset)}
        dic_assetCodes= {k:v for v,k in enumerate(list_codes)}
    else: #dictionary already exists, find only for those elemnts that are extra
        set_asset_new = set(dic_assetName.keys()) - set_asset
        asset_max_SNo = len(dic_assetName)
        dic_assetName_new = {k:v+asset_max_SNo for v,k in enumerate(list(set_asset_new))}
        dic_assetNameSNo_new = {v+asset_max_SNo:k for v,k in enumerate(list(set_asset_new))}
        dic_assetName.update(dic_assetName_new)
        dic_assetNameSNo.update(dic_assetNameSNo_new)
        
        set_codes_new = set(dic_assetCodes.keys()-set_codes)
        codes_max_SNo = len(dic_assetCodes)
        dic_assetCodes_new = {k:v+codes_max_SNo for v,k in enumerate(list(set_codes_new))}
        dic_assetCodes.update(dic_assetCodes_new)
        
    in_news_df["assetNameSNo"] = in_news_df["assetName"].map(dic_assetName)
    in_mkt_df["assetNameSNo"] = in_mkt_df["assetName"].map(dic_assetName)
    in_news_df["assetCodesSNo"] = in_news_df["assetCodes"].map(dic_assetCodes)
    return dic_assetName, dic_assetNameSNo, dic_assetCodes


# In[ ]:


def replace_assetName(in_mkt_df, in_news_df):
    in_news_df["num_rows"] = 1
    assetName_df = in_news_df[["assetCodesSNo","assetNameSNo","num_rows"]].groupby(                                                ["assetCodesSNo","assetNameSNo"], as_index=False).count()
    assetCode_df = assetName_df[["assetCodesSNo",                                  "assetNameSNo","num_rows"]].groupby(["assetCodesSNo"],                                  as_index=False).agg({"assetNameSNo": ["max","min"], "num_rows": "sum"})

    assetCode_df.columns = ["assetCodesSNo", "max_assetNameSNo", "min_assetNameSNo", "count_num_rows"]

    assetCode_df = assetCode_df[assetCode_df["max_assetNameSNo"] != assetCode_df["min_assetNameSNo"]]
    if assetCode_df.shape[0] > 0:
        print(assetCode_df.shape)
        assetCode_df["max_assetName"] = assetCode_df["max_assetNameSNo"].map(dic_assetNameSNo)
        assetCode_df["min_assetName"] = assetCode_df["min_assetNameSNo"].map(dic_assetNameSNo)
        list_mkt_asset = list(in_mkt_df["assetName"].unique())
        dic_mkt_assetName= {k:v for v,k in enumerate(list_mkt_asset)}
        assetCode_df["min_mkt_assetNameSNo"] = assetCode_df["min_assetName"].map(dic_mkt_assetName)
        assetCode_df["max_mkt_assetNameSNo"] = assetCode_df["max_assetName"].map(dic_mkt_assetName)


        assetCode_df["final_assetNameSNo"] = assetCode_df.apply(get_final_assetName, axis=1)
        assetCode_df["replace_assetNameSNo"] = assetCode_df.apply(get_replace_assetName, axis=1)

        assetCode_df["final_assetName"] = assetCode_df["final_assetNameSNo"].map(dic_assetNameSNo)

        df_temp = pd.merge(in_news_df.reset_index()[["index","assetCodesSNo","assetNameSNo"]],                            assetCode_df[["replace_assetNameSNo", "final_assetNameSNo","assetCodesSNo","final_assetName"]],                            how="inner", left_on=["assetCodesSNo","assetNameSNo"], right_on=["assetCodesSNo","replace_assetNameSNo"], left_index=True).set_index("index")
        in_news_df["new_assetNameSNo"] = -1
        in_news_df["new_assetName"] = ""

        in_news_df.loc[df_temp.index, "assetNameSNo"] = list(df_temp["final_assetNameSNo"])
        in_news_df.loc[df_temp.index, "assetName"] = list(df_temp["final_assetName"])


# In[ ]:


dic_rng = {}
dic_min = {}
def scale_col(in_news_df, col):
    global dic_rng
    global dic_min
    if col not in dic_rng.keys(): 
        col_max = in_news_df[col].max()
        col_min = in_news_df[col].min()
        col_rng = col_max - col_min
        dic_rng[col] = col_rng
        dic_min[col] = col_min
    else:
        col_min = dic_min[col] 
        col_rng = dic_rng[col]

    in_news_df[col] = (in_news_df[col]-col_min)/col_rng
    if col_min < 0:
        in_news_df[col] = 2*in_news_df[col] - 1
    print(col, in_news_df[col].min(), in_news_df[col].max())
    
        
def drop_extra_news_cols(in_news_df):
    news_drop_cols = ["sourceTimestamp","firstCreated","sourceId","headline","urgency","takeSequence","provider","audiences","bodySize",                      "headlineTag","sentenceCount","wordCount","firstMentionSentence","sentimentClass","sentimentWordCount"]
    news_drop_cols = [col  for col in news_drop_cols if col in in_news_df.columns ]
    in_news_df.drop(news_drop_cols, axis=1, inplace=True)
    

def drop_irrelevant_news(in_news_df):
    irrelevant_news = in_news_df[in_news_df["relevance"] < 0.3].index
    print("Num low relevance news:", len(irrelevant_news))
    in_news_df.drop(irrelevant_news, axis=0, inplace=True)
    del irrelevant_news
    gc.collect()
 
def drop_neutral_news(in_news_df):
    in_news_df["netSentimentPositive"] = (in_news_df["sentimentPositive"] - in_news_df["sentimentNegative"]) 
    neutral_news = in_news_df[in_news_df["netSentimentPositive"].abs() < 0.05].index
    in_news_df.drop(neutral_news, axis=0, inplace=True)
    in_news_df["netSentimentPositive"] = in_news_df["netSentimentPositive"] * (1 - in_news_df["sentimentNeutral"])
    print("Num neutral news:", len(neutral_news))
    del neutral_news
    gc.collect()

def gen_netnovelty_col(in_news_df):
    # if there is a news in 5 to 7 days then count will be more than 1. This news should have less importance as it is an old news
    arr_novelty = np.array(in_news_df[["noveltyCount12H","noveltyCount24H","noveltyCount3D","noveltyCount5D","noveltyCount7D"]])
    
    in_news_df["inv_netNovelty"] = list(1/(np.argmax(arr_novelty, axis=1)+2))
    col = "inv_netNovelty"
    scale_col(in_news_df, col)

setSector = {"TECH","ENER","BMAT","INDS","CYCS","NCYC","FINS","HECA","TECH","TCOM","UTIL"}
def get_sector_name(in_news_df):
    global setSector 
    sectorList = {"ENER":"Energy", "BMAT":"Basic Materials", "INDS":"Industrials", "CYCS":"Cyclical Consumer Goods & Services",      "NCYC":"Non Cyclical Consumer Goods & Services", "FINS": "Financials", "HECA":"Healthcare","TECH":"Technology",      "TCOM":"Telecommunication Services", "UTIL":"Utilities"}

    in_news_df["sector"] = in_news_df["subjects"].apply(lambda x: str(eval(x) & setSector))
    for col in setSector:
        print(col)
        in_news_df[col] = in_news_df["sector"].apply(lambda x: col in eval(x))
    lst_sector = list(setSector)
    dic_sector = {v: k for v,k in enumerate(list(setSector))}
    lst_sector.append("assetNameSNo")
    asset_df = in_news_df[lst_sector].groupby(["assetNameSNo"]).sum()
    asset_df1 = asset_df.sum(axis=1)
    asset_unknown_sector = list(asset_df1[asset_df1==0].index)
    print("Could not find sector of ", len(asset_unknown_sector) , " assets")
    asset_df.drop(asset_unknown_sector, axis=0, inplace=True)
    asset_df["sectorKey"] = asset_df.values.argmax(axis=1)
    asset_df["sectorName"] = asset_df["sectorKey"].map(dic_sector)
    print(asset_df.head(1))
    asset_df.reset_index(inplace=True)
    merged_df = pd.merge(in_news_df.reset_index()[["assetNameSNo","index"]], asset_df[["assetNameSNo","sectorName"]],                          left_on= ["assetNameSNo"], right_on= ["assetNameSNo"], how="left").set_index("index")
    in_news_df["sectorName"] = ""
    in_news_df.loc[merged_df.index,["sectorName"]] = merged_df.loc[merged_df.index,["sectorName"]]
    
    return asset_df[["assetNameSNo","sectorName"]]

def merge_mkt_news(in_mkt_df, in_news_df):
    in_mkt_df["mkt_cap"] = ((in_mkt_df["close"] + in_mkt_df["open"])/2)  * in_mkt_df["volume"] 
    in_mkt_df_grouped = in_mkt_df[["assetNameSNo","time", "close","returnsClosePrevRaw10","returnsClosePrevMktres10",                                   "returnsOpenNextMktres10","mkt_cap"]].groupby(["assetNameSNo","time"], as_index=False).sum()
    merged_df = pd.merge(in_news_df.reset_index()[["index","assetNameSNo","time"]], in_mkt_df_grouped, left_on= ["assetNameSNo","time"],                          right_on= ["assetNameSNo","time"], how="inner").set_index("index")
    in_news_df["close"] = 0
    in_news_df["returnsClosePrevRaw10"] = 0
    in_news_df["returnsClosePrevMktres10"] = 0
    in_news_df["returnsOpenNextMktres10"] = 0
    in_news_df["mkt_cap"] = 0
    in_news_df.loc[merged_df.index, ["close","returnsClosePrevRaw10","returnsClosePrevMktres10","returnsOpenNextMktres10","mkt_cap"]] =     merged_df.loc[merged_df.index, ["close","returnsClosePrevRaw10","returnsClosePrevMktres10","returnsOpenNextMktres10","mkt_cap"]]
    
    in_news_df["weighted_SenSR"] = in_news_df["SenSR"] * in_news_df["mkt_cap"]
    
    
def get_smooth_asset_sentiment(in_mkt_df, in_news_df):
    col="SenSR"
    asset_col = "asset_SenSR"
    rolling_col = "rolling_" + asset_col
    smooth_col = "smooth_" + asset_col
    in_news_df[asset_col] = in_news_df[col]
    in_news_df[rolling_col] = 0
    in_news_df[smooth_col] = 0
    
    lst_data = []
    for key, val_df in in_news_df.groupby(["assetNameSNo"]):
        print("Generating rolled and smooth col for asset:" + str(key) + ":" + dic_assetNameSNo[key])
        val_df.set_index(["time"], inplace=True)
        val_df[rolling_col] = val_df[asset_col].rolling(window=7, min_periods=1).mean()
        val_df.dropna(axis=0, inplace=True)
        y = list(val_df[rolling_col].values)
        x = range(val_df.shape[0])        
        smooth = lowess(y,x, 0.02)
        val_df[smooth_col] = list(smooth[:,1])
        del smooth, x, y
        lst_data.append(val_df[["assetNameSNo",asset_col,rolling_col,smooth_col]].reset_index())

    asset_df = pd.concat(lst_data, axis=0)
    merged_df = pd.merge(in_news_df.reset_index()[["index","assetNameSNo","time"]], asset_df, how="left",                          left_on=["assetNameSNo", "time"], right_on=["assetNameSNo","time"]).set_index("index")
    in_news_df.loc[merged_df.index, [asset_col,rolling_col,smooth_col]] = merged_df.loc[merged_df.index, [asset_col,rolling_col,smooth_col]]
    return merged_df

def get_smooth_sector_sentiment(in_mkt_df, in_news_df): 
    col = "weighted_SenSR"
    sector_col = "sector_SenSR"
    rolling_col = "rolling_" + sector_col
    smooth_col = "smooth_" + sector_col
    in_news_df[sector_col] = 0
    in_news_df[rolling_col] = 0
    in_news_df[smooth_col] = 0
    lst_data = []
    for key, val_df in in_news_df[[col,"time","sectorName","mkt_cap"]].groupby(["sectorName"]):
        print("Generating rolled and smooth col for sector:" + str(key) )
        val_df = val_df.groupby(["time","sectorName"], as_index=False).sum()
        val_df[sector_col] = val_df[col]/val_df["mkt_cap"]
        val_df.set_index(["time"], inplace=True)
        val_df[rolling_col] = val_df[sector_col].rolling(window=7, min_periods=1).mean()
        val_df.dropna(axis=0, inplace=True)
        y = list(val_df[rolling_col].values)
        x = range(val_df.shape[0])        
        smooth = lowess(y,x, 0.02)
        val_df[smooth_col] = list(smooth[:,1])
        del smooth, x, y
        lst_data.append(val_df[[sector_col,rolling_col,smooth_col, "sectorName"]].reset_index())
    
    sector_df = pd.concat(lst_data, axis=0)
    merged_df = pd.merge(in_news_df.reset_index()[["index","sectorName","time"]], sector_df, how="left",                          left_on=["sectorName", "time"], right_on=["sectorName","time"]).set_index("index")
   
    in_news_df.loc[merged_df.index, [sector_col,rolling_col,smooth_col]] = merged_df.loc[merged_df.index, [sector_col,rolling_col,smooth_col]]
    return merged_df

def get_smooth_overall_sentiment(in_mkt_df, in_news_df): 
    col = "weighted_SenSR"
    all_col = "all_SenSR"
    rolling_col = "rolling_" + all_col
    smooth_col = "smooth_" + all_col
    in_news_df[all_col] = 0
    in_news_df[rolling_col] = 0
    in_news_df[smooth_col] = 0
    lst_data = []
    val_df = in_news_df[[col,"time","mkt_cap"]].groupby(["time"]).sum()
    val_df[all_col] = val_df[col]/val_df["mkt_cap"]
    
    val_df[rolling_col] = val_df[all_col].rolling(window=7, min_periods=1).mean()
    val_df.dropna(axis=0, inplace=True)
    y = list(val_df[rolling_col].values)
    x = range(val_df.shape[0])        
    smooth = lowess(y,x, 0.02)
    
    val_df[smooth_col] = list(smooth[:,1])
    del smooth, x, y
    lst_data.append(val_df[[all_col,rolling_col,smooth_col]].reset_index())
    
    sector_df = pd.concat(lst_data, axis=0)
    merged_df = pd.merge(in_news_df.reset_index()[["index","time"]], sector_df, how="left",                          left_on=["time"], right_on=["time"]).set_index("index")
    
    in_news_df.loc[merged_df.index, [all_col,rolling_col,smooth_col]] = merged_df.loc[merged_df.index, [all_col, rolling_col,smooth_col]]
    return merged_df

def gen_news_features(in_mkt_df, in_news_df):
    
    global setSector
    drop_extra_news_cols(in_news_df)
    drop_irrelevant_news(in_news_df)
    drop_neutral_news(in_news_df)

    gen_netnovelty_col(in_news_df)
    sector_name_df  = get_sector_name(in_news_df)
    in_news_df.drop("subjects", axis=1, inplace=True)
    in_news_df.drop(["noveltyCount12H","noveltyCount24H","noveltyCount3D","noveltyCount5D","noveltyCount7D"], axis=1, inplace=True)
    in_news_df.drop(["volumeCounts12H","volumeCounts24H","volumeCounts3D","volumeCounts5D","volumeCounts7D"], axis=1, inplace=True)

    in_news_df["SenSR"] = in_news_df["netSentimentPositive"] * in_news_df["relevance"] * (in_news_df["inv_netNovelty"])
    in_news_df = in_news_df.groupby(["assetName","sectorName","time","assetNameSNo"], as_index=False).sum()
    merge_mkt_news(in_mkt_df, in_news_df)
    #in_news_df = in_news_df[((in_news_df["returnsClosePrevMktres10"] < 0) & (in_news_df["SenSR"] > 0)) | 
    #                ((in_news_df["returnsClosePrevMktres10"] < 0) & (in_news_df["SenSR"] > 0))]
    asset_df = get_smooth_asset_sentiment(in_mkt_df,in_news_df)
    sector_df = get_smooth_sector_sentiment(in_mkt_df, in_news_df)
    all_df = get_smooth_overall_sentiment(in_mkt_df, in_news_df)
    return sector_name_df, asset_df, sector_df, all_df, in_news_df
    


# In[ ]:


news_train_df.head(1)


# In[ ]:


#news_train_df.drop(["mkt_cap","weighted_SenSR","sector_SenSR","rolling_sector_SenSR","smooth_sector_SenSR"], axis=1, inplace=True)
#get_smooth_sector_sentiment(market_train_df, news_train_df)


# In[ ]:


market_train_df.head(1)


# In[ ]:


fix_time_col(market_train_df, news_train_df)
clean_assetName_col(market_train_df, news_train_df)
dic_assetName, dic_assetNameSNo, dic_assetCodes = set_assetNameCodes_SNo(market_train_df, news_train_df,dic_assetName, dic_assetNameSNo, dic_assetCodes)
replace_assetName(market_train_df, news_train_df)
sector_name_df, asset_df, sector_df, all_df, news_feature_df = gen_news_features(market_train_df, news_train_df)


# In[ ]:


drop_cols = list(set(["companyCount","marketCommentary","assetCodesSNo","num_rows"]) | setSector)
news_feature_df.drop(drop_cols, axis=1, inplace=True)
news_feature_df.tail(1)


# In[ ]:


drop_cols = ["sentimentNegative","sentimentPositive","sentimentNeutral"]
news_feature_df.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


news_feature_df["newsCount"]=0
df_sector = news_feature_df[["sectorName","newsCount"]].groupby(["sectorName"]).count()
df_sector = df_sector.unstack()
df_sector.plot(kind="bar", title="Sector Wise News Count", figsize=(10,4));


# In[ ]:


from pandas.plotting import scatter_matrix
apple_df = news_feature_df[news_feature_df["assetName"]=="apple"]
apple_df["target"] = apple_df["returnsOpenNextMktres10"].apply(lambda x: 1 if x>0 else 2 if x<0 else 3)
apple_df["asset_sector_SenSR"] = apple_df["smooth_asset_SenSR"] * apple_df["smooth_sector_SenSR"]
apple_df["asset_sector_all_SenSR"] = apple_df["smooth_asset_SenSR"] * apple_df["smooth_sector_SenSR"] * apple_df["smooth_all_SenSR"]
col_list1 = ["returnsClosePrevMktres10","returnsOpenNextMktres10",            "asset_SenSR","rolling_asset_SenSR","smooth_asset_SenSR"]
col_list2 = ["returnsClosePrevMktres10","returnsOpenNextMktres10",            "sector_SenSR","rolling_sector_SenSR","smooth_sector_SenSR"]
col_list3 = ["returnsClosePrevMktres10","returnsOpenNextMktres10",            "asset_sector_SenSR","asset_sector_all_SenSR"]
for col in list(set(col_list1)|set(col_list2)|set(col_list3)):
    scale_col(apple_df, col)
scale_col(apple_df, "smooth_all_SenSR")
#apple_df = apple_df[((apple_df["returnsClosePrevMktres10"] < 0) & (apple_df["smooth_asset_SenSR"] > 0)) | 
#                    ((apple_df["returnsClosePrevMktres10"] < 0) & (apple_df["smooth_asset_SenSR"] > 0))]


# In[ ]:




ax = scatter_matrix(apple_df[col_list1],figsize=(15,15))


# Asset Sentiment has corelation with next 10 days returns in the sentiment and previous returns are in opposite direction.

# In[ ]:


scatter_matrix(apple_df[col_list2], figsize=(20,20));


# Smooth Sector Sentiment has corelation with next 10 day returns.

# In[ ]:


scatter_matrix(apple_df[col_list3], figsize=(20,20));


# Looks like Asset Sector SenSR has some corelation with next 10 day returns.

# In[ ]:


#fig, axs = plt.subplots(1,1, figsize=(20,10))
#apple_df.set_index(["time"], inplace=True)
ax1 = apple_df[["smooth_sector_SenSR","smooth_asset_SenSR","smooth_all_SenSR","returnsOpenNextMktres10"]].plot(figsize=(40,10))
ax1.legend()


# In[ ]:


for key, val_df in sector_df.groupby(["sectorName"]):
    val_df.set_index(["time"], inplace=True)
    ax1 = val_df["smooth_sector_SenSR"].plot(figsize=(40,10), label=key)
    ax1.legend()


# # ENERGY Sector
# We can see that energy sector had a major dip in 2015 as well as around 2016
# 
# # FINANCE Sector
# Similarly we can see the finance sector meltdown in 2009. 
# 
# How well these features will impact the model is to be seen.

# In[ ]:




