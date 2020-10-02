#!/usr/bin/env python
# coding: utf-8

# 1. Remove assetCodeSNo as a feature
# 2. Specify sectorSNo as a feature
# 3. Specify sectorSNo as categorical
# 4. Drop records with unidentified sector
# 5. Fix open close for test data as well
# 6. Remove lag features for open and close
# TODO: Add assetCode as categorical feature

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import lightgbm as lgb
import pandas as pd
from kaggle.competitions import twosigmanews
import matplotlib.pyplot as plt
import random
from datetime import datetime, date
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time
import math
import gc
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import inspect, sys
from multiprocessing import Pool


# In[ ]:


dic_assetName={}
dic_assetNameSNo={}
dic_assetCodes={}
dic_assetCode={}
dic_assetCodeSNo={}
dic_rng = {}
dic_min = {}
set_sector = {"TECH","ENER","BMAT","INDS","CYCS","NCYC","FINS","HECA","TECH","TCOM","UTIL"}
dic_sector_code_name = {"ENER":"Energy", "BMAT":"Basic Materials", "INDS":"Industrials", "CYCS":"Cyclical Consumer Goods & Services",      "NCYC":"Non Cyclical Consumer Goods & Services", "FINS": "Financials", "HECA":"Healthcare","TECH":"Technology",      "TCOM":"Telecommunication Services", "UTIL":"Utilities"}
sector_col = list(set_sector)
sector_col.append("assetNameSNo")
dic_sectorSNo = {v: k for v,k in enumerate(list(set_sector))}
dic_sector = {k: v for v,k in enumerate(list(set_sector))}
dic_sectorSNo["-1"] = "Unknown"
dic_sector["Unknown"] = -1
dic_asset_sector = {}

n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
total_market_obs_df = []
total_news_obs_df = []
return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10',"SenSR"] #'open','close',]
sector_return_features = ["weighted_SenSR"]
n_lag = [3,7,14]
debug_func = set()
prv_mkt_data = []
prv_news_data = []


# In[ ]:


# official way to get the data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
days = env.get_prediction_days()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()
#market_train_df = market_train_df[(market_train_df["assetName"]=="Apple Inc") | (market_train_df["assetName"] == "Chevron Corp")]
#news_train_df = news_train_df[(news_train_df["assetName"]=="Apple Inc") | (news_train_df["assetName"]=="Chevron Corp")]


# In[ ]:


if 1==2:
    prv_time_list = market_train_df["time"].dt.date.unique()[-14:]
    prv_time_list.sort()
    prv_time_list
    for prv_time in prv_time_list:
        prv_mkt_data.append(market_train_df[market_train_df["time"].dt.date == prv_time])
        prv_news_data.append(news_train_df[news_train_df["time"].dt.date == prv_time])


# In[ ]:


#market_train_df[market_train_df["assetName"].str.contains("Morgan")]


# In[ ]:


def plotdata(ticker):
    tickerdata = market_train_df[market_train_df["assetCode"]==ticker]
    tickerdata.set_index(["time"],inplace=True)
    fig,ax = plt.subplots(nrows=2,ncols=2, figsize=(27,9))
    tickerdata["close"].plot(ax=ax[0,0])
    tickerdata["returnsOpenNextMktres10"].plot(ax=ax[0,1])
    result =plot_acf(tickerdata["close"].dropna(), ax=ax[1,0])
    result =plot_acf(tickerdata["returnsOpenNextMktres10"].dropna(), ax=ax[1,1])
    #Partial auto corelation sudden drop indicated auto regressive model, gradual drop indicates moving average model
    #Most data exhibit sudden drop
    #result =plot_pacf(tickerdata["close"].dropna(), ax=ax[2,0])
    #result =plot_pacf(tickerdata["returnsOpenNextMktres10"].dropna(), ax=ax[2,1])
    plt.tight_layout()
                                               

#CVX.N
#AAPL.O


# In[ ]:


def fixassetopenclose(in_asset_df):
    #print(in_asset_df["assetCode"].unique())
    in_asset_df["final_close"] = in_asset_df["close"]
    in_asset_df["final_open"] = in_asset_df["open"]
    in_asset_df["adjusted_close"] = in_asset_df["close"].shift(-1)/(1 + in_asset_df["returnsClosePrevRaw1"].shift(-1))
    in_asset_df["ratio"] = in_asset_df["close"]/in_asset_df["adjusted_close"]
    idx_list = in_asset_df[(in_asset_df["ratio"].round(1) <0.9) | (in_asset_df["ratio"].round(1) >1.1)].index
    for idx in idx_list: 
        affected_idx = in_asset_df[in_asset_df.index <= idx].index
        ratio = in_asset_df.loc[idx,"ratio"]
        #print(ratio)
        in_asset_df.loc[affected_idx,["final_close","final_open"]] =  in_asset_df.loc[affected_idx,["final_close","final_open"]] / ratio
        del affected_idx
    changed_recs = in_asset_df[in_asset_df["final_close"] != in_asset_df["close"]].index
    return in_asset_df.loc[changed_recs,["final_close","final_open"]]
        
def fixopenclose(in_mkt_df):
    assetlist = []
    changedlist = []
    df_group = in_mkt_df.groupby(["assetCode"])
    for group in df_group:
        assetlist.append(group[1])
    
    pool = Pool(4)
    changedlist = pool.map(fixassetopenclose, assetlist)
    pool.close()
    
    changed_df = pd.concat(changedlist)
    changed_df.dropna(inplace=True)
   
    in_mkt_df.loc[changed_df.index,["close"]] = list(changed_df["final_close"])
    in_mkt_df.loc[changed_df.index,["open"]] = list(changed_df["final_open"])
       
fixopenclose(market_train_df)


# In[ ]:


plotdata("AAPL.O")
plotdata("CVX.N")
plotdata("JPM.N")


# In[ ]:


news_train_df.head(1)


# In[ ]:


def findnulls(in_mkt_df, in_news_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_mkt_cols = set(in_mkt_df.columns)
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
    for col in in_mkt_df:
        cols = in_mkt_df[col]
        if fname in debug_func:
            print("Market df: Num nulls in " + col + " is " + str(cols[cols.isna()].shape[0]) + " out of " + str(cols.shape[0]))
    for col in in_news_df:
        cols = in_news_df[col]
        if fname in debug_func:
            print("Market df: Num nulls in " + col + " is " + str(cols[cols.isna()].shape[0]) + " out of " + str(cols.shape[0]))
    #---------------------------------
    if fname in debug_func:
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")


# In[ ]:


#findnulls(market_train_df, news_train_df)
#print("As we can see only columns that are null are the prev columns!Let us keep them in the dataset for now.")


# In[ ]:


dic_time = {}
def fix_time_col(in_mkt_df, in_news_df):
    global dic_time
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_mkt_cols = set(in_mkt_df.columns)
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
    if "datetime" in str(in_mkt_df["time"].dtype):
        in_mkt_df["time"] = in_mkt_df["time"].dt.date
    if "datetime" in str(in_news_df["time"].dtype):
        in_news_df["time"] = in_news_df["time"].dt.date
        
    set_time = set(in_mkt_df["time"].unique()) | set(in_news_df["time"].unique())
    lst_time = list(set_time)  
    if len(dic_time) == 0:
        dic_time= {k:v for v,k in enumerate(lst_time)}
    else: #dictionary already exists, find only for those elemnts that are extra
        set_time_new = set_time - set(dic_time.keys()) 
        if fname in debug_func:
            print("Need to get sno for " + str(len(set_time_new)) + " news time.")
        time_max_SNo = len(dic_time)
        dic_time_new = {k:v+time_max_SNo for v,k in enumerate(list(set_time_new))}
        dic_time.update(dic_time_new)
    
    in_mkt_df["timeSNo"] = in_mkt_df["time"].map(dic_time)
    in_news_df["timeSNo"] = in_news_df["time"].map(dic_time)
    #-------------------------------------
    if fname in debug_func:
        print("mkt cols dropped: " + str(org_mkt_cols - set(in_mkt_df.columns)))
        print("new mkt cols added: " + str(set(in_mkt_df.columns) - org_mkt_cols))
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
    
def clean_assetName_col(in_mkt_df, in_news_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_mkt_cols = set(in_mkt_df.columns)
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
    in_mkt_df["assetName"] = in_mkt_df["assetName"].str.lower()
    in_mkt_df["assetName"].replace({"inc":"","llc":"","ltd":""}, inplace=True)
    in_mkt_df["assetName"] = in_mkt_df["assetName"].str.strip()

    in_news_df["assetName"] = in_news_df["assetName"].str.lower()
    in_news_df["assetName"].replace({"inc":"","llc":"","ltd":""}, inplace=True)
    in_news_df["assetName"] = in_news_df["assetName"].str.strip()
   #-------------------------------------
    if fname in debug_func:
        print("mkt cols dropped: " + str(org_mkt_cols - set(in_mkt_df.columns)))
        print("new mkt cols added: " + str(set(in_mkt_df.columns) - org_mkt_cols))
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")


# In[ ]:



def set_assetNameCodes_SNo(in_mkt_df, in_news_df,dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_mkt_cols = set(in_mkt_df.columns)
    org_news_cols = set(in_news_df.columns)
    #--------------------------------------
    set_mkt_asset = set(in_mkt_df["assetName"].unique())
    set_news_asset = set(in_news_df["assetName"].unique())
    if fname in debug_func:
        print("Additional asset names in mkt df:" + str(len(set_mkt_asset - set_news_asset)) + " out of a total of " + str(len(set_mkt_asset)) + " assets")
        print("Additional asset names in news df:" + str(len(set_news_asset - set_mkt_asset)) + " out of a total of " + str(len(set_news_asset)) + " assets")

    set_asset = set_news_asset | set_mkt_asset
    set_codes = set(in_news_df["assetCodes"].unique())
    set_code = set(in_mkt_df["assetCode"].unique())
    list_asset = list(set_asset)
    list_codes = list(set_codes)
    list_code = list(set_code)
    
    if len(dic_assetName) == 0:
        dic_assetName= {k:v for v,k in enumerate(list_asset)}
        dic_assetNameSNo= {v:k for v,k in enumerate(list_asset)}
        dic_assetCodes= {k:v for v,k in enumerate(list_codes)}
        dic_assetCode= {k:v for v,k in enumerate(list_code)}
    else: #dictionary already exists, find only for those elemnts that are extra
        set_asset_new = set_asset - set(dic_assetName.keys()) 
        if fname in debug_func:
            print("Need to get sno for " + str(len(set_asset_new)) + " news assts.")
        asset_max_SNo = len(dic_assetName)
        dic_assetName_new = {k:v+asset_max_SNo for v,k in enumerate(list(set_asset_new))}
        dic_assetNameSNo_new = {v+asset_max_SNo:k for v,k in enumerate(list(set_asset_new))}
        dic_assetName.update(dic_assetName_new)
        dic_assetNameSNo.update(dic_assetNameSNo_new)
        
       
        set_assetCode_new = set_code - set(dic_assetCode.keys()) 
        if fname in debug_func:
            print("Need to get sno for " + str(len(set_assetCode_new)) + " news assts codes.")
        assetCode_max_SNo = len(dic_assetCode)
        dic_assetCode_new = {k:v+assetCode_max_SNo for v,k in enumerate(list(set_assetCode_new))}
        dic_assetCodeSNo_new = {v+assetCode_max_SNo:k for v,k in enumerate(list(set_assetCode_new))}
        dic_assetCode.update(dic_assetCode_new)
        dic_assetCodeSNo.update(dic_assetCodeSNo_new)
        
        
        set_codes_new = set_codes - set(dic_assetCodes.keys())
        codes_max_SNo = len(dic_assetCodes)
        dic_assetCodes_new = {k:v+codes_max_SNo for v,k in enumerate(list(set_codes_new))}
        dic_assetCodes.update(dic_assetCodes_new)
        
    in_news_df["assetNameSNo"] = in_news_df["assetName"].map(dic_assetName)
    in_mkt_df["assetNameSNo"] = in_mkt_df["assetName"].map(dic_assetName)
    in_news_df["assetCodesSNo"] = in_news_df["assetCodes"].map(dic_assetCodes)
    in_mkt_df["assetCodeSNo"] = in_mkt_df["assetCode"].map(dic_assetCode)
    if fname in debug_func:
        print("Unique Name SNo in mkt data: " + str(len(in_mkt_df["assetNameSNo"].unique())))
        print("Unique Code SNo in mkt data: " + str(len(in_mkt_df["assetCodeSNo"].unique())))
        print("Unique Name SNo in news data: " + str(len(in_news_df["assetNameSNo"].unique())))
   
#---------------------------------
    #-------------------------------------
    if fname in debug_func:
        print("mkt cols dropped: " + str(org_mkt_cols - set(in_mkt_df.columns)))
        print("new mkt cols added: " + str(set(in_mkt_df.columns) - org_mkt_cols))
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
    return dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo


# In[ ]:


def get_final_assetName(row):
    min_mkt_assetNameSNo = row["min_mkt_assetNameSNo"]
    max_mkt_assetNameSNo = row["max_mkt_assetNameSNo"]
    min_assetNameSNo = row["min_assetNameSNo"]
    max_assetNameSNo = row["max_assetNameSNo"]
    if ((not math.isnan(min_mkt_assetNameSNo)) & (math.isnan(max_mkt_assetNameSNo))):
        return min_assetNameSNo
    elif ((not math.isnan(max_mkt_assetNameSNo)) & (math.isnan(min_mkt_assetNameSNo))):
        return max_assetNameSNo
    elif ((math.isnan(max_mkt_assetNameSNo)) & (math.isnan(min_mkt_assetNameSNo))):
        return max_assetNameSNo
    else:
        return -1
    
def get_replace_assetName(row):
    min_mkt_assetNameSNo = row["min_mkt_assetNameSNo"]
    max_mkt_assetNameSNo = row["max_mkt_assetNameSNo"]
    min_assetNameSNo = row["min_assetNameSNo"]
    max_assetNameSNo = row["max_assetNameSNo"]
    if ((math.isnan(min_mkt_assetNameSNo)) & (not math.isnan(max_mkt_assetNameSNo))):
        return min_assetNameSNo
    elif ((math.isnan(max_mkt_assetNameSNo)) & (not math.isnan(min_mkt_assetNameSNo))):
        return max_assetNameSNo
    elif ((math.isnan(max_mkt_assetNameSNo)) & (math.isnan(min_mkt_assetNameSNo))):
        return min_assetNameSNo
    else:
        return -1
      
def replace_assetName(in_mkt_df, in_news_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_mkt_cols = set(in_mkt_df.columns)
    org_news_cols = set(in_news_df.columns)
    #--------------------------------------
    
    in_news_df["num_rows"] = 1
    assetName_df = in_news_df[["assetCodesSNo","assetNameSNo","num_rows"]].groupby(                                                ["assetCodesSNo","assetNameSNo"], as_index=False).count()
    if fname in debug_func:
        print("Distinct assetcodes, assetname combination: " + str(assetName_df.shape[0]) + " out of " + str(in_news_df.shape[0]) )
    assetCode_df = assetName_df[["assetCodesSNo",                                  "assetNameSNo","num_rows"]].groupby(["assetCodesSNo"],                                  as_index=False).agg({"assetNameSNo": ["max","min"], "num_rows": "sum"})

    assetCode_df.columns = ["assetCodesSNo", "max_assetNameSNo", "min_assetNameSNo", "count_num_rows"]

    assetCode_df = assetCode_df[assetCode_df["max_assetNameSNo"] != assetCode_df["min_assetNameSNo"]]
    num_with_duplicate_assetName = assetCode_df.shape[0]
    if fname in debug_func:
        print("Rows with different assetname for same assetcode : " + str(num_with_duplicate_assetName) + " out of " + str(assetName_df.shape[0]) )
    in_news_df.drop(["num_rows"], axis=1, inplace=True)
    
    if assetCode_df.shape[0] > 0:
        assetCode_df["max_assetName"] = assetCode_df["max_assetNameSNo"].map(dic_assetNameSNo)
        assetCode_df["min_assetName"] = assetCode_df["min_assetNameSNo"].map(dic_assetNameSNo)
        list_mkt_asset = list(in_mkt_df["assetName"].unique())
        dic_mkt_assetName= {k:v for v,k in enumerate(list_mkt_asset)}
        assetCode_df["min_mkt_assetNameSNo"] = assetCode_df["min_assetName"].map(dic_mkt_assetName)
        assetCode_df["max_mkt_assetNameSNo"] = assetCode_df["max_assetName"].map(dic_mkt_assetName)


        assetCode_df["final_assetNameSNo"] = assetCode_df.apply(get_final_assetName, axis=1)
        assetCode_df["replace_assetNameSNo"] = assetCode_df.apply(get_replace_assetName, axis=1)
        missing_in_mkt = assetCode_df[assetCode_df["final_assetNameSNo"]==-1].shape[0]
        if fname in debug_func:
            print("None of the asset names exist in market df for " + str(missing_in_mkt) + " entries")
        if num_with_duplicate_assetName == missing_in_mkt:
            df1 = assetCode_df[["assetCodesSNo","max_assetNameSNo"]]
            df1.columns = ["assetCodesSNo","assetNameSNo"]
            df2 = assetCode_df[["assetCodesSNo","min_assetNameSNo"]]
            df2.columns = ["assetCodesSNo","assetNameSNo"]
            tot_df = pd.concat([df1,df2])
            rows_to_be_dropped = pd.merge(in_news_df.reset_index(), tot_df, how="inner", on=["assetCodesSNo", "assetNameSNo"])
            print(rows_to_be_dropped.index.unique())
            print("dropping rows:" + str(rows_to_be_dropped.shape[0]))
            in_news_df.drop(rows_to_be_dropped["index"].unique(), axis=0, inplace=True)
        
        assetCode_df["final_assetName"] = assetCode_df["final_assetNameSNo"].map(dic_assetNameSNo)
        assetCode_df = assetCode_df[assetCode_df["final_assetNameSNo"]!=-1]
        df_temp = pd.merge(in_news_df.reset_index()[["index","assetCodesSNo","assetNameSNo"]],                            assetCode_df[["replace_assetNameSNo", "final_assetNameSNo","assetCodesSNo","final_assetName"]],                            how="inner", left_on=["assetCodesSNo","assetNameSNo"], right_on=["assetCodesSNo","replace_assetNameSNo"], left_index=True).set_index("index")
        in_news_df.loc[df_temp.index, "assetNameSNo"] = list(df_temp["final_assetNameSNo"])
        in_news_df.loc[df_temp.index, "assetName"] = list(df_temp["final_assetName"])
        
        in_news_df["num_rows"] = 1
        assetName_df = in_news_df[["assetCodesSNo","assetNameSNo","num_rows"]].groupby(                                                    ["assetCodesSNo","assetNameSNo"], as_index=False).count()
        if fname in debug_func:
            print("After replacement, Distinct assetcodes, assetname combination: " + str(assetName_df.shape[0]) + " out of " + str(in_news_df.shape[0]) )
       
        assetCode_df = assetName_df[["assetCodesSNo",                                      "assetNameSNo","num_rows"]].groupby(["assetCodesSNo"],                                      as_index=False).agg({"assetNameSNo": ["max","min"], "num_rows": "sum"})

        assetCode_df.columns = ["assetCodesSNo", "max_assetNameSNo", "min_assetNameSNo", "count_num_rows"]

        assetCode_df = assetCode_df[assetCode_df["max_assetNameSNo"] != assetCode_df["min_assetNameSNo"]]
        if fname in debug_func:
            print("After replacement, Rows with different assetname for same assetcode : " + str(assetCode_df.shape[0]) + " out of " + str(assetName_df.shape[0]) )
        
        in_news_df.drop(["num_rows"], axis=1, inplace=True)
    in_news_df.drop(["assetCodes","assetCodesSNo"], inplace=True, axis=1)
#-------------------------------------
    if fname in debug_func:
        print("mkt cols dropped: " + str(org_mkt_cols - set(in_mkt_df.columns)))
        print("new mkt cols added: " + str(set(in_mkt_df.columns) - org_mkt_cols))
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
        
        


# In[ ]:





# In[ ]:



def scale_col(in_news_df, col):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
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
    #-------------------------------------
    if fname in debug_func:
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
        
def drop_extra_news_cols(in_news_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
    news_drop_cols = ["sourceTimestamp","firstCreated","sourceId","headline","urgency","takeSequence","provider","audiences","bodySize",                      "headlineTag","sentenceCount","wordCount","firstMentionSentence","sentimentClass","sentimentWordCount","companyCount", "marketCommentary",
                      "volumeCounts12H","volumeCounts24H","volumeCounts3D","volumeCounts5D","volumeCounts7D"]

    news_drop_cols = [col  for col in news_drop_cols if col in in_news_df.columns ]
    in_news_df.drop(news_drop_cols, axis=1, inplace=True)
    #-------------------------------------
    if fname in debug_func:
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
    
    
def drop_irrelevant_news(in_news_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
    if "relevance" in in_news_df.columns:
        irrelevant_news = in_news_df[in_news_df["relevance"] < 0.3].index
        print("Num low relevance news:", len(irrelevant_news))
        in_news_df.drop(irrelevant_news, axis=0, inplace=True)
        del irrelevant_news
        gc.collect()
    #-------------------------------------
    if fname in debug_func:
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
 

def drop_neutral_news(in_news_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
    if "sentimentPositive" in in_news_df.columns:
        in_news_df["netSentimentPositive"] = (in_news_df["sentimentPositive"] - in_news_df["sentimentNegative"]) 
        neutral_news = in_news_df[in_news_df["netSentimentPositive"].abs() < 0.05].index
        in_news_df.drop(neutral_news, axis=0, inplace=True)
        in_news_df["netSentimentPositive"] = in_news_df["netSentimentPositive"] * (1 - in_news_df["sentimentNeutral"])
        print("Num neutral news:", len(neutral_news))
        del neutral_news
        gc.collect()
        in_news_df.drop(["sentimentNegative","sentimentNeutral","sentimentPositive"], axis=1, inplace=True)
   #-------------------------------------
    if fname in debug_func:
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")

    
def gen_netnovelty_col(in_news_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
    # if there is a news in 5 to 7 days then count will be more than 1. This news should have less importance as it is an old news
    if "noveltyCount12H" in in_news_df.columns:
        arr_novelty = np.array(in_news_df[["noveltyCount12H","noveltyCount24H","noveltyCount3D","noveltyCount5D","noveltyCount7D"]])

        in_news_df["inv_netNovelty"] = list(1/(np.argmax(arr_novelty, axis=1)+2))
        col = "inv_netNovelty"
        scale_col(in_news_df, col)
        in_news_df.drop(["noveltyCount12H","noveltyCount24H","noveltyCount3D","noveltyCount5D","noveltyCount7D"], axis=1, inplace=True)
        in_news_df["SenSR"] = in_news_df["netSentimentPositive"] * in_news_df["relevance"] * (in_news_df["inv_netNovelty"])
        in_news_df.drop(["relevance","netSentimentPositive","inv_netNovelty"],axis=1, inplace=True)
  #-------------------------------------
    if fname in debug_func:
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
 


# In[ ]:





# In[ ]:



def create_sector(in_asset_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_news_cols = set(in_asset_df.columns)
    #----------------------------------
    global dic_asset_sector
    global set_sector 
    global sector_col
    global dic_sectorSNo
    
    in_asset_df["sector"] = in_asset_df["subjects"].apply(lambda x: str(eval(x) & set_sector))
    for col in set_sector:
        #print(col)
        in_asset_df[col] = in_asset_df["sector"].apply(lambda x: col in eval(x))

    asset_df = in_asset_df[sector_col].groupby(["assetNameSNo"]).sum()
    asset_df1 = asset_df.sum(axis=1)
    asset_unknown_sector = list(asset_df1[asset_df1==0].index)
    if len(asset_unknown_sector) > 0:
        print("Could not find sector of ", len(asset_unknown_sector) , " assets")
    #asset_df.drop(asset_unknown_sector, axis=0, inplace=True)
    asset_df["sectorNameSNo"] = asset_df.values.argmax(axis=1)
    asset_df["sectorName"] = asset_df["sectorNameSNo"].map(dic_sectorSNo)
    asset_df.loc[asset_unknown_sector,["sectorNameSNo","sectorName"]] = [-1,"Unknown"]
    asset_df = asset_df[["sectorName","sectorNameSNo"]]
    asset_df.reset_index(inplace=True)
    #-------------------------------------
    if fname in debug_func:
        print("news cols dropped: " + str(org_news_cols - set(in_asset_df.columns)))
        print("new news cols added: " + str(set(in_asset_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
    return asset_df
        

def get_sector_name(in_news_df, forpred=False):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_news_cols = set(in_news_df.columns)
    #--------------------------------------
    global dic_asset_sector
    if "sectorName" not in in_news_df.columns:
        in_news_df["sectorName"] = ""
        in_news_df["sectorNameSNo"] = -1
    if len(dic_asset_sector) > 0:
        in_news_df["sectorName"] = in_news_df["assetNameSNo"].map(dic_asset_sector)
        #if forpred==True:
        #    in_news_df["sectorName"].fillna("Unknown", inplace=True)
        #else:
        in_news_df["sectorName"].fillna("", inplace=True)
        in_news_df["sectorNameSNo"] = in_news_df["sectorName"].map(dic_sector)  
    if 1==1: #forpred==False:  
        rec_index = in_news_df[in_news_df["sectorName"] == ""].index
        lst_subjects = in_news_df.loc[rec_index]["subjects"].unique()
        dic_subjects = {k:v for v,k in enumerate(list(lst_subjects))}
        dic_subjectsSNo = {v:k for v,k in enumerate(list(lst_subjects))}
        in_news_df["subjectsSNo"] = -1
        in_news_df.loc[rec_index,["subjectsSNo"]] = in_news_df.loc[rec_index]["subjects"].map(dic_subjects)
        in_asset_df = in_news_df.loc[rec_index]
    
        in_asset_df["tempCol"] = 1
        in_asset_df = in_asset_df[["assetNameSNo","subjectsSNo","tempCol"]].groupby(["assetNameSNo","subjectsSNo"], as_index=False).first()
        in_asset_df.drop(["tempCol"], inplace=True, axis=1)
        in_asset_df["subjects"] = in_asset_df["subjectsSNo"].map(dic_subjectsSNo)
    
        if fname in debug_func:
            print("New asset records in news : " + str(in_asset_df.shape[0]) + " out of " + str(in_news_df.shape[0]) )
            print("***************************")
        all_news = []
        if in_asset_df.shape[0] > 0:
            if forpred==False:
                maxSNo = in_asset_df["assetNameSNo"].max()
                partition_size = (maxSNo // 4) + 1

                for i in range(4):
                    minr = i*partition_size
                    maxr = (i+1)*partition_size
                    all_news.append(in_asset_df[(in_asset_df["assetNameSNo"] >= minr) & (in_asset_df["assetNameSNo"] < maxr)])

                pool = Pool(4)
                all_df = pool.map(create_sector, all_news)

                asset_df = pd.concat(all_df)  
                pool.close()
            else:
                asset_df = create_sector(in_asset_df)

            asset_df1 = asset_df.set_index(["assetNameSNo"])[["sectorName"]]
            new_dict = asset_df1.to_dict()["sectorName"]
            dic_asset_sector.update(new_dict)
          
            #print(dic_asset_sector)

            merged_df = pd.merge(in_news_df.reset_index()[["assetNameSNo","index"]], asset_df[["assetNameSNo","sectorName","sectorNameSNo"]],                                  on=["assetNameSNo"], how="left").set_index("index")
            #print(merged_df.head(1).T)
            in_news_df.loc[merged_df.index,["sectorName","sectorNameSNo"]] = merged_df.loc[merged_df.index,["sectorName","sectorNameSNo"]]
    if 1==1:  
        if "subjects" in in_news_df.columns:
            in_news_df.drop("subjects", axis=1, inplace=True)
        
    #print(in_news_df["sectorNameSNo"].unique())
    #print(in_news_df["sectorName"].unique())
    #-------------------------------------
    if fname in debug_func:
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")


# In[ ]:



def merge_mkt_news(in_mkt_df, in_news_df):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_mkt_cols = set(in_mkt_df.columns)
    org_news_cols = set(in_news_df.columns)
    #----------------------------------
    global dic_asset_sector
    in_mkt_df["sectorName"] = in_mkt_df["assetNameSNo"].map(dic_asset_sector)
    df_unknown_sector = in_mkt_df[in_mkt_df["sectorName"].isna()]
    in_mkt_df["sectorName"].fillna("Unknown", inplace=True)
    print("no of market recs with unresolved sector: " + str(len(df_unknown_sector["assetNameSNo"].unique())))
    #in_mkt_df.drop(df_unknown_sector.index, inplace=True)
    
    in_mkt_df["sectorNameSNo"] = in_mkt_df["sectorName"].map(dic_sector)
    
    in_mkt_df["mkt_cap"] = ((in_mkt_df["close"] + in_mkt_df["open"])/2)  * in_mkt_df["volume"] 
    
    
    in_news_df1 = in_news_df[["timeSNo","assetNameSNo","SenSR"]].groupby(["assetNameSNo","timeSNo"], as_index=False).sum()
    if "SenSR" in in_mkt_df.columns:
        in_mkt_df.drop(["SenSR"], axis=1, inplace=True)
    merged_df = pd.merge(in_mkt_df.reset_index(), in_news_df1,  on= ["assetNameSNo","timeSNo"], how="inner").set_index("index")
    in_mkt_df["SenSR"] = 0
    in_mkt_df.loc[merged_df.index, ["SenSR"]] = merged_df.loc[merged_df.index, ["SenSR"]]

    in_mkt_df["weighted_SenSR"] = in_mkt_df["SenSR"] * in_mkt_df["mkt_cap"]
    #---------------------------------------------------
    if fname in debug_func:
        print("mkt cols dropped: " + str(org_mkt_cols - set(in_mkt_df.columns)))
        print("new mkt cols added: " + str(set(in_mkt_df.columns) - org_mkt_cols))
        print("news cols dropped: " + str(org_news_cols - set(in_news_df.columns)))
        print("new news cols added: " + str(set(in_news_df.columns) - org_news_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")


# In[ ]:


market_train_df.head(1)


# In[ ]:


def create_lag(df_code,n_lag=[3,7,14,],shift_size=1):
    code = df_code['assetCodeSNo'].unique()
    
    for col in return_features:
        for window in n_lag:
            rolled = df_code[col].shift(shift_size).rolling(window=window)
            lag_mean = rolled.mean()
            lag_max = rolled.max()
            lag_min = rolled.min()
            lag_std = rolled.std()
            df_code['%s_lag_%s_mean'%(col,window)] = lag_mean
            df_code['%s_lag_%s_max'%(col,window)] = lag_max
            df_code['%s_lag_%s_min'%(col,window)] = lag_min
#             df_code['%s_lag_%s_std'%(col,window)] = lag_std
    return df_code.fillna(-1)

def generate_lag_features(df,n_lag = [3,7,14]):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_mkt_cols = set(df.columns)
    #----------------------------------
    all_df = []
    df_codes = df.groupby('assetCodeSNo')
    df_codes = [df_code[1][['timeSNo','assetCodeSNo']+return_features] for df_code in df_codes]
    
    
    pool = Pool(4)
    all_df = pool.map(create_lag, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(return_features,axis=1,inplace=True)
    pool.close()
     #-------------------------------------
    if fname in debug_func:
        print("mkt cols dropped: " + str(org_mkt_cols - set(df.columns)))
        print("new mkt cols added: " + str(set(df.columns) - org_mkt_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
    return new_df


# In[ ]:


def create_sector_lag(df_code,n_lag=[3,7,14,],shift_size=1):
    for col in sector_return_features:
        for window in n_lag:
            if df_code["sectorNameSNo"].unique()[0] == -1:
                df_code['%s_lag_%s_mean'%(col,window)] = -1
                df_code['%s_lag_%s_max'%(col,window)] = -1
                df_code['%s_lag_%s_min'%(col,window)] = -1
            else:
                rolled = df_code[[col,"mkt_cap"]].shift(shift_size).rolling(window=window)
                lag_mean = rolled[col].sum()/ rolled["mkt_cap"].sum()
                lag_max = rolled[col].max()/ rolled["mkt_cap"].sum()
                lag_min = rolled[col].mean()/ rolled["mkt_cap"].sum()
                df_code['%s_lag_%s_mean'%(col,window)] = lag_mean
                df_code['%s_lag_%s_max'%(col,window)] = lag_max
                df_code['%s_lag_%s_min'%(col,window)] = lag_min
    #             df_code['%s_lag_%s_std'%(col,window)] = lag_std
    return df_code.fillna(-1)

def generate_sector_lag_features(df,n_lag = [3,7,14]):
    fname = sys._getframe().f_code.co_name
    if fname in debug_func:
        print("Start " + fname)
    start = datetime.now()
    org_mkt_cols = set(df.columns)
    #----------------------------------
    all_df = []
    df_codes = df[['timeSNo','sectorNameSNo',"mkt_cap"] + sector_return_features].groupby(["timeSNo",'sectorNameSNo'], as_index=False).sum()
    df_codes = df_codes.groupby('sectorNameSNo')
    df_codes = [df_code[1][['timeSNo','sectorNameSNo',"mkt_cap"] + sector_return_features] for df_code in df_codes]
    #print('total %s df'%len(df_codes))
    
    pool = Pool(4)
    all_df = pool.map(create_sector_lag, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(sector_return_features + ["mkt_cap"],axis=1,inplace=True)
    pool.close()
    #-------------------------------------
    if fname in debug_func:
        print("mkt cols dropped: " + str(org_mkt_cols - set(df.columns)))
        print("new mkt cols added: " + str(set(df.columns) - org_mkt_cols))
        end = datetime.now()
        print("End " + fname + ": " + str(end-start) + "---------------------")
    return new_df.reset_index()


# In[ ]:



fix_time_col(market_train_df, news_train_df)
market_train_df = market_train_df.loc[market_train_df['time']>=date(2010, 1, 1)]
news_train_df = news_train_df.loc[news_train_df['time']>=date(2010, 1, 1)]
clean_assetName_col(market_train_df, news_train_df)
dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo = set_assetNameCodes_SNo(market_train_df, news_train_df,dic_assetName,                                                                                                           dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo)

replace_assetName(market_train_df, news_train_df)
drop_extra_news_cols(news_train_df)
drop_irrelevant_news(news_train_df)
drop_neutral_news(news_train_df)
gen_netnovelty_col(news_train_df)



# In[ ]:


get_sector_name(news_train_df)
merge_mkt_news(market_train_df, news_train_df)


# In[ ]:


# return_features = ['close']
# new_df = generate_lag_features(market_train_df,n_lag = 5)
# market_train_df = pd.merge(market_train_df,new_df,how='left',on=['time','assetCode'])


# In[ ]:


import gc
del news_train_df
gc.collect()


# In[ ]:


market_train_df.tail(1)


# In[ ]:



new_df = generate_lag_features(market_train_df,n_lag=n_lag)
market_train_df = pd.merge(market_train_df,new_df,how='left',on=['timeSNo','assetCodeSNo'])


# In[ ]:


new_df = generate_sector_lag_features(market_train_df,n_lag=n_lag)
market_train_df = pd.merge(market_train_df,new_df,how='left',on=['timeSNo','sectorNameSNo'])


# In[ ]:


def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

market_train_df = mis_impute(market_train_df)


# In[ ]:


for col in market_train_df:
    cols = market_train_df[col]
    print("Market df: Num nulls in " + col + " is " + str(cols[cols.isna()].shape[0]) + " out of " + str(cols.shape[0]))


# In[ ]:


market_train_df.head(10)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

up = market_train_df['returnsOpenNextMktres10'] >= 0


universe = market_train_df['universe'].values
d = market_train_df['time']

fcol = [c for c in market_train_df if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', "timeSNo",
                                                'universe','sourceTimestamp', "sectorName","index","mkt_cap","open","close"]]


X = market_train_df[fcol].values
up = up.values
r = market_train_df.returnsOpenNextMktres10.values
del market_train_df
gc.collect()

# Scaling of X values
# It is good to keep these scaling values for later
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)

# Sanity check
assert X.shape[0] == up.shape[0] == r.shape[0]

from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time

X_train, X_test, up_train, up_test, r_train, r_test,u_train,u_test,d_train,d_test = model_selection.train_test_split(X, up, r,universe,d,test_size=0.25, random_state=99)
print(X_train.shape)

train_data = lgb.Dataset(X, label=up.astype(int), feature_name=fcol, categorical_feature=["assetCodeSNo","assetNameSNo","sectorNameSNo"], free_raw_data=False)
test_data = lgb.Dataset(X_test, label=up_test.astype(int), feature_name=fcol, categorical_feature=["assetCodeSNo","assetNameSNo","sectorNameSNo"], free_raw_data=False)


# In[ ]:


del X, X_train
gc.collect()


# In[ ]:


fcol


# In[ ]:


#for var, obj in locals().items():
#    print (var, sys.getsizeof(obj))


# In[ ]:


# these are tuned params I found
x_1 = [0.19000424246380565, 2452, 212, 328, 202]
x_2 = [0.19016805202090095, 2583, 213, 312, 220]

def exp_loss(p,y):
    y = y.get_label()
#     p = p.get_label()
    grad = -y*(1.0-1.0/(1.0+np.exp(-y*p)))
    hess = -(np.exp(y*p)*(y*p-1)-1)/((np.exp(y*p)+1)**2)
    
    return grad,hess

params_1 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
#         'objective': 'regression',
        'learning_rate': x_1[0],
        'num_leaves': x_1[1],
        'min_data_in_leaf': x_1[2],
#         'num_iteration': x_1[3],
        'num_iteration': 239,
        'max_bin': x_1[4],
        'verbose': 1
    }

params_2 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
#         'objective': 'regression',
        'learning_rate': x_2[0],
        'num_leaves': x_2[1],
        'min_data_in_leaf': x_2[2],
#         'num_iteration': x_2[3],
        'num_iteration': 272,
        'max_bin': x_2[4],
        'verbose': 1
    }

gbm_1 = lgb.train(params_1,
        train_data,
        num_boost_round=100,
        valid_sets=test_data,
        early_stopping_rounds=5,
#         fobj=exp_loss,
        )

gbm_2 = lgb.train(params_2,
        train_data,
        num_boost_round=100,
        valid_sets=test_data,
        early_stopping_rounds=5,
#         fobj=exp_loss,
        )


# In[ ]:


lgb.plot_importance(gbm_1, importance_type='split', max_num_features=20)


# In[ ]:


lgb.plot_importance(gbm_1, importance_type='gain', max_num_features=20)


# In[ ]:


confidence_test = (gbm_1.predict(X_test) + gbm_2.predict(X_test))/2
confidence_test = (confidence_test-confidence_test.min())/(confidence_test.max()-confidence_test.min())
confidence_test = confidence_test*2-1
print(max(confidence_test),min(confidence_test))

# calculation of actual metric that is used to calculate final score
r_test = r_test.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_test * r_test * u_test
data = {'day' : d_test, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_test = mean / std
print(score_test)


# In[ ]:


del X_test, d_train
gc.collect()


# In[ ]:


def process_mkt_news_data(market_obs_df, news_obs_df):   
    global dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo
    fix_time_col(market_obs_df, news_obs_df)
    curtime = market_obs_df["time"].unique()
    print(curtime)
    found=1
    if len(total_market_obs_df)>1:
        while found==1:
            prev_df = total_market_obs_df[-1]
            if prev_df["time"].unique() == curtime:
                print("removing------------------------")
                total_market_obs_df.pop(-1)
                total_news_obs_df.pop(-1)
            else:
                found=0
    clean_assetName_col(market_obs_df, news_obs_df)
    dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo = set_assetNameCodes_SNo(market_obs_df, news_obs_df,dic_assetName, dic_assetNameSNo,                                                                                             dic_assetCodes,dic_assetCode, dic_assetCodeSNo)
    drop_extra_news_cols(news_obs_df)
    drop_irrelevant_news(news_obs_df)
    drop_neutral_news(news_obs_df)
    gen_netnovelty_col(news_obs_df)
    get_sector_name(news_obs_df, True)
    
    return dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo, curtime


# In[ ]:



if len(total_market_obs_df) == 0:
    for i in range(len(prv_mkt_data)):
        mkt_data = prv_mkt_data[i]
        news_data = prv_news_data[i]
        mkt_data.drop(["universe","returnsOpenNextMktres10"], inplace=True, axis=1)
        dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo, curtime = process_mkt_news_data(mkt_data, news_data)
        total_market_obs_df.append(mkt_data)
        total_news_obs_df.append(news_data)
def make_random_predictions_new(market_obs_df, news_obs_df,predictions_template_df):  
    global n_days
    global prep_time
    global prediction_time
    global packaging_time
    global total_market_obs_df
    global total_news_obs_df
    global dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo
    n_days +=1
    if (n_days%50==0):
        print(n_days,end=' ')
    t = time.time()

    dic_assetName, dic_assetNameSNo, dic_assetCodes, dic_assetCode, dic_assetCodeSNo, curtime = process_mkt_news_data(market_obs_df, news_obs_df)

    total_market_obs_df.append(market_obs_df)
    total_news_obs_df.append(news_obs_df)
    if len(total_market_obs_df)==1:
        history_df = total_market_obs_df[0]
        history_news_df = total_news_obs_df[0]
    else:
        history_df = pd.concat(total_market_obs_df[-(np.max(n_lag)+1):], sort=False)
        history_news_df = pd.concat(total_news_obs_df[-(np.max(n_lag)+1):], sort=False)
    history_df.sort_values(["timeSNo"], inplace=True)
    history_df.index = list(range(history_df.shape[0]))
    history_news_df.sort_values(["timeSNo"], inplace=True)
    history_news_df.index = list(range(history_news_df.shape[0]))
    fixopenclose(history_df)
    merge_mkt_news(history_df, history_news_df)
    
    new_df = generate_lag_features(history_df,n_lag=[3,7,14])
    history_df = pd.merge(history_df,new_df,how='left',on=['timeSNo','assetCodeSNo'])
    new_df = generate_sector_lag_features(history_df,n_lag=[3,7,14])
    history_df = pd.merge(history_df,new_df,how='left',on=['timeSNo','sectorNameSNo'])
    history_df = mis_impute(history_df)
    history_df.dropna(inplace=True)
    market_obs_df = history_df[history_df["time"].isin(curtime)]
    X_live = market_obs_df[fcol].values
    X_live = 1 - ((maxs - X_live) / rng)
   
    prep_time += time.time() - t
    
    t = time.time()
    lp = (gbm_1.predict(X_live) + gbm_2.predict(X_live))/2
    prediction_time += time.time() -t
    
    t = time.time()
    
    confidence = lp
    confidence = (confidence-confidence.min())/(confidence.max()-confidence.min())
    confidence = confidence * 2 - 1
    
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    return market_obs_df


# In[ ]:


market_obs_df, news_obs_df, predictions_template_df = next(days)


# In[ ]:



market_obs_df = make_random_predictions_new(market_obs_df, news_obs_df,predictions_template_df)


# In[ ]:


market_obs_df.head(1)


# In[ ]:




for (market_obs_df, news_obs_df, predictions_template_df) in days:
    market_obs_df = make_random_predictions_new(market_obs_df, news_obs_df,predictions_template_df)
    
print('Done!')


# In[ ]:


#prediction

    
    
env.write_submission_file()
sub  = pd.read_csv("submission.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




