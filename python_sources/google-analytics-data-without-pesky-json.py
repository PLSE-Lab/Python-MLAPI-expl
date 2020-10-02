#!/usr/bin/env python
# coding: utf-8

# ## Google Analytics Customer Revenue Prediction Data Rewrite
# This kernel is merely show how to convert the train.csv and test.csv from large cumbersome with JSON columns to the lighter newtrain.csv and newtest.csv, where the nested JSON entries have been expanded (flattened) into extra columns. Since all JSON field name repetition is removed the resultant files are approximately 1GB lighter (eg. 1.5GB to 250MG for train.csv). Though no information has been lost, the resultant files are in UTF-8 format instead of ASCII.
# 
# This kernel is set up to do the conversion from original data to new data and write it to your working directory, even though dataset **"GA data with json columns"** is already included (it purposefully fails to find the new dataset and proceeds to load and convert the original one). To change this behaviour and load the new dataset immediately uncomment the line  `newdata_dir = "../input/ga-analytics-with-json-columns"` in the first code cell.
# 
# #### dtypes/type changes after reloading data
# It is worth noting that a reload of the manipulated data (eg. `newtrain = pd.read_csv(newtrain_path)`) assigns more appropriate dtypes and types (to columns and elements respectively) than the first csv read. The sequence is (1) read original csv into DataFrame `train`; (2) which is then copied to `newtrain`; (3) JSON fields are added; (4) `newtrain` is written to a new csv; (5) `newtrain` is reloaded from the new csv. Only after (5) is done do we see appropriate values for many columns.
# * Eg: `totals.transactionRevenue` starts out life as an **object** with elements of type **&lt;class 'float'>**, but once reloaded the dtype and type change to **float64** and **&lt;class 'numpy.float'>** respectively. The latter change was important when using `pd.sum()`.
# 
# #### dtype of fullVisitorId
# Many `fullVisitorId`s have leading zeros, which are stripped when a csv is read without forcing the dtype(str). Since the example submission contains leading zeros we suppose we must keep the zeros.
# 
# #### extra stuff
# After the reload, there is some extra stuff which doesn't affect the new cvs such as dropping columns filled only with NaNs. Also, there some rudimentary EDA pies.

# ## Data overview
# 
# _NOTE: Jupyter has removed support for :--- to left justify so have had to use `<p align="left">`_<br>
# _NOTE: JSON indicates many more columns will be added once these columns with json fields have been expanded_
# 
# Field|<p align="left">Description
# ---|:---
#  fullVisitorId        | <p align="left">A unique identifier for each user of the Google Merchandise Store
#  channelGrouping      |<p align="left"> The channel via which the user came to the Store
#  date                 |<p align="left"> The date on which the user visited the Store
#  device               |<p align="left"> (JSON) The specifications for the device used to access the Store
#  geoNetwork           |<p align="left"> (JSON) This section contains information about the geography of the user
#  sessionId            |<p align="left"> A unique identifier for this visit to the store
#  socialEngagementType |<p align="left"> Engagement type, either "Socially Engaged" or "Not Socially Engaged"
#  totals               |<p align="left"> (JSON) This section contains aggregate values across the session
#  trafficSource        |<p align="left"> (JSON) This section contains information about the Traffic Source from which the session originated. 
#  visitId              |<p align="left"> An identifier for this session. This is part of the value usually stored as the `_utmb` cookie, uique only to user. For unique visit ID use: fullVisitorId and visitId
#  visitNumber          |<p align="left"> The session number for this user. If this is the first session, then this is set to 1
#  visitStartTime       |<p align="left"> The timestamp (expressed as POSIX time)
#      
# 

# ##### Useful websites
# https://pandas.pydata.org/pandas-docs/stable/api.html#flat-file

# ## Initialisation
# * import the usual helpful libs
# * initialise path names (using multiple locations)
# * read into newtrain and newtest if available, otherwise train and test

# In[ ]:


import os
import sys
import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import time

warnings.filterwarnings('ignore')
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_colwidth', 90)

def set_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path

newdata_dir = "."
#newdata_dir = "../input/ga-analytics-with-json-columns"
data_dir1   = "../input/ga-customer-revenue-prediction"
data_dir2   = "../input"
newtrain_path = newdata_dir+"/newtrain.csv"
newtest_path  = newdata_dir+"/newtest.csv"
train_path    = set_path([data_dir1+"/train.csv", data_dir2+"/train.csv"])
test_path     = set_path([data_dir1+"/test.csv",  data_dir2+"/test.csv"])

get_ipython().system('ls -ld $newdata_dir/*.csv $data_dir1/*.csv $data_dir2/*.csv')


# In[ ]:


def load_new_or_orig(newpath, path):
    new = None
    orig = None
    if os.path.exists(newpath):
        new = pd.read_csv(newpath, dtype={'fullVisitorId': 'str', 'visitId': 'str'})
        print ("loaded",newpath)
    elif os.path.exists(path):
        orig = pd.read_csv(path, dtype={'fullVisitorId': 'str', 'visitId': 'str'})
        print ("loaded",path)
    else:
        print ("ERROR: loaded nothing")
    return new, orig

newtrain, train = load_new_or_orig(newtrain_path, train_path)
newtest,  test  = load_new_or_orig(newtest_path, test_path)


# ## Generic data parsing functions
# * summary: columns for "describe" and default ("basic") should be self-explanatory
# * is_json: not useful at present since I call get_json with specific fields
# * get_json: apply(json.loads) ensures True/False are ok, but I have to convert strings to NaN explicitly
# <br>*NOTE: normalize will automatically quote json fields that contain a , (comma) so it's ok to convert the result back to a csv*
# * expand_json: add new columns using flattened nested json columns, drop duplicates and return a new df

# In[ ]:


def summary(df, info="describe"):
    if info == "describe":
        headings=['Null','Unique','dType','Type','MinMax','Mean','Std','Skew','Examples']
    else:
        headings=['Null','Unique','dType','Type','Examples']
        
    print('DataFrame shape',df.shape)
    sdf = pd.DataFrame(index=df.columns, columns=headings)
    for col in df.columns:
        sdf['Null'][col]     = df[col].isna().sum()
        sdf['Unique'][col]   = df[col].astype(str).unique().size
        sdf['dType'][col]    = df[col].dtype
        sdf['Type'][col]     = "-" if df[col].notna().sum() == 0 else type(df[col].dropna().iloc[0])
        sdf['Examples'][col] = "-" if df[col].notna().sum() == 0 else df[col].astype(str).unique() #.dropna().values
        if info == "describe":
            if 'float' in str(df[col].dtype) or 'int' in str(df[col].dtype):
                sdf['MinMax'][col] = str(round(df[col].min(),2))+'/'+str(round(df[col].max(),2))
                sdf['Mean'][col]   = df[col].mean()
                sdf['Std'][col]    = df[col].std()
                sdf['Skew'][col]   = df[col].skew()
    return sdf.fillna('-')


def is_json(j):
    if re.match(r'^{\"', j):
        return True
    else:
        return False

def get_json(df, col):
    if is_json(df[col][0]) == False:
        return None
    jdf_lines = df[col].apply(json.loads)   # do normalize separately or it will use just one column
    jdf = pd.io.json.json_normalize(jdf_lines).add_prefix(col+'.')
    for jcol in jdf.columns:
        jdf[jcol].replace('not available in demo dataset', np.nan, inplace=True, regex=True)
        jdf[jcol].replace('(not provided)', np.nan, inplace=True, regex=True)
        jdf[jcol].replace('(not set)', np.nan, inplace=True, regex=True)
    return jdf

def expand_json(df):
    newdf = pd.concat([df, get_json(df, 'device'),
                           get_json(df, 'geoNetwork'),
                           get_json(df, 'totals'),
                           get_json(df, 'trafficSource')], axis=1, sort=False)
    newdf.drop(columns=['device', 'geoNetwork', 'totals', 'trafficSource'], inplace=True)
    return newdf


# In[ ]:


# Adhoc validation
#get_json(train, 'trafficSource')
#get_json(train, 'fullVisitorId')
#summary(train, info="basic")
#summary(get_json(train, 'device'),       info="basic")
#summary(get_json(train, 'geoNetwork'),    info="basic")
#summary(get_json(train, 'totals'),        info="basic")
#summary(get_json(train, 'trafficSource'), info="basic")


# ## Create wider data frames using nested json entries
# * create newtrain/newtest and then save them as csvs; they will be smaller since we have removed duplicate json keys

# In[ ]:


def expand_json_to_df(newdf, df):
    newname = sys._getframe(1).f_code.co_names[1]  # [0] is the function name
    if newdf is None:
        print (time.ctime(), newname, "will contain expanded json entries")
        newdf = expand_json(df)
        print (time.ctime(), newname, "finished: shape =", newdf.shape, "vs original shape =", df.shape)
    else:
        print (time.ctime(), newname, "already loaded")
    return newdf

newtrain = expand_json_to_df(newtrain, train)
newtest  = expand_json_to_df(newtest, test)


# In[ ]:


summary(newtrain, info="basic")


# In[ ]:


summary(newtest, info="basic")


# ## Rewrite and Reload
# * if the directory is writable, write newtrain and newtest if they don't already exist
# * reload back into newtrain and newtest which allows pandas.read_csv to assign more appropriate types (see explanation in heading)
# <br>*NOTE: python3's default ecoding is "utf-8", but explicitly set it so we know what we're getting*

# In[ ]:


# newtrain.csv is only 250MB (from 1.5GB) because we've removed the json repetition
# newtest.csv is only 230MB (from 1.3GB) because we've removed the json repetition

if os.access(newdata_dir, os.W_OK):
    if not os.path.exists(newtrain_path):
        newtrain.to_csv(newtrain_path, index=False, encoding="utf-8")
        print ("wrote", newtrain_path)
        newtrain = pd.read_csv(newtrain_path)
        print ("reloaded newtrain")
    if not os.path.exists(newtest_path):
        newtest.to_csv(newtest_path, index=False, encoding="utf-8")
        print ("wrote", newtest_path)
        newtest = pd.read_csv(newtest_path)
        print ("reloaded newtest")
else:
    print (newdata_dir, "is not writable")


# In[ ]:


get_ipython().system('echo Quick validation of rows:')
get_ipython().system("sed -n '$=' $newtrain_path")
get_ipython().system("sed -n '$=' $train_path")
get_ipython().system("sed -n '$=' $newtest_path")
get_ipython().system("sed -n '$=' $test_path")


# ## Simplifying newtrain and newtest
# * start with the obvious: if a column has no data for newtrain OR newtest, it is useless to both (we'll drop these)
# * there two columns not common to both: **totals.transactionRevenue** (we need); **trafficSource.campaignCode** (we'll just ignore this)

# In[ ]:


def get_unused(df):
    rows = df.shape[0]
    ddf = df.isna().sum()
    return list(ddf[ddf >= rows].index)
    
droplist = get_unused(newtrain)
for dropcol in get_unused(newtest):
    if dropcol not in droplist:
        droplist.append(dropcol)

print ("dropping these columns from both newtrain and newtest:")
print (droplist)
newtrain.drop(columns=droplist, inplace=True)
newtest.drop(columns=droplist, inplace=True)


# In[ ]:


summary(newtrain, info="basic")


# In[ ]:


summary(newtest, info="basic")


# ## EDA of newtrain data
# * **totals.transactionRevenue**: NaN vs >0  (imbalance is obvious without a graph)

# In[ ]:


zrows = newtrain['totals.transactionRevenue'].size
nrows = newtrain['totals.transactionRevenue'].dropna().size
print ("NaN =", zrows, "; >0 =", nrows)
#plt.bar([1,2], [zrows, nrows])
#plt.ylabel('Rows', fontsize=15)
#plt.xticks([1,2], ["NaN", ">0"], fontsize=15, rotation=0)
#plt.title("totals.transactionRevenue", fontsize=15);


# ## Pie chart functions
# * col_by_col_count: eg. (df, OS, session) gets session instances counted per OS (so total OS usage)
# * col_by_col_count: eg. (df, OS, revenue) gets revenue instances counted per OS
# * col_by_col_sum: eg. (df, OS, revenue) gets revenue total per OS
# * myautopct: pie is messy with 0.0 and 0.1 percent markings, so set a threshold
# * mypie: common options

# In[ ]:


def col_by_col_count(df, col1, col2, threshold=0):
    return df.groupby([col1]).count()[col2].apply(lambda x: (x if x>threshold else np.nan)).dropna()

def col_by_col_sum(df, col1, col2, threshold=0):
    return df.groupby([col1]).sum(numeric_only=True)[col2].apply(lambda x: (x if x>threshold else np.nan)).dropna()

def myautopct(pct):
    return ('%.2f' % pct) if pct > 2 else ''

def mypie(df, title, angle=0):
    # autopct='%1.1f%%'
    # textprops={'size': 'small'}  (Kaggle python (3.6.6) + libs didn't recognise this)
    df.plot(kind='pie', figsize=(5, 5), radius=1.2, startangle=angle, autopct=myautopct, pctdistance=0.8,
        rotatelabels=False, legend=True, explode=[0.02]*df.size);
    plt.title(title, weight='bold', size=14, x=2.0, y=-0.01);
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(2.5, 1.0), ncol=2, fontsize=10, fancybox=True, shadow=True);


# ## Pie deductions
# * OSs: 19 in total, and we've removed negligable sales on Nintendo etc.
# * OS split: Windows the clear winner, no suprises there. 10 OSs below the threshold and so not represented here.
# * OS by transaction instance: people buy more on a Mac it seems
# * OS by transaction sum: more or less the above, except people spend bigger amounts when using Chrome OS and Windows

# In[ ]:


newtrain['device.operatingSystem'].astype(str).unique()


# In[ ]:


df = col_by_col_count(newtrain, 'device.operatingSystem', 'sessionId', threshold=100)
mypie(df, 'OS prevalence', angle=100)


# In[ ]:


df = col_by_col_count(newtrain, 'device.operatingSystem', 'totals.transactionRevenue', threshold=100)
mypie(df, 'OS prevalence by revenue instances', angle=0)


# In[ ]:


df = col_by_col_sum(newtrain, 'device.operatingSystem', 'totals.transactionRevenue', threshold=100)
mypie(df, 'OS prevalence by revenue sum', angle=10)


# In[ ]:




