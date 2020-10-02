#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import gc
import warnings
warnings.filterwarnings("ignore")

# data manipulation
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd

# plot
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)

# model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# In[ ]:


#Input data files are available in the "../input/" directory.
#For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields

def load_df(csv_path='../input/train.csv', JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']):

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = load_df("../input/train.csv")')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test = load_df("../input/test.csv")')


# In[ ]:


#READING SUMISSION FILE
submission=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


gc.collect()


# In[ ]:


set(train.columns).difference(set(test.columns))


# In[ ]:


miss_per = {}
for k, v in dict(train.isna().sum(axis=0)).items():
    if v == 0:
        continue
    miss_per[k] = 100 * float(v) / len(train)
    
import operator 
sorted_x = sorted(miss_per.items(), key=operator.itemgetter(1), reverse=True)
print ("There are " + str(len(miss_per)) + " columns with missing values")

kys = [_[0] for _ in sorted_x][::-1]
vls = [_[1] for _ in sorted_x][::-1]
trace1 = go.Bar(y = kys, orientation="h" , x = vls, marker=dict(color="#d6a5ff"))
layout = go.Layout(title="Missing Values Percentage", 
                   xaxis=dict(title="Missing Percentage"), 
                   height=400, margin=dict(l=300, r=300))
figure = go.Figure(data = [trace1], layout = layout)
iplot(figure)


# In[ ]:


device_cols = ["device.browser", "device.deviceCategory", "device.operatingSystem"]

colors = ["#d6a5ff", "#fca6da", "#f4d39c", "#a9fcca"]
traces = []
for i, col in enumerate(device_cols):
    t = train[col].value_counts()
    traces.append(go.Bar(marker=dict(color=colors[i]),orientation="h", y = t.index[:15][::-1], x = t.values[:15][::-1]))

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=["Visits: Category", "Visits: Browser","Visits: OS"], print_grid=False)
fig.append_trace(traces[1], 1, 1)
fig.append_trace(traces[0], 1, 2)
fig.append_trace(traces[2], 1, 3)

fig['layout'].update(height=400, showlegend=False, title="Visits by Device Attributes")
iplot(fig)

## convert transaction revenue to float
train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')

device_cols = ["device.browser", "device.deviceCategory", "device.operatingSystem"]

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=["Mean Revenue: Category", "Mean Revenue: Browser","Mean Revenue: OS"], print_grid=False)

colors = ["red", "green", "purple"]
trs = []
for i, col in enumerate(device_cols):
    tmp = train.groupby(col).agg({"totals.transactionRevenue": "mean"}).reset_index().rename(columns={"totals.transactionRevenue" : "Mean Revenue"})
    tmp = tmp.dropna().sort_values("Mean Revenue", ascending = False)
    tr = go.Bar(x = tmp["Mean Revenue"][::-1], orientation="h", marker=dict(opacity=0.5, color=colors[i]), y = tmp[col][::-1])
    trs.append(tr)

fig.append_trace(trs[1], 1, 1)
fig.append_trace(trs[0], 1, 2)
fig.append_trace(trs[2], 1, 3)
fig['layout'].update(height=400, showlegend=False, title="Mean Revenue by Device Attributes")
iplot(fig)


# In[ ]:


geo_cols = ['geoNetwork.city', 'geoNetwork.continent','geoNetwork.country',
            'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']
geo_cols = ['geoNetwork.continent','geoNetwork.subContinent']

colors = ["#d6a5ff", "#fca6da"]
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=["Visits : GeoNetwork Continent", "Visits : GeoNetwork subContinent"], print_grid=False)
trs = []
for i,col in enumerate(geo_cols):
    t = train[col].value_counts()
    tr = go.Bar(x = t.index[:20], marker=dict(color=colors[i]), y = t.values[:20])
    trs.append(tr)

fig.append_trace(trs[0], 1, 1)
fig.append_trace(trs[1], 1, 2)
fig['layout'].update(height=400, margin=dict(b=150), showlegend=False)
iplot(fig)




geo_cols = ['geoNetwork.continent','geoNetwork.subContinent']
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=["Mean Revenue: Continent", "Mean Revenue: SubContinent"], print_grid=False)

colors = ["blue", "orange"]
trs = []
for i, col in enumerate(geo_cols):
    tmp = train.groupby(col).agg({"totals.transactionRevenue": "mean"}).reset_index().rename(columns={"totals.transactionRevenue" : "Mean Revenue"})
    tmp = tmp.dropna().sort_values("Mean Revenue", ascending = False)
    tr = go.Bar(y = tmp["Mean Revenue"], orientation="v", marker=dict(opacity=0.5, color=colors[i]), x= tmp[col])
    trs.append(tr)

fig.append_trace(trs[0], 1, 1)
fig.append_trace(trs[1], 1, 2)
fig['layout'].update(height=450, margin=dict(b=200), showlegend=False)
iplot(fig)


# In[ ]:


tmp = train["geoNetwork.country"].value_counts()

# plotly globe credits - https://www.kaggle.com/arthurtok/generation-unemployed-interactive-plotly-visuals
colorscale = [[0, 'rgb(102,194,165)'], [0.005, 'rgb(102,194,165)'], 
              [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 
              [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 
              [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]

data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = colorscale,
        showscale = True,
        locations = tmp.index,
        z = tmp.values,
        locationmode = 'country names',
        text = tmp.values,
        marker = dict(
            line = dict(color = '#fff', width = 2)) )           ]

layout = dict(
    height=500,
    title = 'Visits by Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = '#222',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
iplot(fig)


tmp = train.groupby("geoNetwork.country").agg({"totals.transactionRevenue" : "mean"}).reset_index()



# plotly globe credits - https://www.kaggle.com/arthurtok/generation-unemployed-interactive-plotly-visuals
colorscale = [[0, 'rgb(102,194,165)'], [0.005, 'rgb(102,194,165)'], 
              [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 
              [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 
              [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]

data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = colorscale,
        showscale = True,
        locations = tmp['geoNetwork.country'],
        z = tmp['totals.transactionRevenue'],
        locationmode = 'country names',
        text = tmp['totals.transactionRevenue'],
        marker = dict(
            line = dict(color = '#fff', width = 2)) )           ]

layout = dict(
    height=500,
    title = 'Mean Revenue by Countries',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = '#222',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


gc.collect()


# In[ ]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


missing_values_table(train)


# In[ ]:


train[train['totals.transactionRevenue'].isnull()].describe(include = 'all')


# In[ ]:


train[train['totals.transactionRevenue'].notnull()].describe(include = 'all')


# In[ ]:


gc.collect()


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# In[ ]:


# test data doesn't have trafficSource.campaignCode columns which have 100% null record . lets drop this column
train = train.drop('trafficSource.campaignCode',1)

# convert revenue columns to float
# train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')


# In[ ]:


# columns with only one unique values
const_cols = [c for c in train.columns if train[c].nunique(dropna=False)==1 ]
print("There are " + str(len(const_cols))+ " columns in the dataset with only one entry" )
const_cols

# we can drop these columns from train and test data set


# In[ ]:


#Dropping above 19 columns from data set which have unique entry as it won't help creating the model

train = train.drop(columns= const_cols, axis= 1)
test = test.drop(columns= const_cols, axis= 1)

print("Shape of train data set",train.shape)
print("Shape of test data set",test.shape)


# In[ ]:


if test.fullVisitorId.nunique() == len(submission):
    print('Till now, the number of fullVisitorId is equal to the rows in submission. Everything goes well!')
else:
    print('Check it again')


# In[ ]:


print("Number of unique visitors in train set : ",train.fullVisitorId.nunique(), " out of rows : ",train.shape[0])
print("Number of unique visitors in test set : ",test.fullVisitorId.nunique(), " out of rows : ",test.shape[0])
print("Number of common visitors in train and test set : ",len(set(train.fullVisitorId.unique()).intersection(set(test.fullVisitorId.unique())) ))


# In[ ]:


# lets change the date format
train['date'] = pd.to_datetime(train['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
test['date'] = pd.to_datetime(test['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))


# In[ ]:


missing_values_table(train)


# In[ ]:


train_id = train.fullVisitorId
test_id = test.fullVisitorId
train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float').fillna(0)
train_y = train["totals.transactionRevenue"]
train_target = np.log1p(train.groupby("fullVisitorId")["totals.transactionRevenue"].sum())


# In[ ]:


train['trafficSource.adwordsClickInfo.isVideoAd'].replace({False: 'True'}, inplace= True)
test['trafficSource.adwordsClickInfo.isVideoAd'].replace({False: 'True'}, inplace= True)


# In[ ]:


train['trafficSource.adwordsClickInfo.isVideoAd'].fillna(value = 'False', inplace= True)
test['trafficSource.adwordsClickInfo.isVideoAd'].fillna(value = 'False', inplace= True)


# In[ ]:


for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day


# In[ ]:


gc.collect()


# In[ ]:


def browser_mapping(x):
    browsers = ['chrome','safari','firefox','internet explorer','edge','opera','coc coc','maxthon','iron']
    if x in browsers:
        return x.lower()
    elif  ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'
    
    
def adcontents_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'
    
def source_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'


# In[ ]:


train['device.browser'] = train['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
train['trafficSource.adContent'] = train['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
train['trafficSource.source'] = train['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

test['device.browser'] = test['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
test['trafficSource.adContent'] = test['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
test['trafficSource.source'] = test['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')


# In[ ]:


def process_device(data_df):
    print("process device ...")
    data_df['source.country'] = data_df['trafficSource.source'] + '_' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '_' + data_df['trafficSource.medium']
    data_df['browser.category'] = data_df['device.browser'] + '_' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + '_' + data_df['device.operatingSystem']
    return data_df

train = process_device(train)
test = process_device(test)

def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device.deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device.browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device.operatingSystem'] + "_" + data['channelGrouping']
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            data[i + "_" + j] = data[i] + "_" + data[j]
    
    data['content.source'] = data['trafficSource.adContent'] + "_" + data['source.country']
    data['medium.source'] = data['trafficSource.medium'] + "_" + data['source.country']
    return data

train = custom(train)
test = custom(test)


# In[ ]:


train.drop(['fullVisitorId', 'sessionId', 'visitId','visitStartTime'], axis = 1, inplace = True)
test.drop(['fullVisitorId', 'sessionId', 'visitId','visitStartTime'], axis = 1, inplace = True)


# In[ ]:


gc.collect()


# In[ ]:


del train_target


# In[ ]:


del train_id


# In[ ]:


num_col = ["totals.hits", "totals.pageviews", "visitNumber", 'totals.bounces',  'totals.newVisits']
for i in num_col:
    train[i] = train[i].astype('float').fillna(0)
    test[i] = test[i].astype('float').fillna(0)


# In[ ]:


train.shape, test.shape


# In[ ]:


# # total hits and pageviews is totally correlated,we will drop hits column to avoid data leakage
# train.drop("totals.hits", axis= 1 , inplace= True)
# test.drop("totals.hits", axis= 1 , inplace= True)


# In[ ]:


train['device.operatingSystem'] = train['device.operatingSystem'].replace({'Nokia':'Others','Xbox':'Others','SunOS':'Others','Samsung':'Others',
                                                                          'OpenBSD':'Others','(not set)':'Others','Nintendo WiiU':'Others','Nintendo 3DS':'Others',
                                                                          'NTT DoCoMo':'Others','FreeBSD':'Others','Firefox OS':'Others',
                                                                           'BlackBerry':'Others','Nintendo Wii':'Others'})
test['device.operatingSystem'] = test['device.operatingSystem'].replace({'Nokia':'Others','Xbox':'Others','SunOS':'Others','Samsung':'Others',
                                                                          'OpenBSD':'Others','(not set)':'Others','Nintendo WiiU':'Others','Nintendo 3DS':'Others',
                                                                          'NTT DoCoMo':'Others','FreeBSD':'Others','Firefox OS':'Others',
                                                                           'BlackBerry':'Others','Nintendo Wii':'Others','Tizen':'Others','OS/2':'Others',
                                                                         'Playstation Vita':'Others','SymbianOS':'Others'})


# In[ ]:


missing_values_table(train)


# In[ ]:


# # lets drop these columns

# train.drop(columns= ['trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.page','trafficSource.adwordsClickInfo.slot'
#                      ,'trafficSource.adwordsClickInfo.gclId','trafficSource.isTrueDirect','trafficSource.keyword'], axis =1, inplace = True)

# test.drop(columns= ['trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.page','trafficSource.adwordsClickInfo.slot'
#                      ,'trafficSource.adwordsClickInfo.gclId','trafficSource.isTrueDirect','trafficSource.keyword'], axis =1, inplace = True)


# In[ ]:


train.shape, test.shape


# In[ ]:


gc.collect()


# In[ ]:


cat_col = [e for e in train.columns.tolist() if e not in num_col]
cat_col.remove('date')
cat_col.remove('totals.transactionRevenue')


# In[ ]:


train[cat_col].nunique()


# In[ ]:


for i in cat_col:
    lab_en = LabelEncoder()
    train[i] = train[i].fillna('not known')
    test[i] = test[i].fillna('not known')
    lab_en.fit(list(train[i].astype('str')) + list(test[i].astype('str')))
    train[i] = lab_en.transform(list(train[i].astype('str')))
    test[i] = lab_en.transform(test[i].astype('str'))
    print('finish', i)


# In[ ]:


train_y = np.log1p(train["totals.transactionRevenue"])
train_x = train.drop(["totals.transactionRevenue",'date'], axis=1)
test_x = test.copy()
test_x = test_x.drop('date',axis=1)
print(train_x.shape)
print(test_x.shape)


# In[ ]:


del train
del test


# In[ ]:


gc.collect()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_x, train_y, test_size = 0.15, random_state = 2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


params={'learning_rate': 0.05,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 200,
        'verbose': 1,
        "subsample": 0.99,
        "colsample_bytree": 0.99,
        "random_state":33,
        'max_depth': 14,
        'lambda_l2': 0.02085548700474218,
        'lambda_l1': 0.004107624022751344,
        'bagging_fraction': 0.7934712636944741,
        'feature_fraction': 0.686612409641711,
        'min_child_samples': 21
       }
    
train_set = lgb.Dataset(X_train, y_train, silent=False)
valid_set = lgb.Dataset(X_test, y_test, silent=False )
model = lgb.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=100,
                   verbose_eval=200, valid_sets=valid_set)


# In[ ]:


gc.collect()


# In[ ]:


final = pd.DataFrame(test_id)

prediction = model.predict(test_x, num_iteration = model.best_iteration)
prediction[prediction< 0] = 0  
prediction = np.expm1(prediction)

final['PredictedLogRevenue']=pd.Series(prediction)

#GROUPING PREDICTED DATA ON fullVisitorId
final = final.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
final.columns = ["fullVisitorId", "PredictedLogRevenue"]

#AGAIN TAKING LOG AS SUBMISSION HAVE TO BE DONE ON LOG VALUES
final["PredictedLogRevenue"] = np.log1p(final["PredictedLogRevenue"])


# In[ ]:


#CREATING JOIN BETWEEN PREDICTED DATA WITH SUBMISSION FILE
submission=submission.join(final.set_index('fullVisitorId'),on='fullVisitorId',lsuffix='_sub')
submission.drop('PredictedLogRevenue_sub',axis=1,inplace=True)

#HANDLING NaN IN CASE OF MISSING fullVisitorId
submission.fillna(0,inplace=True)

#SUBMITING FILE
submission.to_csv('lgbm_baseline.csv',index=False)
submission.head()


# In[ ]:


lgb.plot_importance(model, height=0.5, max_num_features=20, ignore_zero = False, figsize = (12,6), importance_type ='gain')
plt.show()

