#!/usr/bin/env python
# coding: utf-8

# Here's a map of user flow for the training data. It's similar to the [Users Flow](https://support.google.com/analytics/answer/1709395) report shown in Google Analytics. I chose a few demographic features and  mapped arrival through the first few visits. The Sankey Diagram is a great tool for this purpose. It shows the connections across features in addition to the totals seen in a series of bar charts.
# 
# I used this as the basis for a 'user-level' data model, shown below.

# In[ ]:


import numpy as np
import pandas as pd
pd.options.display.max_rows = 500
pd.options.display.max_columns = 100
import holoviews as hv
hv.extension('bokeh')


# ### Extract and Prepare
# We need to deal with the JSON info to get the features of interest. Lucky for us, Pandas does the hard work and it only takes a few lines of code. The output is a nice Pandas dataframe with the unpacked data and a couple of added features.

# In[ ]:


# get base data
cols = ['fullVisitorId', 'geoNetwork', 'device', 'channelGrouping', 'visitNumber', 'totals', ]
train = pd.read_csv('../input/train.csv', usecols=cols, dtype={'fullVisitorId': str})
train = train[cols]

# unpack json
jsoncols = ['geoNetwork', 'device', 'totals']
def unpack(df):
    for jc in jsoncols:  # parse json
        flat_df = pd.DataFrame(df.pop(jc).apply(pd.io.json.loads).values.tolist())
        flat_df.columns = ['{}_{}'.format(jc, c) for c in flat_df.columns]
        df = df.join(flat_df)
    return df
train = unpack(train)

# prep
train['revcat'] = np.where(train.totals_transactionRevenue.isnull(), 0, 1) 
train.loc[train.visitNumber > 5, 'visitNumber'] = 5
train['visitNumber'] = 'Visit' + train.visitNumber.astype(str) 
train.sort_values(['fullVisitorId', 'visitNumber'], inplace=True)
train.head()


# ### Reshape and Graph
# Now we can group by various features and build an array for plotting. You'll see there are two kinds of features in the Sankey. Features toward the left are relatively static from visit to visit. That is, a user will mostly (not always) come from one continent, with one device type, through one channel, etc. Features toward the right are the 'per visit' features. Here I look at simple outcomes - did the user (1) exit, never to return, (2) make a purchase, or (3) return without purchasing.

# In[ ]:


# get mostly static features
statcols = ['geoNetwork_continent', 'device_deviceCategory', 'channelGrouping', 'visitNumber']
train1 = train[['fullVisitorId'] + statcols].copy()
train1 = train1.groupby('fullVisitorId').first()
graph = []
for n in range(len(statcols)-1):
    ngroups = train1.groupby([statcols[n], statcols[n+1]], as_index=True).size()
    ngroups = ngroups.reset_index()
    graph.append(ngroups)
flow_nd1 = np.concatenate(graph)


# get per visit features - code is a bit ugly...
train2 = train[['fullVisitorId', 'visitNumber', 'revcat']]
train2 = train2.groupby(['fullVisitorId', 'visitNumber'])['revcat'].max().reset_index()
depth = 6
evcols = ['visitNumber'] + ["event"+str(n) for n in range(1,depth)]

for n in range(1,depth):
    train2[evcols[n]] = np.where(train2.revcat.shift(-n+1) == 1, "Purchase"+str(n), "NextVisit"+str(n))
    train2.loc[(train2.fullVisitorId != train2.fullVisitorId.shift(-n)) & (train2.revcat.shift(-n+1) == 0), 
               evcols[n]] = "Exit"+str(n)
train2 = train2.groupby('fullVisitorId').first().drop('revcat', axis=1)

for n in range(1,depth-1):
    train2.loc[train2[evcols[n]] == "Exit"+str(n), evcols[n+1]] = "Exit"+str(n+1)

graph = []
for n in range(1,depth):
    ngroups = train2.groupby(evcols[(n-1):(n+1)], as_index=True).size()
    ngroups = ngroups.reset_index()
    graph.append(ngroups)
    graph
flow_nd2 = np.concatenate(graph)

# combine and graph
flow = np.vstack((flow_nd1, flow_nd2))
display(flow[0:5])
hv.Sankey(flow).options(width=800, height=500)


# By comparing various features and visit outcomes you can get a good high-level view of user behavior. 
# 

# ### Create Data Model
# We can use the Flow Diagram to create a data model. This is a baseline model with just a few added features. If you choose to use the 'date' feature, take a look at the conversion function. It uses a dictionary for lookups that's much faster than pandas parsing for the 730 dates we have here.

# In[ ]:


get_ipython().run_line_magic('reset', '-f')


# In[ ]:


#%% Start here to skip the graph and get a solution
import gc
import numpy as np
import pandas as pd

json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
nan_list = ["not available in demo dataset",
            "unknown.unknown",
            "(not provided)",
            "(not set)"] 
nan_dict = {nl:np.nan for nl in nan_list}


#%% get date features (fast)
def date_conv(df):
    # make a lookup table
    datevals = pd.date_range(start='2016-08-01', end='2018-04-30')
    datekeys = datevals.astype(str)
    datekeys = [d.replace('-', '') for d in datekeys]
    datedict = dict(zip(datekeys, datevals))
    # lookup
    df['date'] = df.date.map(datedict)
    return df


#%% unpack
def unpack(df):
    df.drop('sessionId', axis=1, inplace=True)
    for jc in json_cols:  # parse json
        flat_df = pd.DataFrame(df.pop(jc).apply(pd.io.json.loads).values.tolist())
        flat_df.columns = ['{}_{}'.format(jc, c) for c in flat_df.columns]
        df = df.join(flat_df)
    ad_df = df.pop('trafficSource_adwordsClickInfo').apply(pd.Series) # handle dict column
    ad_df.columns = ['tS_adwordsCI_{}'.format(c) for c in ad_df.columns]
    df = df.join(ad_df)
    return df


#%% process raw data
def clean(df):
    df.drop(['tS_adwordsCI_targetingCriteria', 'totals_visits', 'socialEngagementType'], 
        axis=1, inplace=True)
    df.replace(nan_dict, inplace=True) # convert disguised NaNs
    df.dropna(axis=1, how='all', inplace=True) 
    for col in df.columns:
        if 'totals' in col: # chnage to numeric
            df[col] = pd.to_numeric(df[col])
    df.totals_bounces.fillna(value=0, inplace=True)
    df.totals_newVisits.fillna(value=0, inplace=True)
    df.trafficSource_isTrueDirect.fillna(value=False, inplace=True)
    df.tS_adwordsCI_isVideoAd.fillna(value=True, inplace=True)
    return df


#%% main function
def allprep(file, numrows=None):
    df = pd.read_csv(file, dtype={'fullVisitorId': str, 'date': str}, nrows=numrows)
#     df = date_conv(df)
    df = unpack(df)
    df = clean(df)
    return df


#%% run raw data
train = allprep('../input/train.csv')
train['totals_transactionRevenue'].fillna(value=0, inplace=True)

test = allprep('../input/test.csv')
test['totals_transactionRevenue'] = -99

tt = pd.concat([train, test], ignore_index=True, sort=False)

del train
del test
gc.collect()

cols = list(tt)  # move target to front
cols.insert(0, cols.pop(cols.index('totals_transactionRevenue')))
tt = tt.reindex(columns= cols)


#reduce some memory (not much)
tt['totals_newVisits'] =  tt.totals_newVisits.astype(int)
tt['totals_bounces'] = tt.totals_bounces.astype(int)
tt['totals_pageviews'].fillna(0, inplace=True)
tt['totals_pageviews'] = tt.totals_pageviews.astype(int)

# drop for now - later maybe encode
tt.drop(['device_isMobile', 
'trafficSource_campaignCode', 'tS_adwordsCI_gclId'], axis=1, inplace=True)

#make revenue bins (manual for now)
# cutpts = [0, 25, 100, 200, 1000, 24000]
tt['totals_transactionRevenue'] = tt.totals_transactionRevenue/1e06
tt['revcat'] = 0
tt.loc[tt.totals_transactionRevenue > 0 , 'revcat'] = 1
# tt.loc[tt.totals_transactionRevenue > 100 , 'revcat'] = 2


# work datetime columns
tt['visitStartTime'] = pd.to_datetime(tt['visitStartTime'], unit='s')
tt['month'] = tt.visitStartTime.dt.month
tt['weekyear'] = tt.visitStartTime.dt.weekofyear
tt['dayyear'] = tt.visitStartTime.dt.dayofyear
tt['daymonth'] = tt.visitStartTime.dt.day
tt['weekday'] = tt.visitStartTime.dt.weekday
tt['hour'] = tt.visitStartTime.dt.hour
# tt['dayisdate'] = tt.date.dt.date == tt.visitStartTime.dt.date

# tt['localhour'] based on time zone
# fix geos


# add more features
tt['hitsperpage'] = tt.totals_hits/tt.totals_pageviews
tt['adgoogle'] = np.where(tt.trafficSource_adContent.str.contains('google', case=False, regex=False), 
    True, False)


# fix index
tt['dupevisit'] = tt[['fullVisitorId', 'visitId']].duplicated(keep='last')
tt['newvisitId'] = tt.visitStartTime
tt.set_index(['fullVisitorId', 'newvisitId'], inplace=True, verify_integrity=True)


# clean up columns
tt.drop(['visitStartTime', 'visitId', 'date'], axis=1, inplace=True)
cols = list(tt)  # move binned target to front
cols.insert(0, cols.pop(cols.index('revcat')))
tt = tt.reindex(columns= cols)


#%% get static values 
statcols = ['channelGrouping',
            'device_browser',
            'device_deviceCategory',
            'device_operatingSystem',
            'geoNetwork_city',
            'geoNetwork_continent',
            'geoNetwork_country',
            'geoNetwork_metro',
            'geoNetwork_networkDomain',
            'geoNetwork_region',
            'geoNetwork_subContinent']

dyncols = ['visitNumber',
            'totals_hits',
            'totals_pageviews',
            'trafficSource_isTrueDirect',
            'trafficSource_medium',
            'trafficSource_referralPath',
            'trafficSource_source',
            'tS_adwordsCI_isVideoAd',
            'month',
            'weekyear',
            'dayyear',
            'daymonth',
            'weekday',
            'hour',
            'hitsperpage']


def reshape(df):

    # convert booleans for fillna later
    bools = ['tS_adwordsCI_isVideoAd', 'trafficSource_isTrueDirect']
    for b in bools:
        df[b] = df[b].astype(int)


    #%% get revenue and revcat summed
    dfrevs = df[['revcat', 'totals_transactionRevenue']]
    aggdict = {'revcat': ['max'], 
                'totals_transactionRevenue': ['sum']}
    dfrevs = dfrevs.groupby(level=0).agg(aggdict)
    dfrevs.columns = pd.Index([e[0] + "_" + e[1] for e in 
        dfrevs.columns.tolist()])
    dfstats = df.groupby(level=0)[statcols].first()

    #%% get dynamic values by visit

    # time differences
    dfdyns = df[dyncols].sort_index().reset_index()

    dfdyns.loc[dfdyns.visitNumber > 12, 'visitNumber'] = 12 # are we losing something here?
    # dfdyns['hours_sincelast'] = dfdyns.newvisitId.diff().apply(lambda x: x.seconds/3600)
    dfdyns['hours_sincelast'] = (dfdyns.newvisitId - dfdyns.newvisitId.shift()).apply(lambda x: x.total_seconds())
    # dfdyns['hours_sincelast'] = pd.to_timedelta(dfdyns.hours_sincelast)

    dfdyns.loc[dfdyns.fullVisitorId != dfdyns.fullVisitorId.shift(), 'hours_sincelast'] = 0
    dfdyns['firstvis'] = dfdyns.groupby('fullVisitorId')['newvisitId'].transform('first')
    dfdyns['days_sincefirst'] = (dfdyns.newvisitId - dfdyns.firstvis).apply(lambda x: x.seconds/3600/24)
    dfdyns.drop(['newvisitId', 'firstvis'], axis=1, inplace=True)

    # reshape into months as columns
    dfdyns = dfdyns.groupby(['fullVisitorId', 'visitNumber']).first().unstack() # loses datetimecol
    dfdyns.columns = pd.Index([e[0] + "_" + str(e[1]) for e in dfdyns.columns.tolist()]) 

    #%% join
    df2 = dfrevs.join(dfstats, sort=False)
    df2 = df2.join(dfdyns, sort=False)
    return df2


#%% from features
# tt = pd.read_parquet('./input/tt1a.parq')
train = tt[tt.totals_transactionRevenue >= 0]
test = tt[tt.totals_transactionRevenue < 0]
print(train.shape, test.shape)

train = reshape(train)
test = reshape(test)

tt3 = pd.concat([train, test], sort=False)

del tt
del train
del test
gc.collect()

# fill nas
for c in tt3.columns:
    if tt3[c].dtype == 'object':
        tt3[c].fillna('none', inplace=True)
    else:  
        tt3[c].fillna(-1, inplace=True)

# tt3.to_parquet('./input/tt1a.parq')



# ### LightGBM
# The data structure is probably better suited to other methods. 'll stick with LightGBM here to provide a more direct comparison to other public models.

# In[ ]:


#%%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb


#%% encode string categories
le = LabelEncoder()
catcols = tt3.select_dtypes(['bool', 'object']).columns
for c in catcols:
    tt3[c] = le.fit_transform(tt3[c])


# scale as needed


#%%
def splitdata(df):
    train = df[df.totals_transactionRevenue_sum >= 0]
    X = train.drop(['totals_transactionRevenue_sum', 'revcat_max'], axis=1)
    y = np.log1p(train.totals_transactionRevenue_sum * 1e06)
    y_strat = train.month_1 # month-based folds
    test = df[df.totals_transactionRevenue_sum < 0]
    X_test = test.drop(['totals_transactionRevenue_sum', 'revcat_max'], axis=1)
    return X, y, y_strat, X_test

X, y, y_strat, X_test = splitdata(tt3)

del tt3
gc.collect()

#%% lightgbm

params = {'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.015, 
        'num_leaves': 55, 
        # 'min_data_inleaf': 1000,  
        'feature_fraction': 0.9, 
        'lambda_l1': 10,  
        'lambda_l2': 10,
        'n_jobs': -1}
        # 'bagging_fraction': 0.95,
        # 'bagging_freq': 20, 
        # 'predict_contrib': True, # for shap values
        # 'max_depth': 5}  #7       # setting this causes hang-ups
         # 'min_child_weight': 35,
        # 'min_split_gain': 0.01,       

oof_preds = np.zeros_like(y, dtype=float)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=44)  
for i, [trn_idx, val_idx] in enumerate(cv.split(X, y_strat)):  # stratify by month of first visit
    print("starting fold {}".format(i+1))
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    trainDS = lgb.Dataset(X_train, label=y_train.values)
    valDS = lgb.Dataset(X_val, label=y_val.values, reference=trainDS)

    evalnums = {}
    lmod = lgb.train(params, trainDS, num_boost_round=2000, early_stopping_rounds=40,
        valid_sets=[trainDS, valDS], evals_result=evalnums, verbose_eval=20)

    oof_preds[val_idx] = lmod.predict(X_val)

mean_squared_error(y, oof_preds)**0.5


# In[ ]:


score = mean_squared_error(y, np.clip(oof_preds, 0, None))**0.5
print('RMSE {}'.format(score))


# In[ ]:





#%%
trainDS = lgb.Dataset(X, label=y.values)

test_preds=np.zeros(X_test.shape[0], dtype=float)
for i in range(3):
    
    lmodfull = lgb.train(params, trainDS, num_boost_round=1300, 
        valid_sets=[trainDS], evals_result=evalnums, verbose_eval=50)
    test_preds += lmodfull.predict(X_test)/3



#%% submit
sub = pd.read_csv('../input/sample_submission.csv', index_col='fullVisitorId')
sub.head()
sub['PredictedLogRevenue'] = np.clip(test_preds, 0, None)
sub.to_csv('subtimewiselgb002.csv')


# So for only a few features and no tuning this concept seems to be useful. The script shown here has a  CV of  1.580  and should give a Public LB in the mid 1.477's. Good luck!
