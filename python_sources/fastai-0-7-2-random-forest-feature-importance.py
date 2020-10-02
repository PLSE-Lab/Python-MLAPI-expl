#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# fastai 0.7.2
get_ipython().system('pip install git+https://github.com/fastai/fastai@e85667cfae2e6873b1bb026195b5d09a74dfcff9')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics


# In[ ]:


PATH = "../input/"

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

import os
import json
from pandas.io.json import json_normalize

def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    types = {
        "fullVisitorId": "str", # readme says it should be str
        "channelGrouping": "str",
        "date": "str",
        "socialEngagementType": "str",
        "visitId": "int32",
        "visitNumber": "int8",
        "visitStartTime": "int32",
    }
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype=types,
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[ ]:


def split_vals(a,n): 
    return a[:n].copy(), a[n:].copy()

def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


# df_raw = pd.read_csv(f'{PATH}train_v2.csv', low_memory=False, nrows=500_000, dtype={'fullVisitorId': 'str'})


# In[ ]:


# display_all(df_raw.head().T)


# In[ ]:


# display_all(df_raw.describe(include="all"))


# In[ ]:


df_raw = load_df(f'{PATH}train_v2.csv', nrows=500_000)


# Data Fields
# 
#     fullVisitorId - A unique identifier for each user of the Google Merchandise Store.
#     channelGrouping - The channel via which the user came to the Store.
#     date - The date on which the user visited the Store.
#     device - The specifications for the device used to access the Store.
#     geoNetwork - This section contains information about the geography of the user.
#     socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
#     totals - This section contains aggregate values across the session.
#     trafficSource - This section contains information about the Traffic Source from which the session originated.
#     visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
#     visitNumber - The session number for this user. If this is the first session, then this is set to 1.
#     visitStartTime - The timestamp (expressed as POSIX time).
#     hits - This row and nested fields are populated for any and all types of hits. Provides a record of all page visits.
#     customDimensions - This section contains any user-level or session-level custom dimensions that are set for a session. This is a repeated field and has an entry for each dimension that is set.
#     totals - This set of columns mostly includes high-level aggregate data.
# 

# In[ ]:


display_all(df_raw.head().T)


# In[ ]:


display_all(df_raw.describe(include="all"))


# In[ ]:


const_cols = [c for c in df_raw.columns if df_raw[c].nunique(dropna=False)==1]
df_raw.drop(columns=const_cols + ["customDimensions", "hits"], inplace=True)


# In[ ]:


df_test = load_df(f'{PATH}test_v2.csv')


# In[ ]:


display_all(df_test.head().T)


# In[ ]:


list((set(df_raw.columns).difference(set(df_test.columns))))


# In[ ]:


df_raw.drop(columns=['trafficSource.campaignCode'], inplace=True)


# In[ ]:


df_test.drop(columns=const_cols + ["customDimensions", "hits"], inplace=True)


# We are predicting the natural log of the sum of all transactions per user. Once the data is updated, as noted above, this will be for all users in test_v2.csv for December 1st, 2018 to January 31st, 2019.

# In[ ]:


df_raw['totals.transactionRevenue'].fillna(0, inplace=True)
df_raw['totals.transactionRevenue'] = df_raw['totals.transactionRevenue'].astype(float)
df_raw['totals.transactionRevenue'] = np.log1p(df_raw['totals.transactionRevenue'])


# In[ ]:


df_raw["visitStartTime"] = pd.to_datetime(df_raw["visitStartTime"], infer_datetime_format=True, unit="s")
df_raw["date"] = pd.to_datetime(df_raw["date"], infer_datetime_format=True, format="%Y%m%d")
add_datepart(df_raw, 'date')
add_datepart(df_raw, 'visitStartTime')


# In[ ]:


df_raw['totals.totalTransactionRevenue'].fillna(0, inplace=True)
df_raw['totals.totalTransactionRevenue'] = df_raw['totals.totalTransactionRevenue'].astype('int32')
df_raw['totals.transactions'].fillna(0, inplace=True)
df_raw['totals.transactions'] = df_raw['totals.transactions'].astype('int8')


# In[ ]:


df_test['totals.transactionRevenue'].fillna(0, inplace=True)
df_test['totals.transactionRevenue'] = df_test['totals.transactionRevenue'].astype(float)
df_test['totals.transactionRevenue'] = np.log1p(df_test['totals.transactionRevenue'])


# In[ ]:


df_test["visitStartTime"] = pd.to_datetime(df_test["visitStartTime"], infer_datetime_format=True, unit="s")
df_test["date"] = pd.to_datetime(df_test["date"], infer_datetime_format=True, format="%Y%m%d")
add_datepart(df_test, 'date')
add_datepart(df_test, 'visitStartTime')


# In[ ]:


display_all(df_raw.head().T)


# In[ ]:


train_cats(df_raw)
train_cats(df_test)


# In[ ]:


os.makedirs('tmp', exist_ok=True)
# df_raw.to_feather('tmp/ga-raw')
# df_test.to_feather('tmp/ga-test')


# In[ ]:


# df_raw = pd.read_feather('tmp/ga-raw')
# df_test = pd.read_feather('tmp/ga-test')


# In[ ]:


# display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[ ]:


df_trn, y_trn, nas = proc_df(df_raw, 'totals.transactionRevenue')


# In[ ]:


display_all(df_trn.head().T)


# ## See how much can we overfit - prediction without validation set

# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df_trn, y_trn)
m.score(df_trn, y_trn)


# ## Make it stupid, simple

# In[ ]:


train_required_ratio = 0.8
n_trn = int(len(df_trn) * train_required_ratio)

X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
X_train.shape, X_valid.shape


# In[ ]:


m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# ## Bagging

# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]
preds.shape


# In[ ]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);


# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# No effect?

# In[ ]:


set_rf_samples(50_000)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# # Predict on the test data

# In[ ]:


df_trn, y_trn, nas = proc_df(df_raw, 'totals.transactionRevenue')
train_required_ratio = 0.8
n_trn = int(len(df_trn) * train_required_ratio)

X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)

X_test, y_test, _ = proc_df(df_test, 'totals.transactionRevenue', nas)
X_train.shape, X_valid.shape, X_test.shape


# In[ ]:


m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)


# In[ ]:


# rmse(m.predict(X_train), y_train), rmse(m.predict(X_test), y_test), m.score(X_train, y_train), m.score(X_test, y_test), m.oob_score_
print_score(m)


# In[ ]:


m.feature_importances_


# In[ ]:


X_train.columns[22]


# In[ ]:


predictions = m.predict(X_test)


# # Confidence based on tree variance

# In[ ]:


set_rf_samples(50000)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


raw_train, raw_valid = split_vals(df_raw, n_trn)


# In[ ]:


x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x['totals.transactions'].value_counts().plot.barh();


# In[ ]:


flds = ['totals.transactions', 'totals.transactionRevenue', 'pred', 'pred_std']
tr_summ = x[flds].groupby('totals.transactions', as_index=False).mean()
tr_summ = tr_summ[~pd.isnull(tr_summ['totals.transactionRevenue'])]
tr_summ.plot('totals.transactions', 'totals.transactionRevenue', 'barh', xlim=(0,25));


# In[ ]:


tr_summ.plot('totals.transactions', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0,25));


# # Feature Importance

# In[ ]:


fi = rf_feat_importance(m, df_trn); fi[:30]


# In[ ]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


plot_fi(fi[:30]);


# In[ ]:


fi[fi.imp>0.0001]


# In[ ]:


to_keep = fi[fi.imp>0.0001].cols; len(to_keep)


# In[ ]:


df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi);


# ## One-hot encoding

# In[ ]:


df_trn2, y_trn, nas = proc_df(df_raw, 'totals.transactionRevenue', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25]);


# Not very useful

# # Removing redundant features

# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


df_keep = df_trn[to_keep].copy()


# In[ ]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# In[ ]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# Baseline

# In[ ]:


get_oob(df_keep)


# In[ ]:


for c in ('totals.transactions', 'totals.totalTransactionRevenue'):
    print(c, get_oob(df_keep.drop(c, axis=1)))


# In[ ]:


to_drop = ['totals.transactions']
get_oob(df_keep.drop(to_drop, axis=1))


# In[ ]:


df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


np.save('tmp/keep_cols.npy', np.array(df_keep.columns))


# In[ ]:


keep_cols = np.load('tmp/keep_cols.npy')
df_keep = df_trn[keep_cols]


# In[ ]:


reset_rf_samples()


# And let's see how this model looks on the full dataset.

# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# A bit better

# # Tree interpreter
# 

# In[ ]:


get_ipython().system('pip install treeinterpreter')


# In[ ]:


from treeinterpreter import treeinterpreter as ti


# In[ ]:


df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)


# In[ ]:


row = X_valid.values[None,0]; row


# In[ ]:


prediction, bias, contributions = ti.predict(m, row)


# In[ ]:


prediction[0], bias[0]


# In[ ]:


idxs = np.argsort(contributions[0])


# In[ ]:


# [o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]


# # Extrapolation

# In[ ]:


df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0
x, y, nas = proc_df(df_ext, 'is_valid')


# In[ ]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# In[ ]:


fi = rf_feat_importance(m, x); fi[:10]


# In[ ]:


feats=['totals.timeOnSite']


# In[ ]:


(X_train[feats]/1000).describe()


# In[ ]:


(X_valid[feats]/1000).describe()


# In[ ]:


x.drop(feats, axis=1, inplace=True)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# In[ ]:


fi = rf_feat_importance(m, x); fi[:10]


# In[ ]:


set_rf_samples(50000)


# In[ ]:


feats=['totals.timeOnSite', 'geoNetwork.country', 'totals.hits', 'totals.pageviews']


# In[ ]:


X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


for f in feats:
    df_subs = df_keep.drop(f, axis=1)
    X_train, X_valid = split_vals(df_subs, n_trn)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(f)
    print_score(m)


# In[ ]:


reset_rf_samples()


# In[ ]:


df_subs = df_keep.drop(['geoNetwork.country'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# not much better

# In[ ]:


plot_fi(rf_feat_importance(m, X_train));


# In[ ]:


np.save('tmp/subs_cols.npy', np.array(df_subs.columns))


# # Our final model!

# In[ ]:


m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


X_test = X_test.drop(set(df_raw.columns).difference(set(df_subs.columns)) - {'totals.transactionRevenue'}, axis=1);


# In[ ]:


predictions = m.predict(X_test)


# # Prepare the submission

# In[ ]:


df_actual_test = df_test.copy()


# In[ ]:


actual_predicted_revenue = predictions
df_actual_test["predicted"] = actual_predicted_revenue

df_actual_test = df_actual_test[["fullVisitorId" , "predicted"]]
df_actual_test["fullVisitorId"] = df_actual_test.fullVisitorId.astype('str')
df_actual_test["predicted"] = df_actual_test.predicted.astype(np.float)
df_actual_test.index = df_actual_test.fullVisitorId
df_actual_test = df_actual_test.drop("fullVisitorId",axis=1)


# In[ ]:


df_actual_test["predicted"].value_counts().iloc[:5]


# In[ ]:


df_submission_test = pd.read_csv(filepath_or_buffer="../input/sample_submission_v2.csv",index_col="fullVisitorId")
df_submission_test.shape


# In[ ]:


"test shape is :{} and submission shape is : {}".format(df_actual_test.shape , df_submission_test.shape)
final_df = df_actual_test.loc[df_submission_test.index,:]


# In[ ]:


final_df = final_df[~final_df.index.duplicated(keep='first')]
final_df = final_df.rename(index=str, columns={"predicted": "PredictedLogRevenue"})


# In[ ]:


final_df.PredictedLogRevenue.value_counts(bins=3)


# In[ ]:


final_df = final_df.fillna(0); final_df.iloc[262153]


# In[ ]:


final_df.to_csv("sub.csv")


# In[ ]:




