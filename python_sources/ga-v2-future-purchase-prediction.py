#!/usr/bin/env python
# coding: utf-8

# # This kernel is to predict the future of customer will come back purchase or not
# * Fot train_v2 data, we have 2016/08/01 ~ 2018/04/30 period data
# * For test_v2 data, we have 2018/05/1 ~ 2018/10/15 period data
# * The Public LB  score is base on timeframe 2018/05/1~ 2018/10/15
# * The Private LB score is base on timeframe of 2018/12/1 ~ 2019/01/31 with same visitor ID that in test_v2
# * So this competition become the future prediction question .....

# ## Discussion topic about this idea from AmirH
# https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/71427
# * I use LGBM to predict the user will come back purchase or not (Classification)
# 

# ## Training Set
# * Training period set 1==> 2016/08/01 ~ 2017/1/15 (5.5 month)
# * Target period set 1  ==> 2017/03/1 ~ 2017/04/30 (2 month)
# * Training period set 2==> 2017/06/01 ~ 2017/11/15 (5.5 month)
# * Target period set 2  ==> 2018/1/1 ~ 2018/02/30 (2 month)
# * Concate set 1 and set 2 to be training data
# * Feature engineering on training period feature
# * Target set that those come back purchased user in target period

# ## Valid Set (1 year ago of our test set and target )
# * Valid period set ==> 2017/5/1 ~ 2017/10/15
# * Valid target period set ==> 2017/12/1 ~ 2018/1/31

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print(os.listdir("../input"))


# In[ ]:


import seaborn as sns
import json
import pandas.io.json as pdjson
import ast

from pandas.io.json import json_normalize
def load_df(csv_path='../input/train_v2.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    for column in JSON_COLUMNS:
        column_as_df = pdjson.json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# ## Load Data
# * use Aguiar's dataset (Many thanks): https://www.kaggle.com/jsaguiar/parse-json-v2-without-hits-column

# In[ ]:


get_ipython().run_cell_magic('time', '', 'path = "../input/parse-json-v2-without-hits-column/"\ntrain_df = pd.read_pickle(path + \'train_v2_clean.pkl\')\ntest_df = pd.read_pickle(path + \'test_v2_clean.pkl\')')


# ## Add time feature

# In[ ]:


for df in [train_df,test_df]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df["day"] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['weekofyear'] = df['date'].dt.weekofyear


# In[ ]:


train_df.shape, test_df.shape


# ## Feature engineering 
# * mean, max, min for "totals_pagevies" and "totals_hits "
# * Change to lable encoding for categorical feature
# * Drop 'trafficSource_referralPath','trafficSource_source'

# In[ ]:


train_df['totals_pageviews']=train_df['totals_pageviews'].astype('float')
train_df['totals_hits']=train_df['totals_hits'].astype('float')
test_df['totals_pageviews']=test_df['totals_pageviews'].astype('float')
test_df['totals_hits']=test_df['totals_hits'].astype('float')


# In[ ]:


train_df['totals_pageviews_mean']=train_df.groupby(['fullVisitorId'])['totals_pageviews'].transform('mean')
train_df['totals_pageviews_max']=train_df.groupby(['fullVisitorId'])['totals_pageviews'].transform('max')
train_df['totals_pageviews_min']=train_df.groupby(['fullVisitorId'])['totals_pageviews'].transform('min')
train_df['totals_hits_mean']=train_df.groupby(['fullVisitorId'])['totals_hits'].transform('mean')
train_df['totals_hits_max']=train_df.groupby(['fullVisitorId'])['totals_hits'].transform('max')
train_df['totals_hits_min']=train_df.groupby(['fullVisitorId'])['totals_hits'].transform('min')
test_df['totals_pageviews_mean']=test_df.groupby(['fullVisitorId'])['totals_pageviews'].transform('mean')
test_df['totals_pageviews_max']=test_df.groupby(['fullVisitorId'])['totals_pageviews'].transform('max')
test_df['totals_pageviews_min']=test_df.groupby(['fullVisitorId'])['totals_pageviews'].transform('min')
test_df['totals_hits_mean']=test_df.groupby(['fullVisitorId'])['totals_hits'].transform('mean')
test_df['totals_hits_max']=test_df.groupby(['fullVisitorId'])['totals_hits'].transform('max')
test_df['totals_hits_min']=test_df.groupby(['fullVisitorId'])['totals_hits'].transform('min')


# In[ ]:


"""
def process_totals(data_df):
    print("process totals ...")
    #data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
    #data_df['totals_hits'] = np.log1p(data_df['totals_hits'])
    #data_df['totals_pageviews'] = np.log1p(data_df['totals_pageviews'].fillna(0))
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
    data_df['mean_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('mean')
    data_df['sum_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('sum')
    data_df['max_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('max')
    data_df['min_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('min')    
    return data_df
train_df = process_totals(train_df)
test_df = process_totals(test_df)
"""


# In[ ]:


train_df.drop(['trafficSource_referralPath', 'trafficSource_source'], axis=1, inplace=True)
test_df.drop(['trafficSource_referralPath', 'trafficSource_source'], axis=1, inplace=True)


# In[ ]:


excluded_features = [
    'date','fullVisitorId', 'sessionId','classfication_target','totals_totalTransactionRevenue','totals_transactionRevenue',
    'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','next_session_1','next_session_2'
]
categorical_features = [
    _f for _f in train_df.columns
    if (_f not in excluded_features) & (train_df[_f].dtype == 'object')
]
one_hot_features = ['day','month','weekday']
#'totals.totalTransactionRevenue','totals.TransactionRevenue','classfication_target'


# ## Process one hot encoding on time 

# In[ ]:



for i in one_hot_features:
    print("Process feature =====>"+str(i))
    train_df["one_hot_feature"] = train_df[i]
    train_df["one_hot_feature"] =  str(i) + "." + train_df["one_hot_feature"].astype('str')
    one_hot_combine = pd.get_dummies(train_df["one_hot_feature"])
    print(one_hot_combine.shape)
    train_df = train_df.join(one_hot_combine)
    del train_df["one_hot_feature"]
    del train_df[i]
    del one_hot_combine
    print(train_df.shape)


# ### Factoriza  categorical featuers

# In[ ]:



for f in categorical_features:
    train_df[f], indexer = pd.factorize(train_df[f])
    test_df[f] = indexer.get_indexer(test_df[f])


# In[ ]:


train_df.shape


# In[ ]:


gc.collect()


# ## Split Validate and Train data by timeframe

# ## Training Set
# * Training period set 1==> 2016/08/01 ~ 2017/1/15 (5.5 month)
# * Target period set 1  ==> 2017/03/1 ~ 2017/04/30 (2 month)
# * Training period set 2==> 2017/06/01 ~ 2017/11/15 (5.5 month)
# * Target period set 2  ==> 2018/1/1 ~ 2018/02/30 (2 month)

# In[ ]:


train_df['date'].max(),train_df['date'].min()


# In[ ]:


test_df['date'].max(),test_df['date'].min()


# ## Training period

# In[ ]:


train_period_1 = train_df[(train_df['date']<=pd.datetime(2017,1,15)) & (train_df['date']>=pd.datetime(2016,8,1))]
train_predict_preiod_1 = train_df[(train_df['date']<=pd.datetime(2017,4,30)) & (train_df['date']>=pd.datetime(2017,3,1))]
train_period_2 = train_df[(train_df['date']<=pd.datetime(2017,11,15)) & (train_df['date']>=pd.datetime(2017,6,1))]
train_predict_preiod_2 = train_df[(train_df['date']<=pd.datetime(2018,2,28)) & (train_df['date']>=pd.datetime(2018,1,1))]


# ## Valid period

# In[ ]:


valid_period = train_df[(train_df['date']<=pd.datetime(2017,10,15)) & (train_df['date']>=pd.datetime(2017,5,1))]
valid_predict_preiod = train_df[(train_df['date']<=pd.datetime(2018,1,31)) & (train_df['date']>=pd.datetime(2017,12,1))]


# In[ ]:


print('train_period1_shape',train_period_1.shape) 
print('train_target1_period_shape',train_predict_preiod_1.shape)
print('train_period2_shape',train_period_2.shape) 
print('train_target2_period_shape',train_predict_preiod_2.shape)
print('valid_period_shape',valid_period.shape) 
print('valid_target_period_shape',valid_predict_preiod.shape)


# ## Add the target on training data and validation data

# In[ ]:


def add_target(train_period,target_period):
    
    train_period['totals_totalTransactionRevenue'] = train_period['totals_totalTransactionRevenue'].fillna(0).astype('float64')
    target_period['totals_totalTransactionRevenue'] =target_period['totals_totalTransactionRevenue'].fillna(0).astype('float64')
    train_period['totals_transactionRevenue'] = train_period['totals_transactionRevenue'].fillna(0).astype('float64')
    target_period['totals_transactionRevenue'] = target_period['totals_transactionRevenue'].fillna(0).astype('float64')
    #train_period['totals_transactions'] = train_period['totals_transactions'].fillna(0).astype('float64')
    #target_period['totals_transactions'] = target_period['totals_transactions'].fillna(0).astype('float64')
    
    #train_pd=train_period
    train_pd = train_period.groupby('fullVisitorId').mean().reset_index()
    target_pd = target_period.groupby('fullVisitorId').mean().reset_index()
    #target_pd=target_period
    #Find the visitors those back puchased in future period
    train_visitors = train_pd.fullVisitorId.unique()
    train_predict_visitors = target_pd.fullVisitorId.unique()
    same_visitors = np.intersect1d(train_visitors, train_predict_visitors)
    
    #Process data type
    
    
    #Process back user df
    back_user = target_pd[(target_pd.fullVisitorId.isin(same_visitors)) & (target_pd['totals_transactionRevenue'] > 0)]
    back_user = back_user[['fullVisitorId','totals_transactionRevenue']]
    print('we have',len(back_user['fullVisitorId'].value_counts()),'visitors back to purchase at target periods')
    
    #Add target
    train_pd['classfication_target'] = train_pd['fullVisitorId'].map(lambda x: 1 if x in list(back_user['fullVisitorId']) else 0)
    train_pd['totals_totalTransactionRevenue'] = np.log1p(train_pd['totals_totalTransactionRevenue'])
    train_pd['totals_transactionRevenue'] = np.log1p(train_pd['totals_transactionRevenue'])
    print (train_pd.shape)
    return train_pd


# In[ ]:


train_pd_1=add_target(train_period_1,train_predict_preiod_1)
train_pd_2=add_target(train_period_2,train_predict_preiod_2)
valid_pd = add_target(valid_period,valid_predict_preiod)


# In[ ]:


train_set = pd.concat([train_pd_1,train_pd_2], axis=0)


# In[ ]:


train_set.shape


# In[ ]:


excluded_features = [
    'date','fullVisitorId', 'sessionId','classfication_target',
    'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','next_session_1','next_session_2'
]
train_features = [_f for _f in train_set.columns if _f not in excluded_features ]


# ## Set K fold

# In[ ]:


from sklearn.model_selection import GroupKFold
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids


# In[ ]:


y_target = train_set['classfication_target']
valid_target = valid_pd['classfication_target']


# ## Start training (5 fold LightGBM)

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
params = {
    "max_bin": 512,
    "learning_rate": 0.02,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "min_data": 100,
    "boost_from_average": True
}
n_fold = 5
#print(train_features)
folds = get_folds(df=train_set, n_splits=5)

model = lgb.LGBMClassifier(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

oof_reg_preds = np.zeros(train_set.shape[0])
prediction = np.zeros(valid_pd.shape[0])

for fold_n, (trn_, val_) in enumerate(folds):
    print('Fold:', fold_n)
    #print(f'Train samples: {len(train_index)}. Valid samples: {len(test_index)}')
    trn_x, trn_y = train_set[train_features].iloc[trn_], y_target.iloc[trn_]
    val_x, val_y = train_set[train_features].iloc[val_], y_target.iloc[val_]
    

    model.fit(trn_x, trn_y, 
            eval_set=[(trn_x, trn_y), (val_x, val_y)], eval_metric='AUC',
            verbose=500, early_stopping_rounds=100)
    
    oof_reg_preds[val_] = model.predict(val_x, num_iteration=model.best_iteration_)
    
    pred = model.predict(valid_pd[train_features], num_iteration=model.best_iteration_)
    prediction += pred
    
prediction /= n_fold
#print(accuracy_score(y_target,np.float64(oof_reg_preds>=0.5)))
#print(accuracy_score(valid_target,np.float64(prediction>=0.5)))


# ## Plot feature important

# In[ ]:


lgb.plot_importance(model, figsize=(15, 10))
plt.show()


# In[ ]:


prediction_ans = np.where(prediction >= 0.2, 1, 0)
#valid_ans = np.where(prediction>=0.5,1,0)


# In[ ]:


plt.figure(figsize=(16,6))
false_positive_rate, recall, thresholds = roc_curve(y_target, oof_reg_preds)
roc_auc = auc(false_positive_rate, recall)
plt.subplot(121)
plt.title('Receiver Operating Characteristic (ROC)_train')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')

false_positive_rate, recall, thresholds = roc_curve(valid_target, prediction_ans)
roc_auc = auc(false_positive_rate, recall)
plt.subplot(122)
plt.title('Receiver Operating Characteristic (ROC)_Valid')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()


# ## Plot confusion matrix of prediction

# In[ ]:


import seaborn as sns
#Print Confusion Matrix
plt.figure(figsize=(16,6))
cm1 = confusion_matrix(y_target, oof_reg_preds)
labels = ['0', '1']
plt.subplot(121)
sns.heatmap(cm1, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix_train')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')

cm2 = confusion_matrix(valid_target, prediction_ans)
labels = ['0', '1']
plt.subplot(122)
sns.heatmap(cm2, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix_valid')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# ## Conclusion 
# * We only can see there are only 5 true positives labels....   
# * Try find the key feature, and do another feature enginnering for the future predict 
# * Did anyone have better idea and improve AUC for classification?
# 
# ## Next Step
# * Doing regression for future revenue prediction...
# 

# In[ ]:




