#!/usr/bin/env python
# coding: utf-8

# # This notebook can be a starting point in WNS data challange
# 
# *** This kernel reaches top 70 ranks in the WNS competition with total 6456 participants.
# 
# * This kernel finds the features correlation between various features
# * It identifies relevant features for fitting
# * Performs **lightgbm** and **xgboost** for fitting
#     * Early stopping, best iteration is:
#     
#         [100]	valid_0's auc: 0.685143
#     * the AUC scores on validation and trianing sets for xgboost are:-
# 
#         [218]	train-auc:0.780506	eval-auc:0.705122
# 
# ### references:-
# https://towardsdatascience.com/mobile-ads-click-through-rate-ctr-prediction-44fdac40c6ff
# 
# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
# 
# https://datahack.analyticsvidhya.com/contest/wns-analytics-wizard-2019/

# ### Imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore')
import seaborn as sns
import statsmodels.api as sm
import lightgbm as lgb

### Feature selection modules from sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# ### Reading all data 

# In[ ]:


def unique_num(data, key):
    return len(set(data[key]))

## day of the week and time of day
def add_feature( data, time_key):
    data['day'] = [int(x.split(' ')[0].split('-')[2])%7 for x in  data[time_key]]
    data['hour'] = [float(x.split(' ')[1].split(':')[0]) for x in data[time_key]]
    return data


# In[ ]:



train_data = pd.read_csv('../input/wns2019contest/train.csv')
item_data = pd.read_csv('../input/wns2019contest/item_data.csv')
view_log = pd.read_csv('../input/wns2019contest/view_log.csv')

test_data = pd.read_csv('../input/wns2019contest/test.csv')
#sample_sub = pd.read_csv('./input/wns2019contest/sample_submission.csv')


# extracting basic information from data

# In[ ]:


display(train_data.head(2))
n_click_train = len(train_data[train_data['is_click']==1])
n_noclick_train = len(train_data[train_data['is_click']==0])
print ('Number of unique users :- %s'%(len(np.unique(train_data['user_id']))))
print ('Number of clicks :- %s'%(n_click_train))
print ('Number of no click :- %s'%(n_noclick_train))
print ('----> Percentage of clicks to noclick in train data :- %f'%(n_click_train*100/n_noclick_train))
print ('---------------------------------')
print ('---------------------------------')

chars = [ch for ch in train_data['impression_id'][0] ]
print ('Length of impression id characters %s is same as length of data'%(len (chars)))
print (unique_num(train_data, 'impression_id'))
print ('----> each impression id encodes entire information of user_id, impression time, os version, is_4G and clicking data')

print ('---------------------------------')
print ('---------------------------------')
print ('impression times range from 15 nov to 13 dec which means ')
time_key = ['impression_time', 'server_time']
train_data = add_feature(train_data, time_key[0])
test_data = add_feature(test_data, time_key[0])
view_log = add_feature(view_log, time_key[1])


#print (unique_num(train_data, 'impression_time_int'))
train_data['os_version_num'] = np.zeros(len(train_data))
train_data['os_version_num'][train_data['os_version'] == 'intermediate'] = int(2)
train_data['os_version_num'][train_data['os_version'] == 'old'] = int(1)
train_data['os_version_num'][train_data['os_version'] == 'latest'] = int(3)

display(train_data.head(2))


# So, only 4.8% of all impressions result in a click

# In[ ]:


## identifying the features for modeling
df = pd.DataFrame(train_data)
x_y = df.drop('impression_id',  1)
x_y = x_y.drop ('impression_time', 1)

print (x_y.columns)
#x_y['']
plt.show()
display(x_y.head(2))
plt.figure(figsize=(12,10))
cor = x_y.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.viridis)
plt.show()


# In[ ]:


#Correlation with output variable
cor_target = abs(cor["is_click"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.001]
print (relevant_features)
print ('=============')
print (x_y[['os_version_num', 'is_4G']].corr())

print ('os version and is_4G are correlated makes sense because normally latest version are more likely to have 4G connection')


print ('==========================')
print (x_y[['os_version_num', 'app_code']].corr() )
print (x_y[['app_code', 'hour']].corr() )
print (x_y[['os_version_num', 'hour']].corr() )


print ('None of these features are correlated with each other,\nTherefore I believe the relevant features for traning should be :- ', 'app_code', 'hour', 'os_version_int' )
print ('we can get rid of 2 features user_id, and is_4G.... may be more testin required before gettin rid of these')


# In[ ]:


### AN attempt to understand user profiles using view_log
import warnings
warnings.filterwarnings('ignore')

display(view_log.head(2))



print ('=======================')
unique_devices = np.unique(view_log['device_type'])
for i in range(3):
    
    print (len(view_log[view_log['device_type'] == unique_devices[i]]), '%s users'%(unique_devices[i]))
print ('device type does not matter')
print ('==========')


# In[ ]:



df_view = pd.DataFrame(view_log)
df_view = df_view.drop('server_time',1)
plt.figure(figsize=(12,10))
cor = df_view.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.viridis)
plt.show()


# In[ ]:


display(item_data.head(2))

df_item = pd.DataFrame(item_data)

plt.figure(figsize=(12,10))
cor = df_item.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.viridis)
plt.show()


# In[ ]:


train_data['hour'] = train_data.impression_time.apply(lambda x: x.split(" ")[1].split(':')[0])
train_data.groupby('hour').agg({'is_click':'sum'}).plot(figsize=(12,6))
plt.ylabel('# clicks')
plt.title('Trends of clicks by hour of day');


# ## lightgbm

# In[ ]:


display(train_data.head())
train = train_data
X_train = train.loc[:, ['user_id', 'is_4G', 'day',  'app_code', 'os_version_num']]# train.columns != ['is_click', 'impression_time']]
y_target = train.is_click.values
#create lightgbm dataset
msk = np.random.rand(len(X_train)) < 0.8
lgb_train = lgb.Dataset(X_train[msk], y_target[msk])
lgb_eval = lgb.Dataset(X_train[~msk], y_target[~msk], reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'auc'},
    'num_leaves': 31, # defauly leaves(31) amount for each tree
    'learning_rate': 0.08,
    'feature_fraction': 0.7, # will select 70% features before training each tree
    'bagging_fraction': 0.3, #feature_fraction, but this will random select part of data
    'bagging_freq': 5, #  perform bagging at every 5 iteration
    'verbose': 1
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=4000,
                valid_sets=lgb_eval,
                early_stopping_rounds=1500)


# In[ ]:


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 'axes.linewidth' :3}
pylab.rcParams.update(params)


# In[ ]:


#print (unique_num(train_data, 'impression_time_int'))
test_data['os_version_num'] = np.zeros(len(test_data))
test_data['os_version_num'][test_data['os_version'] == 'intermediate'] = int(2)
test_data['os_version_num'][test_data['os_version'] == 'old'] = int(1)
test_data['os_version_num'][test_data['os_version'] == 'latest'] = int(3)

predictions_lightgbm = gbm.predict(test_data.drop(['impression_time', 'os_version', 'impression_id'], axis =1))
plt.hist(predictions_lightgbm, label = 'lightgbm')

df_kag = pd.read_csv('./DeepFM_submission_copy.csv')
plt.hist(df_kag['is_click'], alpha = 0.7,color = 'g', label ='kaggle')
plt.legend()
plt.show()


# # xgboost 

# In[ ]:


from operator import itemgetter
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def run_default_test(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 6
    subsample = 0.8
    colsample_bytree = 0.8
    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 0,
        "seed": random_state
    }
    num_boost_round = 260
    early_stopping_rounds = 20
    test_size = 0.2

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    
    features = [ 'user_id', 'app_code', 'is_4G', 'day', 'os_version_num']
    return gbm
features = [ 'user_id', 'app_code', 'is_4G', 'day', 'os_version_num']
y_target = train_data['is_click']
xgb_out = run_default_test(train, y_target, features, 'is_click')


# In[ ]:


predictions_xgboost = xgb_out.predict(xgb.DMatrix(test_data.drop(['impression_time', 'os_version', 'impression_id', 'hour'], axis =1)))#, ntree_limit=clf.best_ntree_limit)/folds.n_splits


# In[ ]:


plt.hist(predictions_lightgbm, label = 'lightgbm')
plt.hist(predictions_xgboost, label = 'xgboost', color ='r', alpha = 0.5)


df_kag = pd.read_csv('../input/deepfm-test/DeepFM_submission_copy.csv')
plt.hist(df_kag['is_click'], alpha = 0.7,color = 'g', label ='kaggle')
plt.legend()
plt.show()


# In[ ]:


### predictions_lightgbm and predictions_xgboost are the output from the normal feature enginnering.

