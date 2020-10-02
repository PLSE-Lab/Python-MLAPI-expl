#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time
import datetime 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras as keras
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import gc
import tqdm
from sklearn.metrics import roc_auc_score

# Preprocessing 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

'''
Fraud Detection for Kaggle competition
'''


# In[ ]:


# Run on whole data
data_path = '../input/ieee-fraud-detection'
train_id = pd.read_csv(f'{data_path}/train_identity.csv', index_col = None)
test_id = pd.read_csv(f'{data_path}/test_identity.csv')
train_action = pd.read_csv(f'{data_path}/train_transaction.csv', index_col = None)
test_action = pd.read_csv(f'{data_path}/test_transaction.csv')

# Now this is the version that we don't split data into 2 part

#print(train_action.info())
#print(train_id.info())

# Combine transaction & respective ID features
train = train_action.merge(train_id, how = 'left', on = 'TransactionID')
test = test_action.merge(test_id, how = 'left', on = 'TransactionID')
# Use index: TransactionID to merge two dataframe, use transaction.csv as the base. Preserve both index
del train_action, train_id, test_action, test_id

import warnings
warnings.simplefilter('ignore')

gc.collect()


# In[ ]:



features = pd.read_csv('../input/rfe-feature/rfe_feature.csv')
features = list(features.iloc[:, 1].values)

# Other than the essesntial features, there are some features such as prediction target to preserve, so do not apply these features directly
col_not_drop = ['isFraud', 'TransactionID', 'TransactionDT']
train = train[features + col_not_drop]
col_not_drop.remove('isFraud')
test = test[features + col_not_drop]
print('Useful features selected by RFECV: ', len(features))

print('Current train shape:', train.values.shape)
print('Current test shape:', test.values.shape)

gc.collect()


# In[ ]:


print(train.info())
print(train.head())

gc.collect()


# In[ ]:


# EDA: TransactionAMT decimal part as new feature
train['TransactionAmtDec'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int))*100).astype(int)
test['TransactionAmtDec'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int))*100).astype(int)

# Encode card1 features: using the frequency (value counts in whole dataset)
train['card1_count'] = train['card1'].map(pd.concat([train['card1'], test['card1']], 
                                                  ignore_index = True).value_counts(dropna = False))
test['card1_count'] = test['card1'].map(pd.concat([train['card1'], test['card1']],
                                                  ignore_index = True).value_counts(dropna = False))

# Create new features: day of week, hours in a day   to detect Fraud action
train['day_of_week'] = np.floor(train['TransactionDT']/ (3600*24)) % 7
test['day_of_week'] = np.floor(test['TransactionDT']/ (3600*24)) % 7
train['hour_of_day'] = np.floor(train['TransactionDT']/ (3600)) % 24
test['hour_of_day'] = np.floor(test['TransactionDT']/ (3600)) % 24

# Create feature: Arbitrary feature interaction ????

# Use frequency to create new features for id_01, id_33, ...etc ?? Why??

gc.collect()


# In[ ]:


# Encode category features
# For null values in obj_feature: View NaN as string, then encode in same way
for feat in train.columns:
    le = LabelEncoder()
    if train[feat].dtypes == 'object':
        le.fit(list(train[feat].astype(str).values) + list(test[feat].astype(str).values))
        train[feat] = le.transform(train[feat].astype(str).values)
        test[feat] = le.transform(test[feat].astype(str).values)

print(train.head())
        
gc.collect()


# In[ ]:


null = train.isnull().sum()
print('Null values for now: \n')
print(null[null > 0].sort_values(ascending = False))


# In[ ]:


# LightGBM could deal with missing values better ? Why?
# Why should I take time element into account? This is an classification problem. 
x = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionID', 'TransactionDT'], axis = 1)
y = train.sort_values('TransactionDT')['isFraud']
#test = test.sort_values('TransactionDT').drop(['TransactionID', 'TransactionDT'], axis = 1)

del train
print('Training data shape: ', x.values.shape)
print('Testing data shape:', test.values.shape)

gc.collect()


# In[ ]:


# Start training
params = {
    'num_leaves':490,
    'min_child_weight':0.035,
    'feature_fraction':0.379,
    'baggin_fraction':0.418,
    'min_data_in_leaf':105,
    'objective':'binary',
    'max_depth':-1, 
    'learning_rate':0.0068,
    'boosting':'gbdt',
    'bagging_seed':11,
    'metric':'auc',
    'verbosity':-1,
    'reg_alpha':0.389,
    'reg_lambda':0.648,
    'random_state':47  
}

'''
# Split by time step: check if time step influence the performance later
spliter = TimeSeriesSplit(n_splits = 5)

auc = []
feature_importance = pd.DataFrame({'feature':x.columns})

# Train the data by each fold
for fold, (train_idx, valid_idx) in enumerate(spliter.split(x)):
    start_t = time()
    print('Traning on fold: ', (fold + 1))
    
    train_data = lgb.Dataset(x.iloc[train_idx], label = y.iloc[train_idx])
    valid_data = lgb.Dataset(x.iloc[valid_idx], label = y.iloc[valid_idx])
    clf = lgb.train(params, train_data,
                    num_boost_round= 10000,
                    valid_sets = [train_data, valid_data],
                    verbose_eval = 1000,
                    early_stopping_rounds = 500
                   )
    
    feature_importance['fold_{}'.format(fold + 1)] = clf.feature_importance()
    auc.append(clf.best_score['valid_1']['auc'])
    print('Fold {} finished in {}'.format((fold + 1), 
                                          str(datetime.timedelta(seconds = time() - start_t)) ))
    
print('-'*30)
print('Training has finished')
print('Total training time: ', str(datetime.timedelta(seconds = time() - start_t)))
print('Mean auc: ', np.mean(auc))
print('-'*30)

# Use lgb.train & timeseries split first: observe performance and validation accuracy??
'''


# In[ ]:


# Check the speed of lightgbm vs. xgboost 

start = time.time()
lgbm = lgb.LGBMClassifier(**params, num_boost_round = 1300)
xgbm = XGBClassifier(**params, n_estimators = 1300)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 9)

lgbm.fit(x_train, y_train)
end = time.time()
print('light gbm time:', end - start)

x_train = 
xgbm.fit()


# In[ ]:


'''
best_iter = 1305
print('Best iteration for LGB:', best_iter)
clf2 = lgb.LGBMClassifier(**params, num_boost_round= best_iter)
clf2.fit(x_train, y_train)
score = clf2.score(x_valid, y_valid)
print('Validation accuracy of LGB:', score)
'''

# Fill NaN for other models
drop_null = x.isnull().sum().sort_values(ascending = False)
drop_null = drop_null[drop_null > len(x)*0.7]
x.drop(drop_null.index, axis = 1, inplace = True)
drop_null = x.isnull().sum().sort_values(ascending = False)
x.fillna(-999, inplace = True)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 9)

from keras.
forest = RandomForestClassifier(100)
forest.fit(x_train, y_train)
score = forest.score(x_valid, y_valid)
print('Random Forest score:', score)

logit = LogisticRegression()
logit.fit(x_train, y_train)
score = logit.score(x_valid, y_valid)
print('Logistic Regression score:', score)


# In[ ]:


from keras.applications import vgg16

vgg16 = vgg16(include_top = False)


gc.collect()


# In[ ]:


# Save the predicted result to submission.csv

submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
submission['isFraud'] = clf2.predict_proba(test)[:, 1]
submission.to_csv('submission.csv')

# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe

# create a link to download the dataframe
create_download_link(submission)

