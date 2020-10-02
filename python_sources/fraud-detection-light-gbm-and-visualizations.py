#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold
import lightgbm as lgb
import itertools
from sklearn.metrics import roc_auc_score
import gc
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings("ignore")
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial


# # MEMORY REDUCATION

# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# # Dataset Loading and Merging

# In[ ]:


train_trains = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col = 'TransactionID')
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col = 'TransactionID')
test_trains = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col = 'TransactionID')
test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col = 'TransactionID')


# In[ ]:


train_trains = reduce_mem_usage(train_trains)
train_id = reduce_mem_usage(train_id)
test_trains = reduce_mem_usage(test_trains)
test_id = reduce_mem_usage(test_id)


# In[ ]:


train = pd.merge(train_trains, train_id, on ='TransactionID', how = 'left')
test = pd.merge(test_trains, test_id, on = 'TransactionID', how = 'left')
train = train.reset_index()
test = test.reset_index()
print(train.shape)
print(test.shape)


# In[ ]:


del train_id, train_trains, test_id, test_trains
gc.collect()


# In[ ]:


print('This might be an imbalanced class problem from what we can see')
train['isFraud'].value_counts(normalize = True)*100


# > THIS IS A BEAUTIFUL CLASS IMBALANCE PROBLEM

# > MISSING VALUE COUNT

# In[ ]:


print('THERE ARE {0} MISSING VALUES IN TRAIN'.format(train.isnull().any().sum()))


# In[ ]:


train['isFraud'].value_counts(normalize = True, dropna = False).values


# > Some Overbiased Columns that might be removed later

# In[ ]:


for cols in train.columns[2:]:
    if train[cols].value_counts(normalize = True, dropna = False).values[0]> 0.9:
        print(train[cols].value_counts(normalize = True, dropna = False)*100)
        print('-'*90)
        print('-'*90)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # VISUALIZATIONS

# In[ ]:


card_fraud = train[['isFraud', 'card4']].groupby(by = 'card4').count()
card_fraud = card_fraud.reset_index()
plt.figure(figsize = (10,6))
axes = plt.bar(x = card_fraud['card4'], height = card_fraud['isFraud'], color = ['red', 'yellow', 'green', 'blue'], edgecolor = 'Black')
plt.tick_params(labelsize = 13)
plt.xlabel('CARD TYPE', fontdict = {'fontsize':15})
plt.ylabel('NO OF FRAUDS', fontdict = {'fontsize':15})
for ax in axes.patches:
    plt.text(ax.get_x() + 0.27, ax.get_height() + 5000, str(round(ax.get_height(), 2)), fontdict= {'fontsize':13})
plt.show()


# In[ ]:


card_fraud = train[['isFraud', 'card6']].groupby(by = 'card6').count()
card_fraud = card_fraud.reset_index()
plt.figure(figsize = (10,6))
axes = plt.bar(x = card_fraud['card6'], height = card_fraud['isFraud'], color = ['red', 'yellow', 'green', 'blue'], edgecolor = 'Black')
plt.tick_params(labelsize = 13)
plt.xlabel('CARD TYPE', fontdict = {'fontsize':15})
plt.ylabel('NO OF FRAUDS', fontdict = {'fontsize':15})
for ax in axes.patches:
    plt.text(ax.get_x() + 0.27, ax.get_height() + 5000, str(round(ax.get_height(), 2)), fontdict= {'fontsize':13})
plt.show()


# > FROM THIS WE CAN SEE THAT IT MIGHT BE EASIER TO SCAM THE DEBIT CARD USERS

# In[ ]:


plt.figure(figsize = (10,4))
axes = plt.barh(train['DeviceType'].unique()[1:], train['DeviceType'].value_counts().values, color = ['orange', 'green'], 
               edgecolor = 'black')
plt.tick_params(labelsize = 12)
for ax in axes.patches:
    plt.text(ax.get_width()- 10500, ax.get_y() + 0.34, str(round(ax.get_width(), 2)), fontdict= {'fontsize':13})
plt.title('Total Fraud Occurance on different platforms', fontdict = {'fontsize':16})
plt.show()


# In[ ]:


id31_fruad = train['id_31'][train['isFraud'] == 0].value_counts()
id31 = train['id_31'][train['isFraud'] == 1].value_counts()
fig, axes = plt.subplots(2,1, figsize = (10,8), sharex = False, sharey = False)
axes[0].barh(id31.keys()[:10], width = id31.values[:10])
axes[1].barh(id31_fruad.keys()[:10], width = id31_fruad.values[:10])
axes[0].tick_params(labelsize = 12)
axes[1].tick_params(labelsize = 12)
axes[0].set_title('Browsers from where most frad cases happened', fontdict = {'fontsize':15})
axes[1].set_title('Browsers from where least fraud cases happened',fontdict = {'fontsize':15})
plt.show()


# In[ ]:


dev_fruad = train['DeviceInfo'][train['isFraud'] == 0].value_counts()
dev = train['DeviceInfo'][train['isFraud'] == 1].value_counts()
fig, axes = plt.subplots(2,1, figsize = (10,8), sharex = False, sharey = False)
axes[0].barh(dev.keys()[:10], width = id31.values[:10])
axes[1].barh(dev_fruad.keys()[:10], width = id31_fruad.values[:10])
axes[0].tick_params(labelsize = 12)
axes[1].tick_params(labelsize = 12)
axes[0].set_title('Handsets from where most fraud cases happened', fontdict = {'fontsize':15})
axes[1].set_title('Handsets from where least fraud cases happened',fontdict = {'fontsize':15})
plt.show()


# In[ ]:


dev_fruad = train['P_emaildomain'][train['isFraud'] == 0].value_counts()
dev = train['P_emaildomain'][train['isFraud'] == 1].value_counts()
fig, axes = plt.subplots(2,1, figsize = (10,8), sharex = False, sharey = False)
axes[0].barh(dev.keys()[:10], width = id31.values[:10])
axes[1].barh(dev_fruad.keys()[:10], width = id31_fruad.values[:10])
axes[0].tick_params(labelsize = 12)
axes[1].tick_params(labelsize = 12)
axes[0].set_title('email platforms from where most fraud cases happened', fontdict = {'fontsize':15})
axes[1].set_title('email platforms from where least fraud cases happened',fontdict = {'fontsize':15})
plt.show()


# In[ ]:


fig, axes = plt.subplots(1,2, figsize = (15,6))
frd_amt = train[['TransactionAmt', 'card4']][train['isFraud'] == 1].groupby(by = 'card4').mean().reset_index()
frd_amt1 = train[['TransactionAmt', 'card4']][train['isFraud'] == 0].groupby(by = 'card4').mean().reset_index()
a = axes[0].bar(x = frd_amt['card4'], height = frd_amt['TransactionAmt'],  color = ['red', 'yellow', 'green', 'blue'], edgecolor = 'Black')
axes[0].tick_params(labelsize = 13)
axes[0].set_xlabel('Card Type', fontdict = {'fontsize':15})
axes[0].set_ylabel('Average Fraud Amount', fontdict = {'fontsize':15})
for ax in a.patches:
    axes[0].text(ax.get_x() + 0.27, ax.get_height() + 3, str(round(ax.get_height(), 2)), fontdict= {'fontsize':13})
    
b = axes[1].bar(x = frd_amt1['card4'], height = frd_amt1['TransactionAmt'],  color = ['red', 'yellow', 'green', 'blue'], edgecolor = 'Black')
axes[1].tick_params(labelsize = 13)
axes[1].set_xlabel('Card Type', fontdict = {'fontsize':15})
axes[1].set_ylabel('Average Non-Fraud Amount', fontdict = {'fontsize':15})
for ax in b.patches:
    axes[1].text(ax.get_x() + 0.27, ax.get_height() + 3, str(round(ax.get_height(), 2)), fontdict= {'fontsize':13})
plt.show()


# > Is Resolution related to Fraud

# In[ ]:


res_fraud = train[['isFraud', 'id_33']].groupby(by = 'id_33').sum().reset_index()
res_fraud = res_fraud.sort_values(by = 'isFraud', ascending = False)
res_fraud = res_fraud[:15][:]
plt.figure(figsize = (15,8))
plt.plot(res_fraud['id_33'], res_fraud['isFraud'],'*', color = 'red', markersize = 15)
plt.plot(res_fraud['id_33'], res_fraud['isFraud'], color = 'black') 
plt.xticks(rotation = 45)
plt.xlabel('Resolutions', fontdict = {'fontsize':14})
plt.ylabel('Total Fraud occured', fontdict = {'fontsize':14})
plt.title('Total Fraud occured VS Resolutions', fontdict = {'fontsize':18})
plt.show()


# In[ ]:


fig, axes = plt.subplots(6,2, figsize = (13,25), sharex = False, sharey = False)
sns.distplot(train['D1'].dropna().astype(int), ax = axes[0,0])
sns.distplot(train['D2'].dropna().astype(int), ax = axes[0,1])
sns.distplot(train['D3'].dropna().astype(int), ax = axes[1,0])
sns.distplot(train['D4'].dropna().astype(int), ax = axes[1,1])
sns.distplot(train['D5'].dropna().astype(int), ax = axes[2,0])
sns.distplot(train['D6'].dropna().astype(int), ax = axes[2,1])
sns.distplot(train['D7'].dropna().astype(int), ax = axes[3,0])
sns.distplot(train['D8'].dropna().astype(int), ax = axes[3,1])
sns.distplot(train['D9'].dropna().astype(int), ax = axes[4,0])
sns.distplot(train['D10'].dropna().astype(int), ax = axes[4,1])
sns.distplot(train['D11'].dropna().astype(int), ax = axes[5,0])
sns.distplot(train['D12'].dropna().astype(int), ax = axes[5,1])
plt.show()


# In[ ]:


def label_collector(string):
    label = string.split('.')[0]
    return label

temp = train['P_emaildomain'].astype(str)
train['label_encode'] = temp.apply(label_collector)


# In[ ]:


card_cost = train[['label_encode', 'TransactionAmt','isFraud']][train['isFraud']==1].groupby(by = 'label_encode').mean().reset_index()
card_cost = card_cost.sort_values(by = 'TransactionAmt', ascending = False)
plt.figure(figsize = (14,7))
plt.xticks(rotation = 45)
plt.xlabel('E-mail domain', fontdict = {'fontsize':13})
plt.ylabel('Average Fraud Amount', fontdict = {'fontsize':13})
plt.tick_params(labelsize = 12)
axes = plt.bar(x = card_cost['label_encode'].iloc[0:10], height = card_cost['TransactionAmt'].iloc[0:10], color = ['red','green', 'blue', 'yellow', 'pink', 'black', 'orange','purple', 'brown', 'white'], edgecolor = 'black')
for ax in axes.patches:
    plt.text(ax.get_x() + 0.2, ax.get_height() + 3, str(round(ax.get_height(), 2)), fontdict= {'fontsize':13})
plt.title('E-Mail domain vs Fraud Amount', fontdict = {'fontsize':15})
plt.show()


# In[ ]:


card_cost = train[['label_encode', 'TransactionAmt','isFraud']][train['isFraud']==0].groupby(by = 'label_encode').mean().reset_index()
card_cost = card_cost.sort_values(by = 'TransactionAmt', ascending = False)
plt.figure(figsize = (14,7))
plt.xticks(rotation = 45)
plt.xlabel('E-mail domain', fontdict = {'fontsize':13})
plt.ylabel('Average Non-Fraud Amount', fontdict = {'fontsize':13})
plt.tick_params(labelsize = 12)
axes = plt.bar(x = card_cost['label_encode'].iloc[0:10], height = card_cost['TransactionAmt'].iloc[0:10], color = ['red','green', 'blue', 'yellow', 'pink', 'black', 'orange','purple', 'brown', 'white'], edgecolor = 'black')
for ax in axes.patches:
    plt.text(ax.get_x() + 0.2, ax.get_height() + 3, str(round(ax.get_height(), 2)), fontdict= {'fontsize':13})
plt.title('E-Mail domain vs Fraud Amount', fontdict = {'fontsize':15})
plt.show()


# In[ ]:


train = train.drop('label_encode', axis = 1)


# In[ ]:


cd_fault = train[['ProductCD', 'isFraud']][train['isFraud']==1].groupby(by = 'ProductCD').sum().reset_index()
plt.figure(figsize = (10,6))
axes = plt.bar(x = cd_fault['ProductCD'], height = cd_fault['isFraud'],  color = ['red', 'yellow', 'green', 'blue', 'pink'], edgecolor = 'Black')
plt.tick_params(labelsize = 13)
plt.xlabel('ProductCD', fontdict = {'fontsize':15})
plt.ylabel('Total Fraudulent cases', fontdict = {'fontsize':15})
for ax in axes.patches:
    plt.text(ax.get_x() + 0.2, ax.get_height() + 100, str(round(ax.get_height(), 2)), fontdict= {'fontsize':13})
plt.show()


# > 1. FROM THE ABOVE PLOTS WE CAN SEE THE  MOST USED PLATFORMS AND THEIR FRAUD EXPECTANCY
# > 2. WE CAN SEE THAT MOST AND LEAST FRAUD CAUSING PLATFORMS ARE SAME THIS IS JUST BECAUSE OF THEIR HUGE POPULARITY

# # Handling Missing Values And Encoding Necessary Columns

# In[ ]:


cols_drop_train = [cols for cols in train.columns if train[cols].isnull().sum()/ train.shape[0] > 0.9]
cols_drop_test = [cols for cols in test.columns if test[cols].isnull().sum()/ test.shape[0]> 0.9]
big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
drop_cols = list(set(cols_drop_train + cols_drop_test + big_top_value_cols + big_top_value_cols_test))
drop_cols.remove('isFraud')


# In[ ]:


train.drop(drop_cols, axis = 1, inplace = True)
test.drop(drop_cols, axis = 1, inplace = True)


# In[ ]:


del cols_drop_test, cols_drop_train, big_top_value_cols, big_top_value_cols_test, drop_cols


# In[ ]:


train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)
test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)
test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)


# In[ ]:


print([cols for cols in train.columns if train[cols].dtype == 'O'])


# In[ ]:


def labelencode(train,test):
    for col in train.drop(['TransactionID','isFraud','TransactionDT'],axis = 1).columns:
        if train[col].dtype == 'O' or test[col].dtype == 'O':
            le = LabelEncoder()
            le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
            train[col] = le.transform(list(train[col].astype(str).values))
            test[col] = le.transform(list(test[col].astype(str).values))
    return train,test


# In[ ]:


train, test = labelencode(train, test)


# In[ ]:


y_test = train['isFraud']


# In[ ]:


cols_drops = ['TransactionID','isFraud','TransactionDT']
train = train.drop(cols_drops, axis = 1)


# In[ ]:


train.columns


# In[ ]:


test = test.drop(['TransactionID','TransactionDT'], axis = 1)


# In[ ]:


train = train.fillna(-999)
test = test.fillna(-999)


# # Modelling the Dataset

# In[ ]:


train_m, val_m_train, val1, val2 = train_test_split(train,y_test, test_size = 0.3, random_state = 10, stratify = y_test)
train_m_index = train_m.index
val_m_index = val_m_train.index
val1_index = val1.index
val2_index = val2.index


# In[ ]:


val_m_train.shape


# # Bayesian Optimization for Hyperparameter Optimization

# In[ ]:


def objective(num_leaves,min_child_weight,feature_fraction,bagging_fraction,
              max_depth,learning_rate,reg_alpha,reg_lambda,min_data_in_leaf):
    global train_m
    global val_m
    global y_test
    global train_m_index
    global val_m_index
    global val1,val2, val1_index, val2_index
    num_leaves = int(num_leaves)
    max_depth = int(max_depth)
    min_data_in_leaf = int(min_data_in_leaf)
    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    params = {'num_leaves': num_leaves,
          'min_child_weight': min_child_weight,
          'feature_fraction': feature_fraction,
          'bagging_fraction': bagging_fraction,
          'min_data_in_leaf': min_data_in_leaf,
          'objective': 'binary',
          'max_depth': max_depth,
          'learning_rate': learning_rate,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': reg_alpha,
          'reg_lambda': reg_lambda,
          'random_state':42,
         }
    oof = np.zeros(len(train_m))
    early_stopping_rounds = 50
    xgtrain = lgb.Dataset(train_m, label=val1[val1_index])
    xgvalid = lgb.Dataset(val_m_train, label=val2[val2_index])
    num_boost_round = 200
    model_lgb = lgb.train(params, xgtrain , valid_sets = [xgtrain, xgvalid], num_boost_round = num_boost_round,
                            early_stopping_rounds = early_stopping_rounds, verbose_eval = 0)
    score  = roc_auc_score(val2, model_lgb.predict(val_m_train))
    return score


# In[ ]:


bound_lgb = {'num_leaves': (70,600),
              'min_child_weight': (0.001, 0.07),
              'feature_fraction': (0.1,0.9),
              'bagging_fraction': (0.1,0.9),
              'max_depth': (-1,50),
              'learning_rate': (0.2,0.9),
              'reg_alpha': (0.3,0.9),
              'reg_lambda': (0.3,0.9),
              'min_data_in_leaf':(50,300)
         }


# In[ ]:


LGB_BO = BayesianOptimization(objective, bound_lgb, random_state=42)


# In[ ]:


LGB_BO.space.keys


# In[ ]:


init_points = 10
n_iter = 15
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter,acq='ucb', xi=0.0, alpha=1e-5)


# In[ ]:


LGB_BO.max['target']


# In[ ]:


#LGB_BO.max['params']


# In[ ]:


'''params = {'num_leaves': int(LGB_BO.max['params']['num_leaves']),
          'min_child_weight': LGB_BO.max['params']['min_child_weight'],
          'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
          'feature_fraction':LGB_BO.max['params']['feature_fraction'],
          'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': LGB_BO.max['params']['learning_rate'],
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha':LGB_BO.max['params']['reg_alpha'],
          'reg_lambda': LGB_BO.max['params']['reg_lambda'],
          'random_state':42
         }'''


# In[ ]:


params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
         }


# # Lightgbm Implementation

# In[ ]:


d_train = lgb.Dataset(train, label=y_test)


# In[ ]:


clf = lgb.train(params, d_train, verbose_eval=False, num_boost_round = 1000)


# In[ ]:


'''v_results = lgb.cv(params, d_train, nfold = 5, num_boost_round = 1000, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50, verbose_eval=100)'''


# In[ ]:


predict = clf.predict(test)


# # Feature Importance

# In[ ]:


def feature_important(model, X , num = 50):
    feature_import = pd.DataFrame(sorted(zip(model.feature_importance(), X.columns)), columns = ['values', 'columns'])
    plt.figure(figsize = (12,15))
    sns.barplot(x = 'values', y = 'columns', data = feature_import.sort_values(by = 'values', ascending = False)[:num])
    plt.show()
    
feature_important(clf, train)


# In[ ]:


submitss = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')


# In[ ]:


submission = pd.DataFrame({
        "TransactionID": submitss['TransactionID'],
        "isFraud": predict
    })

submission.TransactionID = submission.TransactionID.astype(int)

submission.to_csv("submit.csv", index=False)

