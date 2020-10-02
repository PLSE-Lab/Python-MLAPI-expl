#!/usr/bin/env python
# coding: utf-8

# I am trying to analyze data thru diff views to get a clue which features may be impacting target. My intenetion is to keep things simple and easily comprehendable. I myself get lost sometimes in good kernels which are bit low on structure part.  I have tried to keep it structured and scalable for new features and models. 

# -  [Import and Read](#LibLink)
# -  [Basic EDA](#EDALink)
# -  [Functions](#FuncLink)
# -  [Plotting](#PlotLink)
# -  [Corr and Bin](#CorLink)
# -  [Features](#FeatLink)
# -  [Model](#ModLink)

# <div id="LibLink">
# **Import libraries**
# </div>

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import lightgbm as lgb
from sklearn import metrics
import gc
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import itertools
from sklearn import metrics
from scipy.stats import norm, rankdata
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', 200)
# below is to have multiple outputs from same Jupyter cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')
from feature_selector import FeatureSelector


# Read files 

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df = train_df.sample(n = 20000, random_state = 42)
test_df = test_df.sample(n = 20000, random_state = 42)


# <div id="EDALink">
#  **Basic EDA**
#  </div>

# Number of rows and columns in Dataset

# In[ ]:


print("Train Shape\n")
print("*"*80)
train_df.shape
print("*"*80)
print("Test Shape\n")
print("*"*80)
test_df.shape


# Basic statistics for datasets

# In[ ]:


print("Train Describe\n")
train_df.describe() 
print("Test Describe\n")
test_df.describe()


# Distribution of target in training dataset. This shows its a imbalance data set, with 90% of data being 0 and 10% as 1.

# In[ ]:


train_df["target"].value_counts()/train_df.shape[0]*100
fig,ax= plt.subplots()
sns.countplot(data=train_df,x="target",ax=ax)
ax.set(xlabel="Target",
       ylabel="Count", 
       Title = "Target Distribution"
       )


# Looking at output of describe for both df, data seems to be similar in both the datasets (test and train).  Another point is test is of same size as train. we need to find a way to extract some info from test data.

# ### Missing values

# None of dataset has any missing values. 

# In[ ]:


train_df.isnull().sum().sum()
test_df.isnull().sum().sum()


# <div id="FuncLink">
# ** Utility Functions for EDA and Feature Engineering **
#     </div>

# To reduce memory footprint

# In[ ]:


# reduce memory
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()/2.0
            c_max = df[col].max()/2.0
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

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# To plot distributions features of two datasets  **plot_feature_distribution**

# In[ ]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# To plot boxplot of features of two datasets, along with class split  **plot_feature_boxplot**

# In[ ]:


def plot_feature_boxplot(df1,df2,label1,label2,features,target):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(100,2,figsize=(10,180))

    for feature in features:
        i += 1
        plt.subplot(100,2,i)
        sns.boxplot(y=df1[feature], x=target, showfliers=False)
        plt.title(feature+'_train', fontsize=10)
        plt.ylabel('')
        plt.xlabel('')
        i += 1
        plt.subplot(100,2,i)
        sns.boxplot(df2[feature],orient='v',color='r')
        plt.title(feature+'_test', fontsize=10)
        plt.ylabel('')
        plt.xlabel('')

        #locs, labels = plt.xticks()
        #plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        #plt.tick_params(axis='y', which='major', labelsize=6)
        #plt.gca().axes.get_xaxis().set_visible(False)
        #plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()  
    plt.show();
    
    


# To plot violinplot of features of two datasets, along with class split  **plot_feature_violinplot**

# In[ ]:


def plot_feature_violinplot(df1,df2,label1,label2,features,target):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(100,2,figsize=(10,180))

    for feature in features:
        i += 1
        plt.subplot(100,2,i)
        sns.violinplot(y=df1[feature], x=target, showfliers=False)
        plt.title(feature+'_train', fontsize=10)
        plt.ylabel('')
        plt.xlabel('')
        i += 1
        plt.subplot(100,2,i)
        sns.violinplot(df2[feature],orient='v',color='r')
        plt.title(feature+'_test', fontsize=10)
        plt.ylabel('')
        plt.xlabel('')

        #locs, labels = plt.xticks()
        #plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        #plt.tick_params(axis='y', which='major', labelsize=6)
        #plt.gca().axes.get_xaxis().set_visible(False)
        #plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()  
    plt.show();
    


# To plot violinplot of binned features for training along with class split  **plot_binned_feature_target_violinplot**

# In[ ]:


def plot_binned_feature_target_violinplot(df,features,target):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(200,1,figsize=(8,380))

    for feature in features:
        bins = np.nanpercentile(df[feature], range(0,101,10))
        df[feature+"_binned"] = pd.cut(df[feature],bins=bins)
        i += 1
        plt.subplot(200,1,i)
        sns.violinplot(y=df[feature+"_binned"], x=target, showfliers=False)
        plt.title(feature+'_binned & Target', fontsize=12)
        plt.ylabel('')
        plt.xlabel('')
       
        locs, labels = plt.xticks()
        plt.xticks([0.0,1.0])
        plt.tick_params(axis='y', which='major', labelsize=8)
        #ax.set_xticks([0.15, 0.68, 0.97])
        #plt.gca().axes.get_xaxis().set_visible(False)
        #plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()  
    plt.show();


# To add new features row wise  **add_new_feature_row**

# In[ ]:


def add_new_feature_row(df,features):
    for feature in features:
        df[feature+"_pct"] = df[feature].pct_change()
        df[feature+"_diff"] = df[feature].diff()
        df.drop(feature,axis=1)
    return df
    


# To normailize features using combined dataset **add_feature_df**

# In[ ]:


def add_feature_df(df,features):
    # count +ve and -ve
    df['count+'] = np.array(df>0).sum(axis=1)
    df['count-'] = np.array(df<0).sum(axis=1)
    #sum
    #df['sum_outside'] = df.sum(axis=1)
        
    for feature in features:
        #normalize
        #df[feature+'_norm'] = (df[feature] - df[feature].mean())/df[feature].std()
        #percentage change row wise
        #df[feature+"_pct"] = df[feature].pct_change() # didnt give boost
        #diff change row wise
        #df[feature+"_diff"] = df[feature].diff() # didnt give boost
        # Square
        #df[feature+'^2'] = df[feature] * df[feature]
        # Cube
        #df[feature+'^3'] = df[feature] * df[feature] * df[feature]
        # 4th power
        #df[feature+'^4'] = df[feature] * df[feature] * df[feature] * df[feature]
        # Cumulative percentile (not normalized)
        #df[feature+'_cp'] = rankdata(df[feature]).astype('float32')
        # Cumulative normal percentile, probabilites
        #df[feature+'_cnp'] = norm.cdf(df[feature]).astype('float32')
        # sqrt
        #df[feature+'_sqrt'] = np.sqrt(df[feature])
        #binning
        #bins = np.nanpercentile(df[feature], range(0,101,10))
        #df[feature+"_binned"] = pd.cut(df[feature],bins=bins)
        #rounding
        #df[feature+'_r2'] = np.round(df[feature], 2)
        #rounding
        #df['r1_'+feature] = np.round(df[feature], 1)
        #exp
        #df['exp_'+feature] = np.exp(df[feature])
        #exp and feature
        #df['xintoexp_'+feature] = np.exp(df[feature])*df[feature]
        #sum
        #df['sum_inside'] = df[[feature]].sum(axis=1)
        #max
        #df['max'] = df[[feature]].max(axis=1)
        #min
        #df['min'] = df[[feature]].min(axis=1)
        #max
        #df['std'] = df[[feature]].std(axis=1)
        #skew
        #df['skew'] = df[[feature]].skew(axis=1)
        #kurt
        #df['kurt'] = df[[feature]].kurtosis(axis=1)
        #median
        #df['med'] = df[[feature]].median(axis=1)
        #tanh
        df['tanh_'+feature] = np.tanh(df[feature])
        
    return df
    


# In[ ]:


import gc
gc.collect()


# <div id = FeatLink>
# ** Feature Engineering **
#     </div>

# In[ ]:


test_df['target']= np.nan
combine_df = train_df.append(test_df,ignore_index=True)


# In[ ]:


#features = train_df.columns[~train_df.columns.isin(['target','ID_code'])]
#combine_df = add_feature_df(combine_df,features)


# In[ ]:


#combine_df.head()


# Separate out train and test. Append new features created to training dataset.

# In[ ]:


train_df = combine_df[combine_df['target'].notnull()].reset_index(drop=True)
test_df = combine_df[combine_df['target'].isnull()].reset_index(drop=True)


# In[ ]:


# features will have added cols defined as part of feature engineering
features = train_df.columns[~train_df.columns.isin(['target','ID_code'])]


# In[ ]:


# Features are in train and labels are in train_labels
fs = FeatureSelector(data = train_df[features], labels = train_df["target"])


# In[ ]:


fs.identify_all(selection_params = {'missing_threshold': 0.6,    
                                    'correlation_threshold': 0.95, 
                                    'task': 'classification',    
                                    'eval_metric': 'auc', 
                                    'cumulative_importance': 0.95})


# In[ ]:


collinear_features = fs.ops['collinear']
fs.record_collinear.head(10)


# In[ ]:


fs.feature_importances.head(200)


# In[ ]:


# plot the feature importances
fs.plot_feature_importances(threshold = 0.99, plot_n = 25)


# In[ ]:


# list of zero importance features
zero_importance_features = fs.ops['zero_importance']
zero_importance_features


# In[ ]:


train_removed_all = fs.remove(methods = 'all', 
                                          keep_one_hot=False)


# In[ ]:


fs.ops # stats for which all are removed by what method.


# In[ ]:


train_removed_all.head()


# In[ ]:


feat_sel=train_removed_all.columns.values


# In[ ]:


feat_sel


# In[ ]:


with open ("feat_sel.csv","w")as fp:
   for line in feat_sel:
       fp.write(str(line)+"\n")


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
test_df['target']= np.nan
combine_df = train_df.append(test_df,ignore_index=True)
features = train_df.columns[~train_df.columns.isin(['target','ID_code'])]
combine_df = add_feature_df(combine_df,features)
train_df = combine_df[combine_df['target'].notnull()].reset_index(drop=True)
test_df = combine_df[combine_df['target'].isnull()].reset_index(drop=True)


# <div id = ModLink>
# ** Modeling **
#     </div>

# This is model lifted and shifted from [Fayaz's](https://www.kaggle.com/fayzur/lightgbm-customer-transaction-prediction) kernel.

# In[ ]:


#test_df = test_df.drop("target",axis=1)
predictors = train_removed_all.columns
nfold = 5
target = 'target'


# In[ ]:


'''param = {
     'num_leaves': 18,
     'max_bin': 63,
     'min_data_in_leaf': 5,
     'learning_rate': 0.010614430970330217,
     'min_sum_hessian_in_leaf': 0.0093586657313989123,
     'feature_fraction': 0.056701788569420042,
     'lambda_l1': 0.060222413158420585,
     'lambda_l2': 4.6580550589317573,
     'min_gain_to_split': 0.29588543202055562,
     'max_depth': 49,
     'save_binary': True,
     'seed': 1337,
     'feature_fraction_seed': 1337,
     'bagging_seed': 1337,
     'drop_seed': 1337,
     'data_random_seed': 1337,
     'objective': 'binary',
     'boosting_type': 'gbdt',
     'verbose': 1,
     'metric': 'auc',
     'is_unbalance': True,
     'boost_from_average': False
}
'''
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}

nfold = 10

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)

oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

i = 1
for train_index, valid_index in skf.split(train_df, train_df.target.values):
    print("\nfold {}".format(i))
    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,
                           label=train_df.iloc[train_index][target].values,
                           feature_name='auto',
                           categorical_feature = 'auto',
                           free_raw_data = False
                           )
    print("after lgb train")
    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,
                           label=train_df.iloc[valid_index][target].values,
                           feature_name='auto',
                           categorical_feature = 'auto',
                           free_raw_data = False
                           )   
    print("after lgb test")
    nround = 1000000
    clf = lgb.train(param, 
                    xg_train, 
                    nround, 
                    valid_sets = [xg_train,xg_valid], 
                    early_stopping_rounds=3000,
                    verbose_eval=1000)
    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=nround) 
    print("after lgb fit")
    predictions += clf.predict(test_df[predictors], num_iteration=nround) / nfold
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = predictors
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] =  i = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
   


print("\n\nCV AUC: {:<0.4f}".format(metrics.roc_auc_score(train_df.target.values, oof)))


# In[ ]:





# Feature Importance as per model

# In[ ]:


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.savefig('FI.png')


# In[ ]:


# Get feature importances
imp_df = pd.DataFrame()
imp_df["feature"] = predictors
imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
imp_df["importance_split"] = clf.feature_importance(importance_type='split')
imp_df['trn_score'] = roc_auc_score(train_df['target'], clf.predict(train_df.loc[:,predictors]))


# In[ ]:


imp_df.sort_values(by="importance_gain",ascending=False).to_csv("imp_lgb.csv", index=False)


# In[ ]:


imp_df.sort_values(by="importance_gain",ascending=False)[:150]


# Submission file

# In[ ]:


sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub_df["target"] = predictions
sub_df.to_csv("sant_lgb.csv", index=False)
sub_df[:10]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




