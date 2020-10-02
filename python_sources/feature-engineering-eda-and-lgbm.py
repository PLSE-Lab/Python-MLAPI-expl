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

pd.set_option('display.max_columns', 200)
# below is to have multiple outputs from same Jupyter cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')


# Read files 

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# <div id="EDALink">
#  **Basic EDA**
#  </div>

# Number of rows and columns in Dataset

# In[ ]:


print("Train Shape\n")
train_df.shape
print("Test Shape\n")
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
    


# To normailize features using combined dataset **normalize_df**

# In[ ]:


def normalize_df(df,features):
    for feature in features:
        #normalize
        df[feature+'_norm'] = (df[feature] - df[feature].mean())/df[feature].std()
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
        # Cumulative normal percentile
        #df[feature+'_cnp'] = norm.cdf(df[feature]).astype('float32')
    return df
    


# Lets separate the dataset for positive and negative class and check if feature distributions give us some signal.Lift and shift from [Gabriel's](https://www.kaggle.com/gpreda/santander-eda-and-prediction) kernel.

# <div id="PlotLink">
# **Plotting**
#     </div>

# Distplot for 1-100 features

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0] # segregate in two datasets correseponding to target
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[2:102] # run time errors forced this step to split into 100 sets
plot_feature_distribution(t0, t1, '0', '1', features)


# Distplot for 100-200 features

# In[ ]:


features = train_df.columns.values[102:200] 
plot_feature_distribution(t0, t1, '0', '1', features)


# Lets see how the train and test features affects target. It may give some signal if some feature is more important for target prediction.

# Boxplot 1-100 features

# In[ ]:


target   = train_df["target"]
features = train_df.columns.values[2:102]
plot_feature_boxplot(train_df, test_df, 'train', 'test', features, target)


# Boxplot 100-200 features

# In[ ]:


features = train_df.columns.values[102:200]
plot_feature_boxplot(train_df, test_df, 'train', 'test', features, target )


# Violin Plot 1-100 features

# In[ ]:


features = train_df.columns.values[2:102]
plot_feature_violinplot(train_df, test_df, 'train', 'test', features, target)


# Violin Plot 100-200 features

# In[ ]:


features = train_df.columns.values[102:200]
plot_feature_violinplot(train_df, test_df, 'train', 'test', features, target)


# All these plots above shows data between train and test is very much similar and mostly normally distributed. We may be able to use test dataset for extracting some info assuming its homogeneous with train. 

# <div id=CorLink>
# ** Correlation and Binning **
# </div>

# Lets find out corr between features, we may be able to drop couple of features if highly correlated. 

# In[ ]:


correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index() #
correlations = correlations[correlations['level_0'] != correlations['level_1']] # remove corr between same cols


# Highest correlation between top 10 features is as follows

# In[ ]:


#highest correlated features are
correlations.tail(10)


# In[ ]:


correlations = correlations.iloc[-25:,]
plt.figure()
fig, ax = plt.subplots(figsize=(10,12))
sns.heatmap(correlations.pivot_table(index='level_0',columns='level_1'))
plt.xlabel("")
plt.ylabel("")
#plt.xticks([], [])
#plt.yticks([], [])
plt.xticks(rotation=70)
plt.title("Corr plot between top 25 vars",fontsize=14)
plt.show()


# Correlation plot also shows not relation between features, looks to be pretty independent of each other.

# ### Binning

# lets try to find if binning of the features shows some trend for predicting target. We will use consistent pattern of using aa utility function and calling with 100 features in one call. 

# In[ ]:


#features = train_df.columns.values[2:102]
#plot_binned_feature_target_violinplot(train_df,features,target)


# In[ ]:


#features = train_df.columns.values[102:200]
#plot_binned_feature_target_violinplot(train_df,features,target)


# Binning plots also does not show any different story. 

# In[ ]:


import gc
gc.collect()


# <div id = FeatLink>
# ** Feature Engineering **
#     </div>

# Lets try to find if data is some sort of time series data. as one of the post was doubting. We will try to add features which will be row wise, like percentage increase from one row to next, difference from one row to next, ratio etc.

# Next we are going to combine is two data sets and try to extract some info from test dataset into train features. This idea is from [William's](https://www.kaggle.com/blackblitz/gaussian-naive-bayes) kernel. Wel will create a ratio /pct_change/diff as new features to factor for time series hypothesis.

# In[ ]:


test_df['target']= np.nan
combine_df = train_df.append(test_df,ignore_index=True)


# In[ ]:


features = train_df.columns.values[2:]
combine_df = normalize_df(combine_df,features)


# Separate out train and test. Append new features created to training dataset.

# In[ ]:


train_df = combine_df[combine_df['target'].notnull()].reset_index(drop=True)
test_df = combine_df[combine_df['target'].isnull()].reset_index(drop=True)


# Plot selectively if new any new features give some insight for target prediction.

# In[ ]:


#features = train_df.columns.values[201:]
#plot_binned_feature_target_violinplot(train_df,features,target)


# <div id = ModLink>
# ** Modeling **
#     </div>

# This is model lifted and shifted from [Fayaz's](https://www.kaggle.com/fayzur/lightgbm-customer-transaction-prediction) kernel.

# In[ ]:


#test_df = test_df.drop("target",axis=1)
predictors = train_df.columns.values.tolist()[2:]
nfold = 10
target = 'target'


# In[ ]:


param = {
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


nfold = 10

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)

oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

i = 1
for train_index, valid_index in skf.split(train_df, train_df.target.values):
    print("\nfold {}".format(i))
    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,
                           label=train_df.iloc[train_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    #print("after lgb train")
    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,
                           label=train_df.iloc[valid_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   
    #print("after lgb test")
    nround = 8523
    clf = lgb.train(param, 
                    xg_train, 
                    nround, 
                    valid_sets = [xg_valid], 
                    early_stopping_rounds=250,
                    verbose_eval=250)
    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=nround) 
    #print("after lgb fit")
    predictions += clf.predict(test_df[predictors], num_iteration=nround) / nfold
    i = i + 1

print("\n\nCV AUC: {:<0.4f}".format(metrics.roc_auc_score(train_df.target.values, oof)))


# Feature Importance as per model

# In[ ]:


fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(clf, max_num_features=100, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# Submission file

# In[ ]:


sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub_df["target"] = predictions
sub_df.to_csv("sant_lgb.csv", index=False)
sub_df[:10]

