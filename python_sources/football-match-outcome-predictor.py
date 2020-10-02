#!/usr/bin/env python
# coding: utf-8

# # Football Match Outcome Predictor
# 
# ## Problem Statement:
# 
# Build a model to predict the outcome of a football match, given data for the past 9 years. 
# 
# ## Goal:
# 
# Come up with an optimal solution to predict if a Home Team would win or lose or draw (FTR - Target Feature) for the year of 2017 - 18.
# 
# ## Feature Details:
# 
# **HomeTeam:** Home Team
# 
# **AwayTeam:** Away Team
# 
# **FTR:** Full Time Result (Target Feature)
# 
# **HTHG:** Half-Time Home Team Goals
# 
# **HTAG:** Half-Time Away Team Goals
# 
# **HS:** Home Team Shots
# 
# **AS:** Away Team Shots
# 
# **HST:** Home Team Shots on Target
# 
# **AST:** Away Team Shots on Target
# 
# **AC:** Away Team Corners
# 
# **HF:** Home Team Fouls Committed
# 
# **AF:** Away Team Fouls Committed
# 
# **HC:** Home Team Corners
# 
# **HY:** Home Team Yellow Cards
# 
# **AY:** Away Team Yellow Cards
# 
# **HR:** Home Team red Cards
# 
# **AR:** Away Team Red Cards
# 
# **Date:** On which day the match was played
# 
# **league:** Under which league the match was played
# 
# ### Instructions before executing this kernel:
# 
# **Libraries needed:** LightGBM, seaborn. Others I have managed to keep standard packages and modules.
# 
# **Folder Structure:** Create input and output folder in the level of .ipynb file. Add input files to input folder and output will get written to output folder.

# In[ ]:


# Import libraries

import pandas as pd
import numpy as np

# Check the folder structure
import os
import datetime
print(os.listdir('../input'))

#Visualisations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings("ignore")

#Model
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb

#Metrics
from sklearn.metrics import roc_auc_score, roc_curve


# In[ ]:


#Let's load the train and test datasets
df_train = pd.read_csv('../input/train.csv', parse_dates=['Date'])
df_test = pd.read_csv('../input/test-3.csv', index_col='index', parse_dates=['Date'])


# ### Let's check the data and understand the data types and other statistics of the features

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# Referee feature has too many NaNs and is present only in test set. Let's drop that feature.

# In[ ]:


df_test = df_test.drop('Referee', axis=1)


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# Observations:
# 
# The AwayTeam, HomeTeam and League features are objects. 
# 
# Mean values are distributed over a long range.
# 
# Min, max, mean, std values for train and test data looks close.
# 

# In[ ]:


df_train.shape, df_test.shape


# # Let's check the output feature distribution.

# In[ ]:


df_train['FTR'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x=df_train['FTR'].value_counts().index, y = df_train['FTR'].value_counts().values, color='orange')
ax.set_title('Distribution of Full Time Result', fontsize=14)
ax.set_xlabel('Full Time Result', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.show()


# Home Teams won ~46% of the matches. Away team won ~28% of the matches. ~26% matches were "draw". We will deal with the class imbalance later.

# # Null handling

# In[ ]:


df_train[df_train['AF'].isnull()]


# To treat the null values we will follow the process as below:
# 
# 1. Drop the samples for which both hometeam and awayteam are null
# 2. For the rest, Group all the features by hometem and take mode and fill null values

# In[ ]:


drop_index = df_train[(df_train['HomeTeam'].isnull()) & (df_train['AwayTeam'].isnull())].index
df_train.drop(drop_index, inplace=True)


# In[ ]:


#Define a function which returns a null dataframe
def check_null(df):
    df_null = df.isna().sum().reset_index()
    df_null.columns = ['Column', 'Null_Count']
    df_null = df_null[df_null['Null_Count'] > 0]
    df_null = df_null.sort_values(by='Null_Count', ascending=False).reset_index(drop=True)
    return df_null


# In[ ]:


df_null_train = check_null(df_train)
df_null_train


# In[ ]:


#Let's fill null values grouped by the hometeam
for _, item in df_null_train.iterrows():
    column = item['Column']
    df_train[column] = df_train.groupby(['HomeTeam'])[column].transform(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


check_null(df_train)


# In[ ]:


check_null(df_test)


# # Unique Values

# In[ ]:


def check_unique(df):
    df_unique = df.nunique().reset_index()
    df_unique.columns = ['Column', 'Unique_Count']
    df_unique = df_unique[df_unique['Unique_Count'] < 2]
    df_unique = df_unique.sort_values(by='Unique_Count', ascending=False).reset_index(drop=True)
    return df_unique


# In[ ]:


check_unique(df_train)


# In[ ]:


check_unique(df_test)


# There are no features with unique values.

# # Exploratory Data Analysis
# 
# ## AC - Away Team Corners - Training set

# In[ ]:


def plot_feature_distributions(df, features, palette):
    i = 0
    plt.figure()
    fig, ax = plt.subplots(len(features),1,figsize=(14,35))
    plt.subplots_adjust(bottom=0.001)

    for feature in features:
        i += 1
        plt.subplot(len(features),1,i)
        sns.barplot(df[feature].value_counts().index, df[feature].value_counts().values, palette=palette)
        plt.title('Distribution of {0}'.format(feature), fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Count', fontsize=12)
    plt.show()


# In[ ]:


def plot_feature_distribution_hue_target(df, features, palette):
    i = 0
    plt.figure()
    fig, ax = plt.subplots(len(features),1,figsize=(14,35))
    plt.subplots_adjust(bottom=0.001)

    for feature in features:
        i += 1
        plt.subplot(len(features),1,i)
        sns.countplot(x=feature, hue='FTR', data=df, palette=palette)
        plt.title('Distribution of {0} by Full Time Result'.format(feature), fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Count', fontsize=12)
    plt.show()


# ## Let's check the distribution of training set features for Away Team

# In[ ]:


features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HTAG']
plot_feature_distributions(df_train, features, 'RdBu')


# ## Let's check the distribution of test set features for Away Team

# In[ ]:


features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HTAG']
plot_feature_distributions(df_test, features, 'PuOr')


# As we can see in the above distributions, there are feature items whose value counts are too less. We will bin some of the classes which I have decided after checking the skew and kurtosis plotted down below.

# In[ ]:


#Let's bin Away Team Corners values greater than 11.0 to 11.0
df_train['AC'] = df_train['AC'].transform(lambda x: 11.0 if x > 11.0 else x)
df_test['AC'] = df_test['AC'].transform(lambda x: 11.0 if x > 11.0 else x)

#Let's bin values greater than 30.0 to 30.0 and test values greater than 26.0 to 26.0
df_train['AF'] = df_train['AF'].transform(lambda x: 30.0 if x > 30.0 else x)
df_test['AF'] = df_test['AF'].transform(lambda x: 26.0 if x > 26.0 else x)

#Let's bin Away Team shots values greater than 27.0 to 27.0
df_train['AS'] = df_train['AS'].transform(lambda x: 27.0 if x > 27.0 else x)
df_test['AS'] = df_test['AS'].transform(lambda x: 27.0 if x > 27.0 else x)

#Let's bin Away Team shots on Target values greater than 12.0 to 12.0
df_train['AST'] = df_train['AST'].transform(lambda x: 12.0 if x > 12.0 else x)
df_test['AST'] = df_test['AST'].transform(lambda x: 12.0 if x > 12.0 else x)

#Let's bin Away Team Yellow Card values greater than 6.0 to 6.0
df_train['AY'] = df_train['AY'].transform(lambda x: 6.0 if x > 6.0 else x)
df_test['AY'] = df_test['AY'].transform(lambda x: 6.0 if x > 6.0 else x)


# ## Let's now plot the distribution of Away Team features by Full Time Result

# In[ ]:


features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HTAG']
plot_feature_distribution_hue_target(df_train, features, 'coolwarm')


# **Observations:**
# 
# Away Team Corners did not increase the prospect of away team winning irrespective of the numbers of corners.
# 
# As away team committed more fouls, we see the home team winning more.
# 
# Red card for away team increased the prospect of home team winning.
# 
# The more the shots and shots on target, away team won more.
# 
# Yellow card marginally had impact on the matches away team won.
# 
# Half-time away team goals had a huge impact in away teams winning.

# ## Let's do the same for home team features in training set

# In[ ]:


features = ['HC', 'HF', 'HR', 'HS', 'HST', 'HTHG', 'HY']
plot_feature_distributions(df_train, features, 'RdYlBu')


# ## Let's check the distribution of test set features for Away Team

# In[ ]:


features = ['HC', 'HF', 'HR', 'HS', 'HST', 'HTHG', 'HY']
plot_feature_distributions(df_test, features, 'YlGnBu')


# In[ ]:


#Let's bin Home Team Corners values greater than 14.0 to 14.0
df_train['HC'] = df_train['HC'].transform(lambda x: 14.0 if x > 11.0 else x)
df_test['HC'] = df_test['HC'].transform(lambda x: 11.0 if x > 11.0 else x)

#Let's bin Home Team values greater than 27.0 to 27.0 and test values greater than 24.0 to 24.0
df_train['HF'] = df_train['HF'].transform(lambda x: 27.0 if x > 27.0 else x)
df_test['HF'] = df_test['HF'].transform(lambda x: 24.0 if x > 24.0 else x)

#Let's bin Home Team shots values greater than 30.0 to 30.0
df_train['HS'] = df_train['HS'].transform(lambda x: 30.0 if x > 30.0 else x)
df_test['HS'] = df_test['HS'].transform(lambda x: 30.0 if x > 30.0 else x)

#Let's bin Home Team shots on Target values greater than 14.0 to 14.0 and 12.0 to 12.0 for test set
df_train['HST'] = df_train['HST'].transform(lambda x: 14.0 if x > 14.0 else x)
df_test['HST'] = df_test['HST'].transform(lambda x: 12.0 if x > 12.0 else x)

#Let's bin Away Team Yellow Card values greater than 6.0 to 6.0
df_train['HY'] = df_train['HY'].transform(lambda x: 6.0 if x > 6.0 else x)
df_test['HY'] = df_test['HY'].transform(lambda x: 6.0 if x > 6.0 else x)


# In[ ]:


features = ['HC', 'HF', 'HR', 'HS', 'HST', 'HTHG', 'HY']
plot_feature_distribution_hue_target(df_train, features, 'coolwarm')


# **Observations:**
# 
# Home Team Corners increases the prospect of home team winning the matches.
# 
# Fouls committed by home team did not have any significnt impact.
# 
# Red card for home team increased the prospect of away team winning.
# 
# The more the shots and shots on target, home team won more.
# 
# Yellow card had minimal impact on the matches home team won.
# 
# Half-time home team goals had a huge impact in home teams winning.

# ## More Data Analysis
# 
# ### Let's see how teams performed home and away.
# 
# First let's plot how many matches each teams played.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 35))
sns.barplot(x=df_train['HomeTeam'].value_counts().values, y=df_train['HomeTeam'].value_counts().index, color='Orange')
ax.set_title('Matches played by teams home between 2009 and 2017', fontsize=14)
ax.set_xlabel('Matches Played', fontsize=12)
ax.set_ylabel('Teams', fontsize=12)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 35))
sns.barplot(x=df_train['AwayTeam'].value_counts().values, y=df_train['AwayTeam'].value_counts().index, color='Green')
ax.set_title('Matches played by teams away between 2009 and 2017')
ax.set_xlabel('Matches Played', fontsize=12)
ax.set_ylabel('Teams', fontsize=12)
plt.show()


# Teams played home and away matches almost equally.
# 
# **Now, let's how home teams and away teams fared with half time goals**

# In[ ]:


df_train['Year'] = df_train['Date'].dt.year
df_train['Month'] = df_train['Date'].dt.month
df_train['FTR'] = df_train['FTR'].map({'H':0, 'D':1, 'A':2})
df_test['Year'] = df_test['Date'].dt.year
df_test['Month'] = df_test['Date'].dt.month


# In[ ]:


df_HTHG = df_train.groupby(['HomeTeam'])['HomeTeam', 'HTHG'].sum().reset_index()
df_HTHG = df_HTHG.sort_values(by='HTHG', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 35))
sns.barplot(x=df_HTHG['HTHG'], y=df_HTHG['HomeTeam'], color='yellow')
ax.set_title('Half Time goals by home teams', fontsize=14)
ax.set_xlabel('Goals', fontsize=12)
ax.set_ylabel('Teams', fontsize=12)
plt.show()


# In[ ]:


df_HTAG = df_train.groupby(['AwayTeam'])['AwayTeam', 'HTAG'].sum().reset_index()
df_HTAG = df_HTAG.sort_values(by='HTAG', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 35))
sns.barplot(x=df_HTAG['HTAG'], y=df_HTAG['AwayTeam'], color='blue')
ax.set_title('Half Time goals by away teams', fontsize=14)
ax.set_xlabel('Goals', fontsize=12)
ax.set_ylabel('Teams', fontsize=12)
plt.show()


# Leading goal scorers when playing at home: Real Madrid, Barcelona, Bayern Munich, Chelsea, ManU, Paris SG.
# 
# Leading goal scorers when playing away: Real Madris, Barcelona, Arsenal, Man City, Dotmund.
# 
# **Let's plot Half Time goals scored by home and away teams by season**

# In[ ]:


df_HTHG_season = df_train.groupby(['Year'])['HTHG'].sum().reset_index()
fig, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x=df_HTHG_season['Year'], y=df_HTHG_season['HTHG'], palette='RdBu')
ax.set_title('Half Time goals by Home Team by season', fontsize=14)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Goals', fontsize=12)
plt.show()


# In[ ]:


df_HTAG_season = df_train.groupby(['Year'])['HTAG'].sum().reset_index()
fig, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x=df_HTAG_season['Year'], y=df_HTAG_season['HTAG'], palette='RdBu')
ax.set_title('Half Time goals by Away Team by season', fontsize=14)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Goals', fontsize=12)
plt.show()


# Let's add a new analysis for FTW(Full Time Winner) and HTW(Half Time Winner)

# In[ ]:


conditions = [df_train['FTR']==2,df_train['FTR']==0,df_train['FTR']==1]
select = [df_train['AwayTeam'],df_train['HomeTeam'],'Draw']
df_train['FTW']=np.select(conditions, select)


# In[ ]:


df_Winner = df_train['FTW'].value_counts().reset_index()
df_Winner.columns = ['Team', 'Win_Counts']

#Dropping Winner Feature as we will not be able to produce the same in test set and for modelling
df_train.drop('FTW', axis=1, inplace=True)


#Drop Draws from the dataframe
df_Winner.drop(df_Winner.head(1).index, axis=0, inplace=True)
df_Winner = df_Winner.head(20)
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x=df_Winner['Team'], y=df_Winner['Win_Counts'], palette='GnBu_r')
ax.set_title('Teams with maximum full-time wins', fontsize=14)
ax.set_xlabel('Teams', fontsize=12)
ax.set_ylabel('Wins', fontsize=12)
ax.set_xticklabels(df_Winner['Team'], rotation=45)
plt.show()


# In[ ]:


#Lambda Function
def select_winner(x):
    if x > 0:
        return 0
    elif x < 0:
        return 2
    else:
        return 1

def transform_HTR(df):
    df['HTW'] = df['HTHG'] - df['HTAG']
    df['HTW'] = df['HTW'].transform(lambda x: select_winner(x))
    conditions = [df['HTW']==2,df['HTW']==0,df['HTW']==1]
    select = [df['AwayTeam'],df['HomeTeam'],'Draw']
    df['HTW']=np.select(conditions, select)
    return df['HTW']
    
df_train['HTW'] = transform_HTR(df_train)


# In[ ]:


df_Winner = df_train['HTW'].value_counts().reset_index()
df_Winner.columns = ['Team', 'Win_Counts']

#Dropping Winner Feature as we will not be able to produce the same in test set and for modelling
df_train.drop('HTW', axis=1, inplace=True)

#Drop Draws from the dataframe
df_Winner.drop(df_Winner.head(1).index, axis=0, inplace=True)
df_Winner = df_Winner.head(20)
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x=df_Winner['Team'], y=df_Winner['Win_Counts'], palette='GnBu_r')
ax.set_title('Teams with maximum half-time wins', fontsize=14)
ax.set_xlabel('Teams', fontsize=12)
ax.set_ylabel('Wins', fontsize=12)
ax.set_xticklabels(df_Winner['Team'], rotation=45)
plt.show()


# # Statistical analysis of features by row
# 
# **Let's try to get the statistical values per row for train and test set.**

# In[ ]:


plt.figure(figsize=(16,8))
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(df_train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(df_test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(df_train[features].std(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(df_test[features].std(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']
plt.title("Distribution of min values per row in the train and test set")
sns.distplot(df_train[features].min(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(df_test[features].min(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']
plt.title("Distribution of max values per row in the train and test set")
sns.distplot(df_train[features].max(axis=1),color="gold", kde=True,bins=120, label='train')
sns.distplot(df_test[features].max(axis=1),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']
plt.title("Distribution of skew values per row in the train and test set")
sns.distplot(df_train[features].skew(axis=1),color="gold", kde=True,bins=120, label='train')
sns.distplot(df_test[features].skew(axis=1),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']
plt.title("Distribution of kurtosis values per row in the train and test set")
sns.distplot(df_train[features].kurtosis(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(df_test[features].kurtosis(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# # Correlation of train and test features

# In[ ]:


df_train_corr = df_train.corr()


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_train_corr, cmap='RdYlBu_r', annot=True)
ax.set_title('Correlation of training set features', fontsize=14)
plt.show()


# In[ ]:


df_test_corr = df_test.corr()


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_train_corr, cmap='RdYlBu_r', annot=True)
ax.set_title('Correlation of test set features', fontsize=14)
plt.show()


# **Observation:**
# 
# We have similar correlation of input features for both train and test set features

# # Feature Engineering
# 
# Let's start with calculating few aggregate values for the existing features

# In[ ]:


get_ipython().run_cell_magic('time', '', "idx = features = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'HC','HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HY']\nfor df in [df_test, df_train]:\n    df['sum'] = df[idx].sum(axis=1)  \n    df['max'] = df[idx].max(axis=1)\n    df['mean'] = df[idx].mean(axis=1)\n    df['std'] = df[idx].std(axis=1)\n    df['skew'] = df[idx].skew(axis=1)\n    df['kurt'] = df[idx].kurtosis(axis=1)\n    df['med'] = df[idx].median(axis=1)")


# Let's add features based on Date

# In[ ]:


get_ipython().run_cell_magic('time', '', "for df in [df_test, df_train]:\n    df['weekofyear'] = df['Date'].dt.weekofyear\n    df['dayofweek'] = df['Date'].dt.dayofweek\n    df['weekend'] = (df['Date'].dt.dayofweek >= 5).astype('int')\n    df['quarter'] = df['Date'].dt.quarter\n    df['is_month_start'] = df['Date'].dt.is_month_start\n    df['month_diff'] = ((datetime.datetime.today() - df['Date']).dt.days)//30")


# Let's add aggregate of away team features based on "HomeTeam" and home team based on "AwayTeam".

# In[ ]:


def aggregate_away_metrics(df, prefix):
    agg_func = {
        'AC': ['sum', 'mean', 'max', 'std', 'count'],
        'AF': ['sum', 'mean', 'max', 'std', 'count'],
        'AR': ['sum', 'mean', 'max', 'std', 'count'],
        'AS': ['sum', 'mean', 'max', 'std', 'count'],
        'AST': ['sum', 'mean', 'max', 'std', 'count'],
        'AY': ['sum', 'mean', 'max', 'std', 'count'],
        'HTAG': ['sum', 'mean', 'max', 'std', 'count']
    }
    
    agg_transactions = df.groupby(['HomeTeam']).agg(agg_func)
    agg_transactions.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_transactions.columns.values]
    agg_transactions.reset_index(inplace=True)
    return agg_transactions


# In[ ]:


agg_away = aggregate_away_metrics(df_train, 'away_')


# In[ ]:


df_train = pd.merge(df_train, agg_away, on='HomeTeam', how='left')
df_train.shape


# In[ ]:


agg_away = aggregate_away_metrics(df_test, 'away_')


# In[ ]:


df_test = pd.merge(df_test, agg_away, on='HomeTeam', how='left')
df_test.shape


# In[ ]:


def aggregate_home_metrics(df, prefix):
    agg_func = {
        'HC': ['sum', 'mean', 'max', 'std', 'count'],
        'HF': ['sum', 'mean', 'max', 'std', 'count'],
        'HR': ['sum', 'mean', 'max', 'std', 'count'],
        'HS': ['sum', 'mean', 'max', 'std', 'count'],
        'HST': ['sum', 'mean', 'max', 'std', 'count'],
        'HY': ['sum', 'mean', 'max', 'std', 'count'],
        'HTHG': ['sum', 'mean', 'max', 'std', 'count']
    }
    
    agg_transactions = df.groupby(['AwayTeam']).agg(agg_func)
    agg_transactions.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_transactions.columns.values]
    agg_transactions.reset_index(inplace=True)
    return agg_transactions


# In[ ]:


agg_home = aggregate_home_metrics(df_train, 'home_')


# In[ ]:


df_train = pd.merge(df_train, agg_home, on='AwayTeam', how='left')
df_train.shape


# In[ ]:


agg_home = aggregate_home_metrics(df_test, 'home_')


# In[ ]:


df_test = pd.merge(df_test, agg_home, on='AwayTeam', how='left')
df_test.shape


# We will remove Date, Hometeam, AwayTeam and OneHot encode league feature. 

# In[ ]:


df_train = df_train.join(pd.get_dummies(df_train['league']))
df_train.drop('league', axis=1, inplace=True)
df_train.head()


# In[ ]:


df_test = df_test.join(pd.get_dummies(df_test['league']))
df_test.drop('league', axis=1, inplace=True)
df_test.head()


# In[ ]:


df_train.drop(['Date', 'HomeTeam', 'AwayTeam'], axis=1, inplace=True)


# In[ ]:


df_test.drop(['Date', 'HomeTeam', 'AwayTeam'], axis=1, inplace=True)


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


Y_train = df_train['FTR']
X_train = df_train.drop(['FTR'], axis=1)
X_test = df_test


# # Model
# 
# Let's define the hyperparameters for the model.

# In[ ]:


param = {
    'max_bin': 119,
    'min_data_in_leaf': 11,
    'learning_rate': 0.001,
    'min_sum_hessian_in_leaf': 0.00245,
    'bagging_fraction': 0.7, 
    'bagging_freq': 5, 
    'lambda_l1': 4.972,
    'lambda_l2': 2.276,
    'min_gain_to_split': 0.65,
    'max_depth': 14,
    'save_binary': True,
    'seed': 1337,
    'feature_fraction_seed': 1337,
    'bagging_seed': 1337,
    'drop_seed': 1337,
    'data_random_seed': 1337,
    'verbose': 1,
    'is_unbalance': True,
    'boost': 'gbdt',
    'feature_fraction' : 0.8,  # colsample_bytree
    'metric':'multi_logloss',
    'num_leaves': 30,
    'objective' : 'multiclass',
    'num_class' : 3,
    'verbosity': 1
}


# In[ ]:


folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros((len(X_train), 3))
predictions = np.zeros((len(X_test), 3))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, Y_train.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(X_train.iloc[trn_idx][X_train.columns], label=Y_train.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx][X_train.columns], label=Y_train.iloc[val_idx])

    num_round = 20000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=2000, early_stopping_rounds = 500)
    oof[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    print('Fold Validation Set Accuracy: {0}'.format(np.mean(Y_train[val_idx] == np.argmax(oof[val_idx],axis=1))))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X_train.columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(X_test[features], num_iteration=clf.best_iteration) / folds.n_splits


# Based on the feature importance plotted below, we can see the top 10 important features: 
# 
# **HST, AST, HTHG, HTAG, month_diff, HC, std, away_HTAG_std, away_AC_mean, kurtosis**

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


output = pd.DataFrame({ 'index' : df_test.index, 'FTR': np.argmax(predictions, axis=1) })
output.tail()


# By applying additional boosting algorithms like XGBoost and CatBoost, the performance can be improved. 
# I wil also source extra data and improve this model and send it as v2.
# 
# Happy Programming!!
