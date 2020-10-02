#!/usr/bin/env python
# coding: utf-8

# Inspired by [this topic](https://www.kaggle.com/c/microsoft-malware-prediction/discussion/75468)
# 
# I am constantly updating this kernel.
# 
# 
# 
# **Content:**
# 
# [Loading data](#1)
# 
# [Extracting date features from OsBuildLab](#2)
# * [Creating a new feature: OsBuildReleaseYear](#3)
# * [Distribution in the combined train and test set](#4)
# * [Distribution in training set](#5)
# * [Distribution in test set](#6)
# * [Lets go a littler deeper and add month](#7)
# * [Distribution by months](#8)
#     * [Year 2018](#9)
#     * [Year 2017](#10)
#     * [Year 2016](#11) 
#   
# [Correlation with other features](#12)
# * [[0] correlation with 'OsBuild' and 'Census_OSBuildNumber'](#13)
# * [[2] correlation with 'Processor' and 'Census_OSArchitecture'](#14)
# * [[3] correlation with 'OsPlatformSubRelease' and 'Census_OSBranch'](#15)
#      * [Plotting rs3 values correlation with HasDetections from OsPlatformSubRelease](#16)
#      * [Plotting rs3_release and rs3_release_svc_escrow values correlation with HasDetections from Census_OSBranch](#17)
#      * [Plotting rs3_release and rs3_release_svc_escrow values correlation with HasDetections from [2] - feature exctracted from OsBuildLab](#18)

# <a id="1"></a> <br>
# # Loading data

# In[ ]:


import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import gc
import multiprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 83)
pd.set_option('display.max_rows', 20)
for package in [np, pd, sns]:
    print(package.__name__, 'version:', package.__version__)
import os
print(os.listdir("../input"))


# In[ ]:


dtypes = {
    'OsBuildLab':                                           'category',
    'Processor':                                            'category',
    'OsPlatformSubRelease':                                 'category',
    'Census_OSArchitecture':                                'category',
    'Census_OSBranch':                                      'category',
    'Census_OSBuildNumber':                                 'int16',
    'Census_OSBuildRevision':                               'int32',
    'OsBuild':                                              'int16',
    'HasDetections':                                        'int8'
}


# In[ ]:


def load_dataframe(dataset):
    usecols = dtypes.keys()
    if dataset == 'test':
        usecols = [col for col in dtypes.keys() if col != 'HasDetections']
    df = pd.read_csv(f'../input/{dataset}.csv', dtype=dtypes, usecols=usecols)
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with multiprocessing.Pool() as pool: \n    train, test = pool.map(load_dataframe, ["train", "test"])')


# In[ ]:


df = pd.concat([train.drop('HasDetections', axis=1), test])


# In[ ]:


df['OsBuildLab'].value_counts().head()


# <a id="2"></a>
# # Extracting date features from OsBuildLab
# <a id="3"></a>
# ## Creating a new feature: OsBuildReleaseYear

# In[ ]:


df['OsBuildReleaseYear'] = df['OsBuildLab'].str.slice(start=-11, stop=-9)
train['OsBuildReleaseYear'] = train['OsBuildLab'].str.slice(start=-11, stop=-9)
test['OsBuildReleaseYear'] = test['OsBuildLab'].str.slice(start=-11, stop=-9)


# <a id="4"></a>
# ## Distribution in the combined train and test set

# In[ ]:


df['OsBuildReleaseYear'].value_counts(dropna=False).plot(kind='bar', figsize=(12,6), rot=0);


# <a id="5"></a>
# ## Distribution in training set

# In[ ]:


train['OsBuildReleaseYear'].value_counts(dropna=False).plot(kind='bar', figsize=(12,6), rot=0);


# <a id="6"></a>
# ## Distribution in test set

# In[ ]:


test['OsBuildReleaseYear'].value_counts(dropna=False).plot(kind='bar', figsize=(12,6), rot=0);


# So distribution by years is almost identical amongst train and test sets.
# 
# Lets see the correlation of BuildYear with target value.

# In[ ]:


plt.subplots(figsize=(14,6))
sns.countplot(x='OsBuildReleaseYear', hue='HasDetections', data=train);


# <a id="7"></a>
# ## Lets go a littler deeper and add month

# In[ ]:


df['OsBuildReleaseYearMonth'] = df['OsBuildLab'].str.slice(start=-11, stop=-7).astype('float16')
train['OsBuildReleaseYearMonth'] = train['OsBuildLab'].str.slice(start=-11, stop=-7).astype('float16')
test['OsBuildReleaseYearMonth'] = test['OsBuildLab'].str.slice(start=-11, stop=-7).astype('float16')


# In[ ]:


df['OsBuildReleaseYearMonth'].value_counts(dropna=False).head(10).plot(kind='bar', rot=0, figsize=(12,6));


# In[ ]:


train['OsBuildReleaseYearMonth'].value_counts(dropna=False).head(10).plot(kind='bar', rot=0, figsize=(12,6));


# In[ ]:


test['OsBuildReleaseYearMonth'].value_counts(dropna=False).head(10).plot(kind='bar', rot=0, figsize=(12,6));


# Distribution is still almost identical. Now lets see year and month of build correlation with target value

# <a id="8"></a>
# ## Distribution by months
# <a id="9"></a>
# ### Year 2018

# In[ ]:


plt.subplots(figsize=(14,6))
sns.countplot(x='OsBuildReleaseYearMonth', hue='HasDetections', data=train[train['OsBuildReleaseYearMonth'] >= 1800]);


# <a id="10"></a>
# ### Year 2017

# In[ ]:


plt.subplots(figsize=(14,6))
sns.countplot(x='OsBuildReleaseYearMonth', hue='HasDetections', data=train[(train['OsBuildReleaseYearMonth'] < 1800) & (train['OsBuildReleaseYearMonth'] >= 1700)]);


# <a id="11"></a>
# ### Year 2016

# In[ ]:


plt.subplots(figsize=(14,6))
sns.countplot(x='OsBuildReleaseYearMonth', hue='HasDetections', data=train[(train['OsBuildReleaseYearMonth'] < 1700) & (train['OsBuildReleaseYearMonth'] >= 1600)]);


# Ok, so builds of 2018 (especially of april 2018) have biger detection rate than build of 2017.

# <a id="12"></a>
# # Correlation with other features
# 
# Now lets take a good look at what other information this feature contains because it consist not only of release date and time value.
# 
# Releasing some memory first. We gonna need it.

# In[ ]:


del df
del test
del train['OsBuildReleaseYear'], train['OsBuildReleaseYearMonth']
gc.collect();


# Some of the OsBuildLab feature values contains a typo - * (asterisk) instead of . (dot), so lets fix this and then split the values by . (dot) and save it as a separate features.

# In[ ]:


train = pd.concat([train, train['OsBuildLab'].str.replace('*', '.').str.split('.', expand=True)], axis=1)


# In[ ]:


train[0] = train[0].fillna(-1).astype('int16')
train[1] = train[1].fillna(-1).astype('int16')
train.head()


# In[ ]:


sns.heatmap(train.corr(), cmap="YlGnBu");


# <a id="13"></a>
# ### [0] correlation with 'OsBuild' and 'Census_OSBuildNumber'

# Alright, we already can see by the first 5 rows that some data is exactly the same for OsBuildLab splitted by dot and some other features, for example OsBuild (even the features names are giving us a hint). But lets see all the values just to make sure that this is true.

# In[ ]:


print(train[(train['Census_OSBuildNumber'] != train['OsBuild']) | (train['Census_OSBuildNumber'] != train[0]) | (train['OsBuild'] != train[0])][[0, 'OsBuild', 'Census_OSBuildNumber']].shape[0], 'differences')
train[(train['Census_OSBuildNumber'] != train['OsBuild']) | (train['Census_OSBuildNumber'] != train[0]) | (train['OsBuild'] != train[0])][[0, 'OsBuild', 'Census_OSBuildNumber']].head()


# Dataset has 445510 differences amongst OsBuild, Census_OsBuildNumber and a [0] feature extraceted from OsBuildLab. And most of this differences comes from Census_OSBuildNumber feature.

# In[ ]:


train[['OsBuild', 'Census_OSBuildNumber', 0]].corr()


# These 3 features are highly correlated it is it pretty safe to say that either [0] or OsBuild might be dropped. But since there is a slight difference between Census_OSBuildNumber and other 2 features you should decide which one of them to keep.

# <a id="14"></a>
# ### [2] correlation with 'Processor' and 'Census_OSArchitecture'

# Next in the line is Processor feature and Census_OSArchitecture featrure. It is easy to see that feature 2, extracted from OsBuildLab is the same as those two features, just listed differently.

# In[ ]:


train['Processor'].value_counts(dropna=False)


# In[ ]:


train['Census_OSArchitecture'].value_counts(dropna=False)


# In[ ]:


train[2].value_counts(dropna=False)


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,12))
sns.countplot(x='Processor', hue='HasDetections', data=train, ax=axes[0], order=['x64', 'arm64', 'x86']);
sns.countplot(x='Census_OSArchitecture', hue='HasDetections', data=train, ax=axes[1], order=['amd64', 'arm64', 'x86']);
sns.countplot(x=2, hue='HasDetections', data=train, ax=axes[2], order=['amd64fre', 'arm64fre', 'x86fre']);


# It is easy to see that this 3 features are exactly the same. So drop 2 of them and leave only 1.

# <a id="15"></a>
# ### [3] correlation with 'OsPlatformSubRelease' and 'Census_OSBranch'

# Alring, so we see that OsBuildLab contains the exact information that is already presented in the data set. Lets see the third one.

# In[ ]:


train[['OsPlatformSubRelease', 'Census_OSBranch', 3]].head(10)


# In[ ]:


train['OsPlatformSubRelease'].value_counts(dropna=False)


# In[ ]:


train['Census_OSBranch'].value_counts(dropna=False).head(10)


# In[ ]:


train[3].value_counts(dropna=False).head(10)


# As expected - the information is almost the same but it has slight difference - 'OsPlatformSubRelease' contains more generalized information (e.g. rs3), whereas feature 3, exctacted from OsBuildLab and 'Census_OSBranch' has information presented in more specified way (e.g. rs3 splitted in rs3_release_svc_escrow and rs3_release).
# 
# Lets plot those values correlations with target value and see the results.

# <a id="16"></a>
# #### Plotting rs3 values correlation with HasDetections from OsPlatformSubRelease

# In[ ]:


train.loc[train['OsPlatformSubRelease'] == 'rs3', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['green', 'orange']).set_xlabel('rs3', fontsize=18);


# So release3 (rs3) has almost the same detection rate distibution. But lets see if its (rs3) division into 2 specific subreleases do the same.

# <a id="17"></a>
# #### Plotting rs3_release and rs3_release_svc_escrow values correlation with HasDetections from Census_OSBranch

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
train.loc[train['Census_OSBranch'] == 'rs3_release', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['green', 'orange'], ax=axes[0]).set_xlabel('rs3_release', fontsize=18);
train.loc[train['Census_OSBranch'] == 'rs3_release_svc_escrow', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['orange', 'green'], ax=axes[1]).set_xlabel('rs3_release_svc_escrow', fontsize=18);
plt.gca().invert_xaxis();


# <a id="18"></a>
# #### Plotting rs3_release and rs3_release_svc_escrow values correlation with HasDetections from [2] - feature exctracted from OsBuildLab

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
train.loc[train[3] == 'rs3_release', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['green', 'orange'], ax=axes[0]).set_xlabel('rs3_release', fontsize=18);
train.loc[train[3] == 'rs3_release_svc_escrow', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['orange', 'green'], ax=axes[1]).set_xlabel('rs3_release_svc_escrow', fontsize=18);
plt.gca().invert_xaxis();


# Alright, the distibution differs on those sub-releases. And one more time - there are 3 features representing almost the same thing but this time values are slightly different (see value_counts output for each feature). Which one to leave and which to remove is up to you.
