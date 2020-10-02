#!/usr/bin/env python
# coding: utf-8

# The dataset, provided by Microsoft, containt some features, that are higly correlated with each other and some of them are even completely the same. This kernel is all about this kind of features.
# 
# **Content**
# 
# * [Loading data](#1)
# * [Processor / Census_OSArchitecture](#2)
# * [OsPlatformSubRelease / Census_OSBranch](#3)
# * [OsVer / Platform:](#4)
#     
#     A. [Correlation](#5)
#     
#     B. [Interaction](#6)
# * [Census_OSEdition / Census_OSSkuName](#7)
# * [Census_OSInstallLanguageIdentifier / Census_OSUILocaleIdentifier](#8)
# * [Census_OSBuildNumber / OsBuild](#9)
# * [Census_PowerPlatformRoleName / Census_MDC2FormFactor / Census_ChassisTypeName](#10)

# Libraries versions been used in this kernel (usefull for results reproducibility)

# In[ ]:


import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import gc
import platform
import multiprocessing
import seaborn as sns
import sys
from tqdm import tqdm
pd.set_option('display.max_columns', 83)
pd.set_option('display.max_rows', 83)
plt.style.use('seaborn')
import os
import matplotlib
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import cufflinks
import plotly
for package in [pd, np, sns, matplotlib, plotly]:
    print(package.__name__, 'version:', package.__version__)
print('python version:', platform.python_version())
init_notebook_mode()


# <a id="1"></a>
# # Loading data

# In[ ]:


dtypes = {
    'Census_OSSkuName':                                     'category',
    'SkuEdition':                                           'category',
    'Census_OSEdition':                                     'category',
    'Platform':                                             'category',
    'OsVer':                                                'category',
    'Census_OSInstallLanguageIdentifier':                   'float16',
    'Census_OSUILocaleIdentifier':                          'int16',
    'Firewall':                                             'float16',
    'HasTpm':                                               'int8',
    'OsBuildLab':                                           'category',
    'Census_OSBuildNumber':                                 'int16',
    'OsBuild':                                              'int16',
    'Processor':                                            'category',
    'Census_OSArchitecture':                                'category',
    'Census_OSBranch':                                      'category',
    'OsPlatformSubRelease':                                 'category',
    'Census_MDC2FormFactor':                                'category',
    'Census_PowerPlatformRoleName':                         'category',
    'Census_ChassisTypeName':                               'category',
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


def label_encoder(df, features, keep_original=True, suffix='_label_encoded'):
    """
    Features should be list
    """
    for feature in tqdm(features):
        df[feature + suffix] = pd.factorize(df[feature])[0]
        if not keep_original:
            del df[feature]
    return df


# <a id="2"></a>
# # Processor / Census_OSArchitecture

# In[ ]:


df['Processor'].value_counts(dropna=False)


# In[ ]:


df['Census_OSArchitecture'].value_counts(dropna=False)


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,10))
sns.countplot(x='Processor', hue='HasDetections', data=train, ax=axes[0], order=['x64', 'arm64', 'x86']);
sns.countplot(x='Census_OSArchitecture', hue='HasDetections', data=train, ax=axes[1], order=['amd64', 'arm64', 'x86']);


# The distibution almost the same and correlation with HasDetections as well.
# 
# Next doing simple label encoding and building a correlation matrix. First for the whole dataset and then only for training set also including 'HasDetections'

# In[ ]:


df = label_encoder(df, ['Processor', 'Census_OSArchitecture'])
train = label_encoder(train, ['Processor', 'Census_OSArchitecture'])
df[['Processor_label_encoded', 'Census_OSArchitecture_label_encoded']].corr()


# In[ ]:


train[['Processor_label_encoded', 'Census_OSArchitecture_label_encoded', 'HasDetections']].corr()


# Values in this features are almost the same. One of them can be easily removed, or even both of them.

# <a id="3"></a>
# # OsPlatformSubRelease / Census_OSBranch

# In[ ]:


df['OsPlatformSubRelease'].value_counts(dropna=False).head()


# In[ ]:


df['Census_OSBranch'].value_counts(dropna=False).head()


# In[ ]:


df = label_encoder(df, ['OsPlatformSubRelease', 'Census_OSBranch'])
train = label_encoder(train, ['OsPlatformSubRelease', 'Census_OSBranch'])
df[['OsPlatformSubRelease_label_encoded', 'Census_OSBranch_label_encoded']].corr()


# In[ ]:


df[['OsPlatformSubRelease_label_encoded', 'Census_OSBranch_label_encoded']].corr().columns


# Another example of higly correlated features, but this time we have a little difference in the values. OsPlatformSubRelease has only rs3 release whereas Census_OSBranch contains the same information but splitted into two subcategories of rs3 release: rs3_release and rs3_release_svc_escrow.
# 
# Also Census_OSBranch has 32 different values whilst OsPlatformSubRelease has only 9

# In[ ]:


plt.figure(figsize=(14,6))
sns.countplot(x='OsPlatformSubRelease', hue='HasDetections', data=train);


# In[ ]:


os_platform_rs4 = train.loc[train['OsPlatformSubRelease'] == 'rs4', 'HasDetections'].value_counts().sort_index().to_dict()
os_branch_rs4 = train.loc[train['Census_OSBranch'] == 'rs4_release', 'HasDetections'].value_counts().sort_index().to_dict()
os_platform_rs2 = train.loc[train['OsPlatformSubRelease'] == 'rs2', 'HasDetections'].value_counts().sort_index().to_dict()
os_branch_rs2= train.loc[train['Census_OSBranch'] == 'rs2_release', 'HasDetections'].value_counts().sort_index().to_dict()
x = ['OsPlatformSubRelease', 'Census_OSBranch']
trace1 = go.Bar(x=x, y=[os_platform_rs4[0] / sum(os_platform_rs4.values()), os_branch_rs4[0] / sum(os_branch_rs4.values())], name='0 (no detections)', text=[os_platform_rs4[0], os_branch_rs4[0]], textposition="inside")
trace2 = go.Bar(x=x, y=[os_platform_rs4[1] / sum(os_platform_rs4.values()), os_branch_rs4[1] / sum(os_branch_rs4.values())], name='1 (has detections)', text=[os_platform_rs4[1], os_branch_rs4[1]], textposition="inside")
trace3 = go.Bar(x=x, y=[os_platform_rs2[0] / sum(os_platform_rs2.values()), os_branch_rs2[0] / sum(os_branch_rs2.values())], name='0 (no detections)', text=[os_platform_rs2[0], os_branch_rs2[0]], textposition="inside")
trace4 = go.Bar(x=x, y=[os_platform_rs2[1] / sum(os_platform_rs2.values()), os_branch_rs2[1] / sum(os_branch_rs2.values())], name='1 (has detections)', text=[os_platform_rs2[1], os_branch_rs2[1]], textposition="inside")
fig = plotly.tools.make_subplots(rows=1, cols=2, subplot_titles=('rs4', 'rs2'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(title='rs4 and rs2 comparison', font=dict(size=18), barmode='group')
iplot(fig)


# The values and detection rate are almost equal for **rs4** and **rs2** in both of this features.
# 
# But **Census_OSBranch** contains more specified information about **rs3** release, which is split in two subcategories.

# In[ ]:


os_platform_rs3 = train.loc[train['OsPlatformSubRelease'] == 'rs3', 'HasDetections'].value_counts().sort_index().to_dict()
os_branch_rs3 = train.loc[train['Census_OSBranch'] == 'rs3_release', 'HasDetections'].value_counts().sort_index().to_dict()
os_branch_rs3_svc = train.loc[train['Census_OSBranch'] == 'rs3_release_svc_escrow', 'HasDetections'].value_counts().sort_index().to_dict()
trace1 = go.Bar(x=['OsPlatformSubRelease'], y=[os_platform_rs3[0] / sum(os_platform_rs3.values())], name='0 (no detections)', text=os_platform_rs3[0], textposition="inside")
trace2 = go.Bar(x=['OsPlatformSubRelease'], y=[os_platform_rs3[1] / sum(os_platform_rs3.values())], name='1 (has detections)', text=os_platform_rs3[1], textposition="inside")
trace3 = go.Bar(x=['Census_OSBranch'], y=[os_branch_rs3[0] / sum(os_branch_rs3.values())], name='0 (no detections)', text=os_branch_rs3[0], textposition="inside")
trace4 = go.Bar(x=['Census_OSBranch'], y=[os_branch_rs3[1] / sum(os_branch_rs3.values())], name='1 (has detections)', text=os_branch_rs3[1], textposition="inside")
trace5 = go.Bar(x=['Census_OSBranch'], y=[os_branch_rs3_svc[0] / sum(os_branch_rs3_svc.values())], name='0 (no detections)', text=os_branch_rs3_svc[0], textposition="inside")
trace6 = go.Bar(x=['Census_OSBranch'], y=[os_branch_rs3_svc[1] / sum(os_branch_rs3_svc.values())], name='1 (has detections)', text=os_branch_rs3_svc[1], textposition="inside")
fig = plotly.tools.make_subplots(rows=1, cols=3, subplot_titles=('rs3', 'rs3_release', 'rs3_release_svc_escrow'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 1, 3)
fig.append_trace(trace6, 1, 3)
fig['layout'].update(title='rs3 comparison', font=dict(size=18), barmode='group')
iplot(fig)


# <a id="4"></a>
# # OsVer / Platform
# <a id="5"></a>
# ## Correlation

# In[ ]:


df['OsVer'].value_counts(dropna=False).head()


# In[ ]:


df['Platform'].value_counts(dropna=False)


# Numbers are suspiciously similar.
# 
# Lets see distribution of OsVer by Platform

# In[ ]:


for i in df['Platform'].value_counts(dropna=False).index:
    print('Value counts for', i)
    print(df[df['Platform'] == i]['OsVer'].value_counts().head())
    print()


# Distribution of OsVer by Platform is as follows:
# * Windows 10: mostly 10.0.0.0
# * Windows 8: mostly 6.3.0.0
# * Windows 7: mostly 6.1.1.0
# * Windows 2016: only 10.0.0.0

# Making a simple label encoding for this two features

# In[ ]:


df = label_encoder(df, ['Platform', 'OsVer'])


# And printing out a correlation martix for them within the whole dataset

# In[ ]:


df[['Platform_label_encoded', 'OsVer_label_encoded']].corr()


# Correlation of this features in training datasets along with HasDetections

# In[ ]:


train = label_encoder(train, ['Platform', 'OsVer'])
train[['Platform_label_encoded', 'OsVer_label_encoded', 'HasDetections']].corr()


# For now we can see that they are highly correlated

# <a id="6"></a>
# ## Interaction
# Gonna make some plots to see how OS Versions are distributed by Platforms also with respect to a malware detection rate.

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
platforms = train['Platform'].value_counts(dropna=False).index
values_to_show = 3
for i in range(len(platforms)):
    sub_df = train[train['Platform'] == platforms[i]]['OsVer'].value_counts(normalize=True, dropna=False).head(values_to_show)
    sub_df.plot(kind='bar', rot=0, ax=axes[i // 2,i % 2], fontsize=14).set_xlabel('OsVer', fontsize=14);
    for j in range(values_to_show):
        try:
            axes[i // 2,i % 2].plot(j, train.loc[(train['Platform'] == platforms[i]) & (train['OsVer'] == sub_df.index[j]), 'HasDetections'].value_counts(True)[1], marker='.', color="black", markersize=24)
        except:
            continue
    axes[i // 2,i % 2].legend(['Detection rate (%)'])
    axes[i // 2,i % 2].set_title(platforms[i], fontsize=18)


# Even though these two features are highly correlated there is some difference in detection rate when they are interracting with each other. As an example you can see that if users OS is Windows 10 and his OS Version is 10.0.0.0 (which is most likely) then the detection rate for the presented dataset is around 0.5. But if the user has Windows 2016 with the same 10.0.0.0 OS Version then detection rate is lower than 0.4.
# Although the percentage of such cases is very low it might be worth trying to keep both of thease features or maybe create some new features from their interraction.

# <a id="7"></a>
# # Census_OSEdition / Census_OSSkuName

# In[ ]:


df['Census_OSEdition'].value_counts(dropna=False).head()


# In[ ]:


df['Census_OSSkuName'].value_counts(dropna=False).head()


# Not only the numbers but the names of the values looks the same.

# In[ ]:


check_values = 5
fig, axes = plt.subplots(nrows=check_values, ncols=2, figsize=(14,check_values * 4))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
for i in range(check_values):
    os_edition_value = train['Census_OSEdition'].value_counts(dropna=False).index[i]
    os_skuname_value = train['Census_OSSkuName'].value_counts(dropna=False).index[i]
    train.loc[train['Census_OSEdition'] == os_edition_value, 'HasDetections'].value_counts(True).sort_index().plot(kind='bar', rot=0, ax=axes[i, 0]).set_xlabel(os_edition_value, fontsize=16);
    train.loc[train['Census_OSSkuName'] == os_skuname_value, 'HasDetections'].value_counts(True).sort_index().plot(kind='bar', rot=0, ax=axes[i, 1]).set_xlabel(os_skuname_value, fontsize=16);
    axes[i, 0].text(x=-0.175, y=0.4, s=train.loc[train['Census_OSEdition'] == os_edition_value, 'HasDetections'].value_counts()[0], fontsize=18, color='white', fontweight='bold');
    axes[i, 0].text(x=0.825, y=0.4, s=train.loc[train['Census_OSEdition'] == os_edition_value, 'HasDetections'].value_counts()[1], fontsize=18, color='white', fontweight='bold');
    axes[i, 1].text(x=-0.175, y=0.4, s=train.loc[train['Census_OSSkuName'] == os_skuname_value, 'HasDetections'].value_counts()[0], fontsize=18, color='white', fontweight='bold');
    axes[i, 1].text(x=0.825, y=0.4, s=train.loc[train['Census_OSSkuName'] == os_skuname_value, 'HasDetections'].value_counts()[1], fontsize=18, color='white', fontweight='bold');
axes[0, 0].set_title('Census_OSEdition', fontsize=18, fontweight='bold');
axes[0, 1].set_title('Census_OSSkuName', fontsize=18, fontweight='bold');


# And the distribution is almost the same. Again doing label encoding and looking at the correlation.

# In[ ]:


train = label_encoder(train, ['Census_OSEdition', 'Census_OSSkuName'], keep_original=True)


# In[ ]:


train[['HasDetections', 'Census_OSEdition_label_encoded', 'Census_OSSkuName_label_encoded']].corr()


# <a id="8"></a>
# # Census_OSInstallLanguageIdentifier / Census_OSUILocaleIdentifier

# In[ ]:


df['Census_OSInstallLanguageIdentifier'].value_counts().head()


# In[ ]:


df['Census_OSUILocaleIdentifier'].value_counts().head()


# Another case of similar numbers. 

# In[ ]:


train['Census_OSInstallLanguageIdentifier'].fillna(-1, inplace=True)
train[['Census_OSInstallLanguageIdentifier', 'Census_OSUILocaleIdentifier', 'HasDetections']].corr()


# <a id="9"></a>
# # Census_OSBuildNumber / OsBuild

# In[ ]:


train['Census_OSBuildNumber'].value_counts().head()


# In[ ]:


train['OsBuild'].value_counts().head()


# In[ ]:


train[['Census_OSBuildNumber', 'OsBuild', 'HasDetections']].corr()


# In[ ]:


diff = df[df['Census_OSBuildNumber'] != df['OsBuild']].shape[0]
print(diff, 'differences ({:<.3f} %)'.format(diff / df.shape[0]))


# <a id="10"></a>
# # Census_PowerPlatformRoleName / Census_MDC2FormFactor / Census_ChassisTypeName

# In[ ]:


df['Census_PowerPlatformRoleName'].value_counts(dropna=False).head()


# In[ ]:


df['Census_MDC2FormFactor'].value_counts(dropna=False).head()


# In[ ]:


df['Census_ChassisTypeName'].value_counts(dropna=False).head()


# Once again numbers and names look similar

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,10))
sns.countplot(x='Census_PowerPlatformRoleName', hue='HasDetections', data=train, ax=axes[0], order=train['Census_PowerPlatformRoleName'].value_counts(dropna=False).index.tolist());
sns.countplot(x='Census_MDC2FormFactor', hue='HasDetections', data=train, ax=axes[1], order=train['Census_MDC2FormFactor'].value_counts(dropna=False).index.tolist());


# In[ ]:


train = label_encoder(train, ['Census_PowerPlatformRoleName', 'Census_MDC2FormFactor', 'Census_ChassisTypeName'], keep_original=True)


# In[ ]:


train[['HasDetections', 'Census_PowerPlatformRoleName_label_encoded', 'Census_MDC2FormFactor_label_encoded', 'Census_ChassisTypeName_label_encoded']].corr()


# Though correlation of this 3 features is not that big
