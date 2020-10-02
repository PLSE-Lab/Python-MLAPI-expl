#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime

import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# In[ ]:


########################### Vars
#################################################################################
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')


# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')
train_df = pd.read_pickle('../input/ieee-data-minification/train_transaction.pkl')
train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 
test_df = train_df[train_df['DT_M']==train_df['DT_M'].max()].reset_index(drop=True)
train_df = train_df[train_df['DT_M']<(train_df['DT_M'].max()-1)].reset_index(drop=True)
del train_df['DT_M'], test_df['DT_M']
print('Shape control:', train_df.shape, test_df.shape)


# In[ ]:


########################### Create day agg
for df in [train_df, test_df]:
    
    # Temporary variables for aggregation
    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)
    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear
    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month


# In[ ]:


########################### D Columns
i_cols = ['D15']
periods = ['DT_D']

temp_df = pd.concat([train_df[['TransactionDT']+i_cols+periods], test_df[['TransactionDT']+i_cols+periods]])
for period in periods:
    for col in i_cols:
        for df in [temp_df]:
            df.set_index(period)[col].plot(style='.', title=col, figsize=(15, 3))
            plt.show()


# In[ ]:


# We can see that maximum value is growing
# Possible explanation that it is "comulative" feature
# In this case minimal value will be same
# But maxmium and mean will increase
# Lets check destributions


# In[ ]:


plt.figure(figsize=(16, 6))

df = train_df[['D15','isFraud']].dropna()
sns.distplot(df['D15'], color = 'skyblue', kde= True, label = 'Train')

df = test_df[['D15','isFraud']].dropna()
sns.distplot(df['D15'], color = 'red', kde= True, label = 'Test')


# In[ ]:


# Can't see difference
# log transform?
plt.figure(figsize=(16, 6))

df = train_df[['D15','isFraud']].dropna().astype(float)
df = df[df['D15']>0]
df['D15'] = np.log10(df['D15'])
sns.distplot(df['D15'], color = 'red', kde= True , label = 'Train')

df = test_df[['D15','isFraud']].dropna().astype(float)
df = df[df['D15']>0]
df['D15'] = np.log10(df['D15'])
sns.distplot(df['D15'], color = 'skyblue', kde= True, label = 'Test')


# In[ ]:


# We can do min-max scaling by day or week
def values_normalization(dt_df, periods, columns):
    for period in periods:
        for col in columns:
            new_col = col +'_'+ period
            dt_df[col] = dt_df[col].astype(float)  

            temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
            temp_min.index = temp_min[period].values
            temp_min = temp_min['min'].to_dict()

            temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
            temp_max.index = temp_max[period].values
            temp_max = temp_max['max'].to_dict()

            dt_df['temp_min'] = dt_df[period].map(temp_min)
            dt_df['temp_max'] = dt_df[period].map(temp_max)

            dt_df[new_col+'_min_max'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])

            del dt_df['temp_min'],dt_df['temp_max']
    return dt_df


# By Week
periods = ['DT_W']
new_col = 'D15_DT_W_min_max'
plt.figure(figsize=(16, 6))

df = train_df[['D15','isFraud','DT_W']].dropna().astype(float)
df = values_normalization(df, periods, ['D15'])
df = df[df[new_col]>0]
df[new_col] = np.log10(df[new_col])
sns.distplot(df[new_col], color = 'red', kde= True , label = 'Train')

df = test_df[['D15','isFraud','DT_W']].dropna().astype(float)
df = values_normalization(df, periods, ['D15'])
df = df[df[new_col]>0]
df[new_col] = np.log10(df[new_col])
sns.distplot(df[new_col], color = 'skyblue', kde= True , label = 'Test')


# In[ ]:


# By Month

periods = ['DT_M']
new_col = 'D15_DT_M_min_max'
plt.figure(figsize=(16, 6))

df = train_df[['D15','isFraud','DT_M']].dropna().astype(float)
df = values_normalization(df, periods, ['D15'])
df = df[df[new_col]>0]
df[new_col] = np.log10(df[new_col])
sns.distplot(df[new_col], color = 'red', kde= True , label = 'Train')

df = test_df[['D15','isFraud','DT_M']].dropna().astype(float)
df = values_normalization(df, periods, ['D15'])
df = df[df[new_col]>0]
df[new_col] = np.log10(df[new_col])
sns.distplot(df[new_col], color = 'skyblue', kde= True , label = 'Test')


# In[ ]:


# By day
periods = ['DT_D']
new_col = 'D15_DT_D_min_max'
plt.figure(figsize=(16, 6))

df = train_df[['D15','isFraud','DT_D']].dropna().astype(float)
df = values_normalization(df, periods, ['D15'])
df = df[df[new_col]>0]
df[new_col] = np.log10(df[new_col])
sns.distplot(df[new_col], color = 'red', kde= True , label = 'Train')

df = test_df[['D15','isFraud','DT_D']].dropna().astype(float)
df = values_normalization(df, periods, ['D15'])
df = df[df[new_col]>0]
df[new_col] = np.log10(df[new_col])
sns.distplot(df[new_col], color = 'skyblue', kde= True , label = 'Test')


# In[ ]:


########################### Quantile cut
i_cols = ['D15']
periods = ['DT_D']
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
plt.figure(figsize=(16, 6))

temp_df = pd.concat([train_df[['TransactionDT']+i_cols+periods], test_df[['TransactionDT']+i_cols+periods]])
for period in periods:
    for col in i_cols:
        for df in [temp_df]:
            df[col+'_sc'] = df[col].transform(lambda x: x.rolling(10000, 1000).quantile(.99)).fillna(0)
            ax = sns.scatterplot(x="DT_D", y="D15",palette=cmap,data=df)
            ax = sns.scatterplot(x="DT_D", y="D15_sc",palette=cmap,data=df)

