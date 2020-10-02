#!/usr/bin/env python
# coding: utf-8

# ## Geotab Intersection Congestion
# 
# The data consists of aggregated trip logging metrics from commercial vehicles, such as semi-trucks. The data have been grouped by intersection, month, hour of day, direction driven through the intersection, and whether the day was on a weekend or not.
# 
# For each grouping in the test set, you need to make predictions for three different quantiles of two different metrics covering how long it took the group of vehicles to drive through the intersection. Specifically, the 20th, 50th, and 80th percentiles for the total time stopped at an intersection and the distance between the intersection and the first place a vehicle stopped while waiting. You can think of your goal as summarizing the distribution of wait times and stop distances at each intersection.
# 
# To see full data description go to this link: https://www.kaggle.com/c/bigquery-geotab-intersection-congestion/data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as train_valid_split
from sklearn.multioutput import MultiOutputRegressor,RegressorChain

from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_squared_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')
test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')
train.info()


# ## Target Variables
# 
# Based from the competition description we have 6 target variables. So we will do an analysis of that first.

# In[ ]:


target_cols = ['TotalTimeStopped_p20','TotalTimeStopped_p50', 'TotalTimeStopped_p80',
               'DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']
train[target_cols].head()


# In[ ]:


nrow=3
ncol=2
fig, axes = plt.subplots(nrow, ncol,figsize=(20,10))
count=0
for r in range(nrow):
    for c in range(ncol):
        if(count==len(target_cols)):
            break
        col = target_cols[count]
        
        axes[r,c].hist(np.log1p(train[col]),bins=100)
        axes[r,c].set_title('log1p( '+str(col)+' )',fontsize=15)
        count = count+1

plt.show()


# ## Feature Columns
# 
# Since not all columns in the trainset are present in the testset we will only be consider 13 colums

# In[ ]:


feature_cols = ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName',
                'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend',
                'Month','Path','City']
train[feature_cols].head()


# In[ ]:


train[feature_cols].describe(include='all').T


# ## eXtreme Gradient Boosting Model

# In[ ]:


def category_mapping(df,map_dict):
    for col in map_dict.keys():
        df[col] = df[col].map(map_dict[col])
        df[col] = df[col].fillna(0).astype(np.int16)
    return df


# In[ ]:


cat_maps = {}
cat_cols = ['EntryStreetName','ExitStreetName','EntryHeading','ExitHeading','Path','City']
for col in cat_cols:
    values = list(train[col].unique())+list(test[col].unique())
    LE = LabelEncoder().fit(values)
    cat_maps[col] = dict(zip(LE.classes_, LE.transform(LE.classes_)))
    
train = category_mapping(train,cat_maps)


# In[ ]:


non_feature_cols = ['RowId','IntersectionId','TimeFromFirstStop_p20',
                    'TimeFromFirstStop_p40','TimeFromFirstStop_p50',
                    'TimeFromFirstStop_p60','TimeFromFirstStop_p80',
                    'DistanceToFirstStop_p40', 'DistanceToFirstStop_p60',
                    'TotalTimeStopped_p40','TotalTimeStopped_p60']
feature_cols = set(train.columns)-set(non_feature_cols+target_cols)

ys = train[target_cols]
Xs = train[feature_cols]
del train
X_train,X_valid, y_train, y_valid = train_valid_split(Xs, ys, test_size = .05, random_state=0)
X_train.shape,y_train.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "multioutput_model = MultiOutputRegressor(XGBRegressor(random_state=0,n_jobs=-1,n_estimators=1000,\n                                                      objective='reg:squarederror',max_depth=10,\n                                                      tree_method='gpu_hist', predictor='gpu_predictor'))\nmultioutput_model.fit(X_train,y_train);")


# In[ ]:


preds = multioutput_model.predict(X_train)
print('Training R2: ',r2_score(y_train,preds),
      ' MSE: ',mean_squared_error(y_train,preds))
preds = multioutput_model.predict(X_valid)
print('Validation R2: ',r2_score(y_valid,preds),
      ' MSE: ',mean_squared_error(y_valid,preds))


# In[ ]:


nrow=3
ncol=2
fig, axes = plt.subplots(nrow, ncol,
                         sharex=True,
                         sharey=True,
                         figsize=(15,10))
count=0
for r in range(nrow):
    for c in range(ncol):
        if(count==len(target_cols)):
            break
        col = target_cols[count]
        
        axes[r,c].hist(np.log1p(preds[:,count]),bins=100)
        axes[r,c].set_title('log1p( '+str(col)+' )',fontsize=15)
        count = count+1

plt.show()


# ## Submission Pipeline

# In[ ]:


test = test[feature_cols]
test = category_mapping(test,cat_maps)
test.head()


# In[ ]:


preds = multioutput_model.predict(test)


# In[ ]:


nrow=3
ncol=2
fig, axes = plt.subplots(nrow, ncol,
                         sharex=True,
                         sharey=True,
                         figsize=(15,10))
count=0
for r in range(nrow):
    for c in range(ncol):
        if(count==len(target_cols)):
            break
        col = target_cols[count]
        
        axes[r,c].hist(np.log1p(preds[:,count]),bins=100)
        axes[r,c].set_title('log1p( '+str(col)+' )',fontsize=15)
        count = count+1

plt.show()


# In[ ]:


TargetIds = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/sample_submission.csv')['TargetId'].values

sub_df = pd.DataFrame()
sub_df['Target'] = list(preds)
sub_df = sub_df.explode('Target')

sub_df['TargetId'] = TargetIds
sub_df = sub_df[['TargetId','Target']]
sub_df.to_csv('the-sub-mission.csv',index=False)
sub_df.head(10)


# ### Do UPVOTE if this notebook is helpful to you in some way :) <br/> Comment below any suggetions that can help improve this notebook. TIA
