#!/usr/bin/env python
# coding: utf-8

# # LTFS Data Science FinHack 2
# 
# LTFS receives a lot of requests for its various finance offerings that include housing loan, two-wheeler loan, real estate financing and micro loans. The number of applications received is something that varies a lot with season. Going through these applications is a manual process and is tedious. Accurately forecasting the number of cases received can help with resource and manpower management resulting into quick response on applications and more efficient processing.
# 
# ## Problem Statement
# You have been appointed with the task of forecasting daily cases for next 3 months for 2 different business segments aggregated at the country level keeping in consideration the following major Indian festivals (inclusive but not exhaustive list): Diwali, Dussehra, Ganesh Chaturthi, Navratri, Holi etc. (You are free to use any publicly available open source external datasets). Some other examples could be:
# 
# Weather Macroeconomic variables Note that the external dataset must belong to a reliable source.
# 
# Data Dictionary The train data has been provided in the following way:
# 
# * For business segment 1, historical data has been made available at branch ID level For business segment 2, historical data has been made available at State level.
# 
# Train File Variable Definition application_date Date of application segment Business Segment (1/2) branch_id Anonymised id for branch at which application was received state State in which application was received (Karnataka, MP etc.) zone Zone of state in which application was received (Central, East etc.) case_count (Target) Number of cases/applications received
# 
# Test File Forecasting needs to be done at country level for the dates provided in test set for each segment.
# 
# Variable Definition id Unique id for each sample in test set application_date Date of application segment Business Segment (1/2)
# 
# ### Evaluation
# Evaluation Metric The evaluation metric for scoring the forecasts is **MAPE (Mean Absolute Percentage Error)* M with the formula:
# 
# 
# Where At is the actual value and Ft is the forecast value.
# 
# The Final score is calculated using MAPE for both the segments using the formula:

# **Reference taken from this github link**
# 
# https://github.com/rajat5ranjan/AV-LTFS-Data-Science-FinHack-2

# In[ ]:


import numpy as np
import pandas as pd
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,GroupKFold


# In[ ]:


train=pd.read_csv('../input/ltfs-2/train_fwYjLYX.csv')
test=pd.read_csv('../input/ltfs-2/test_1eLl9Yf.csv')


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


train.head()


# In[ ]:


train['application_date']=pd.to_datetime(train['application_date'])
test['application_date']=pd.to_datetime(test['application_date'])


# In[ ]:


import holidays
hol_list = holidays.IND(years = [2017,2018,2019])
hol_list = [date for date,name in hol_list.items()]
train['hol'] = train['application_date'].isin(hol_list) * 1
test['hol'] = test['application_date'].isin(hol_list) * 1


# In[ ]:


def dateFeatures(df, label=None,seg=None):
    features = ['day','week','dayofweek','month','quarter','year','dayofyear','weekofyear','is_month_start','is_month_end','is_quarter_start','is_quarter_end','is_year_start','is_year_end']
    date = df['application_date']
    for col in features:
        df[col] = getattr(date.dt,col) * 1


# Based on the segment, We will split dataset into two.

# In[ ]:


train = train[['application_date','segment','case_count']]
train_s1=train[train['segment']==1].groupby(['application_date']).sum().reset_index().sort_values('application_date')
train_s2=train[train['segment']==2].groupby(['application_date']).sum().reset_index().sort_values('application_date')
test_s1=test[test['segment']==1][['application_date']].sort_values('application_date')
test_s2=test[test['segment']==2][['application_date']].sort_values('application_date')


# In[ ]:


dateFeatures(train_s1)
dateFeatures(train_s2)
dateFeatures(test_s1)
dateFeatures(test_s2)


# In[ ]:


test_s2.head()


# Outlier detection

# In[ ]:


sns.boxplot(train_s1['case_count'])


# In[ ]:


sns.distplot(train_s1['case_count'])


# In[ ]:


train_s1['case_count'].describe()


# Case counts are rightly skewed let's analyse the maximum value data

# In[ ]:


case_max = train_s1['case_count'].max()
train_s1[train_s1['case_count']==case_max]


# Observation : 
# Maximum case count recorded date was 1 year earlier than the given data
# It was a Saturday
# The count was higher might be either employee has to finish their target or simulated value. So, Let's analyse the same day on previous and next year

# In[ ]:


train_s1[(train_s1['application_date'] >= '2017-03-01') & (train_s1['application_date'] <= '2017-03-31')]


# In[ ]:


train_s1[(train_s1['application_date'] >= '2018-03-01') & (train_s1['application_date'] <= '2018-03-31')]


# In[ ]:


train_s1[(train_s1['application_date'] >= '2019-03-01') & (train_s1['application_date'] <= '2019-03-31')]


# It seems that value was simulated one. So, lets review some of higher case count values

# In[ ]:


train_s1[train_s1['case_count'] > 7000]


# In[ ]:


train_s1[train_s1['case_count']<20]


# In[ ]:


train_s1 = train_s1[(train_s1['case_count'] > 20) & (train_s1['case_count'] < 7000)]


# In[ ]:


train_s1 = train_s1[train_s1['case_count']<=10000]
train_s1=train_s1.reset_index().drop('index',axis=1)


# Outlier detection

# In[ ]:


train_s2.describe()


# In[ ]:


sns.distplot(train_s2['case_count'])


# In[ ]:


sns.boxplot(train_s2['case_count'])


# In[ ]:


train_s2[train_s2['case_count']>35000]


# In[ ]:


train_s2 = train_s2[train_s2['case_count']<36000]
train_s2=train_s2.reset_index().drop('index',axis=1)


# In[ ]:


y1 = train_s1['case_count']
y2 = train_s2['case_count']
train_s1.drop(['case_count','segment','application_date'],axis=1,inplace=True)
train_s2.drop(['case_count','segment','application_date'],axis=1,inplace=True)


# In[ ]:


test_s1.drop(['application_date'],axis=1,inplace=True)
test_s2.drop(['application_date'],axis=1,inplace=True)


# In[ ]:


kf=GroupKFold(n_splits=20)
s1models = []
s2models = []

X = train_s1
y = y1
loss = []

print("loss:")
grp = train_s1['day'].values
for train_index, test_index in kf.split(X,y,grp):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model=RandomForestRegressor(n_estimators = 150 ,random_state=42,max_features =8)
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    print(mean_absolute_percentage_error(y_test,preds))
    loss.append(mean_absolute_percentage_error(y_test,preds))
    s1models.append(model.predict(test_s1))


# In[ ]:


s1models = s1models[1:19]


# In[ ]:


X = train_s2
y = y2
loss = []
print("loss : ")
grp = train_s2['dayofyear'].values
for train_index, test_index in kf.split(X,y,grp):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model=RandomForestRegressor(n_estimators=150,random_state=42,max_features=8)
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    print(mean_absolute_percentage_error(y_test,preds))
    loss.append(mean_absolute_percentage_error(y_test,preds))
    s2models.append(model.predict(test_s2))


# In[ ]:


del s2models[2]


# In[ ]:


test.loc[test.segment==1, 'case_count']=np.mean(s1models,0)
test.loc[test.segment==2, 'case_count']=np.mean(s2models,0)


# In[ ]:


test.to_csv('submission.csv',index=False) 

