#!/usr/bin/env python
# coding: utf-8

# Assalamu Alaikum, Covid-19 the world pandemic is now hot topic of the world. Let's do some work with this virus current circumstance by Kaggle given dataset. Our work is most easiest one. The result may differ because the leaderboard is calculated with approximately 28% of the test data and final on 72% test data

# So let's go

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# First import some library.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
import datetime as dt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Now import the dataset, you find the dataset pathway in rigth side of of kaggle notebook in (input) box

# In[ ]:


train=pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
test=pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
submission=pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')


# Let's see what are columns have our dataset

# In[ ]:


train.columns


# In[ ]:


test.columns


# In dataset we see many of the countries have no Province/State so we can fill the (nan) values by 'No province', because when we do the catagorical encoding it will count as a catagory.
# If you want see how many unique type of province uncomment the below code

# In[ ]:


#train['Province/State'].unique()


# In[ ]:


train['Province/State'].fillna('No Province',inplace=True)


# In[ ]:


test['Province/State'].fillna('No Province',inplace=True)


# Ok, now we encode the 

# In[ ]:


ohe=OneHotEncoder(handle_unknown='ignore')


# Here we transform the normal date to pandas datetime

# In[ ]:


train['Date']= pd.to_datetime(train['Date']) 
test['Date']= pd.to_datetime(test['Date']) 


# Individual second,minutes,hours,date help the algorithm predict good

# In[ ]:


def create_time_features(df):
    df['date'] = df['Date']
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    return X


# Let's put the dataset on the function for converting

# In[ ]:


create_time_features(train)
create_time_features(test)


# So, the normal date value no more need, now we can drop this features

# In[ ]:


train=train.drop(columns=['Date'],axis=1)
test=test.drop(columns=['Date'],axis=1)


# The (create_time_features) function create another date , we also drop that

# In[ ]:


train=train.drop(columns=['date'],axis=1)
test=test.drop(columns=['date'],axis=1)


# In[ ]:


train.head(3)


# Now, we detect the object type data in our dataset have and convert it to catagorical features.
# First,we get the objects in object_cols

# In[ ]:


s = (train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# Pandas has it own built in get_dummies method for catagorical encoding

# We encode both the Province/State and Country/Region

# In[ ]:


train_dummies=pd.get_dummies(train['Country/Region'])


# In[ ]:


test_dummies=pd.get_dummies(test['Country/Region'])


# In[ ]:


train1_dummies=pd.get_dummies(train['Province/State'])
test1_dummies=pd.get_dummies(test['Province/State'])


# Drop the dataset Province and Country , we no more need it.

# In[ ]:


train=train.drop(['Country/Region','Province/State','Id'],axis=1)
test=test.drop(['Country/Region','Province/State','ForecastId'],axis=1)


# Concate the main dataset and dummies features

# In[ ]:


train=pd.concat([train,train_dummies,train1_dummies],axis=1)


# In[ ]:


test=pd.concat([test,test_dummies,test1_dummies],axis=1)


# The dataset may like horrible, because of the dummies variable, if you have not idea about this, please search in the google and clear the concep

# In[ ]:


train.head(3)


# Separate the features and target, we have two target one for ConfirmedCases and another Fatalities, so the target also two target1 and target2

# In[ ]:


features=train.drop(['ConfirmedCases','Fatalities'],axis=1)
target1=train['ConfirmedCases']
target2=train['Fatalities']


# Here we use the DecisonTreeRregressor as our algorithm, which is very much good for catagorical prediction, we use Regressor based algorithm because it continious value prediction problem

# In[ ]:


rf=DecisionTreeRegressor(criterion='mse', splitter='best')


# Let's see the shape of the features and target

# In[ ]:


print(features.shape)
print(target1.shape)
print(test.shape)


# Train the model by fit method

# In[ ]:


rf.fit(features,target1)


# And get the prediction by predict method.
# 

# In[ ]:


predict_cases=rf.predict(test)


# Also we input our prediction result in submission file

# In[ ]:


submission['ConfirmedCases']=predict_cases


# We now want to predict the target2(Fatalities) by ConfirmedCases because we get the prediction of ConfirmedCases from our prediction so concate and make anorh

# Train the model,predict and input the result in submission file for another target(target2)

# In[ ]:


rf.fit(features,target2)
predict_fatalities=rf.predict(test)
submission['Fatalities']=predict_fatalities


# The submissio result comes with float number so we round it and convert to int values

# In[ ]:


submission.round().astype(int)


# In[ ]:


submission.head(3)


# Now get the submission file by to_csv

# In[ ]:


submission.to_csv('submission.csv',index=False)

