#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import GridSearchCV, KFold

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
sample = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')


# In[ ]:


train = train.drop(['County','Province_State','Country_Region','Target'],axis=1)
test = test.drop(['County','Province_State','Country_Region','Target'],axis=1)
train


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df


# In[ ]:


def train_dev_split(df, days):
    #Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]


# In[ ]:


test_date_min = test['Date'].min()
test_date_max = test['Date'].max()


# In[ ]:


def avoid_data_leakage(df, date=test_date_min):
    return df[df['Date']<date]


# In[ ]:


def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day


# In[ ]:


train['Date']=pd.to_datetime(train['Date'])
test['Date']=pd.to_datetime(test['Date'])


# In[ ]:


test['Date']=test['Date'].dt.strftime("%Y%m%d").astype(int)
train['Date']=train['Date'].dt.strftime("%Y%m%d").astype(int)


# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['TargetValue', 'Id'], axis=1)
target = train["TargetValue"]
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state = 0)


# In[ ]:


test = test.drop(['ForecastId'], axis=1)


# In[ ]:


model = XGBRegressor(n_estimators = 1000 , random_state = 0)
model.fit(X_train, y_train)

scores = []

scores.append(model.score(X_test, y_test))

y_pred2 = model.predict(X_test)
y_pred2

predictions = model.predict(test)

pred_list = [x for x in predictions]

output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})
print(output)


# In[ ]:


a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()


# In[ ]:


a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05']
a['q0.5']=a['q0.5']
a['q0.95']=a['q0.95']
a


# In[ ]:


a['Id'] =a['Id']+ 1
a


# In[ ]:


sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()

