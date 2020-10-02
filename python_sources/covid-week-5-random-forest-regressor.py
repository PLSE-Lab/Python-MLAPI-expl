#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as po
import plotly.express as px
import plotly.graph_objs as go
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')


# In[ ]:


def aboutdata(dataframe):
    df=dataframe
    print('About datatypes of columns and memory usage:')
    df.info()
    print('Shape of data frame:')
    print(df.shape)
    for col in df.columns:
        print("Unique number of values in ")
        print(col)
        print(df.loc[:,col].nunique())
    print("number of null values present in each column")
    print(df.isnull().sum())
    print(df.head(5))


# In[ ]:


aboutdata(train)


# In[ ]:


aboutdata(test)


# In[ ]:


aboutdata(submission)


# In[ ]:


#train=train.drop(['County','Province_State','Country_Region'],axis=1)


# In[ ]:


train=train.drop(['County','Province_State'],axis=1)


# In[ ]:


#test=test.drop(['County','Province_State','Country_Region'],axis=1)


# In[ ]:


test=test.drop(['County','Province_State'],axis=1)


# In[ ]:


y=train.TargetValue  


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


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X = train['Country_Region'].values
train['Country_Region'] = labelencoder.fit_transform(X.astype(str))


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X = test['Country_Region'].values
test['Country_Region'] = labelencoder.fit_transform(X.astype(str))


# In[ ]:


#features=[ 'Population', 'Weight','Date']


# In[ ]:


features=[ 'Population','Weight','Country_Region','pw','Date']


# In[ ]:


train['pw']=train['Population']/train['Weight']
test['pw']=test['Population']/test['Weight']


# In[ ]:


X=train[features]


# In[ ]:


from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y =train_test_split(X,y,test_size=0.3,random_state=7)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


"""from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline_dt = Pipeline([('scaler2' , StandardScaler()),
                        ('RandomForestRegressor: ', RandomForestRegressor())])
pipeline_dt.fit(train_X , train_y)"""


# In[ ]:


my_model=RandomForestRegressor(n_estimators=100, n_jobs=-1)


# In[ ]:


my_model.fit(train_X,train_y)


# In[ ]:


predictions=my_model.predict(val_X)


# In[ ]:


#predictions=pipeline_dt.predict(val_X)


# In[ ]:


predictions


# In[ ]:


from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(predictions,val_y)


# In[ ]:


val_mae


# In[ ]:


test_X=test[features]
test_preds=my_model.predict(test_X)


# In[ ]:


test_preds


# In[ ]:


pred_list =[int(x) for x in test_preds]


# In[ ]:


output = pd.DataFrame({'Id': test.index,
                      'TargetValue': pred_list})
#output.to_csv('submission.csv', index=False)


# In[ ]:


a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()


# In[ ]:


a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05'].clip(0,10000)
a['q0.5']=a['q0.5'].clip(0,10000)
a['q0.95']=a['q0.95'].clip(0,10000)


# In[ ]:


a['Id'] =a['Id']+ 1


# In[ ]:


sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)


# In[ ]:


sub

