#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
df_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")


# In[ ]:


df_train.info()


# # **EDA for Global *Cases* and *Fatalities* for max affected countries**

# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])


# In[ ]:


Global_Top_Cases = df_train[df_train['Date'] == df_train['Date'].max()].groupby(['Country_Region','Date'])


# In[ ]:


G_cases = Global_Top_Cases.sum().sort_values(['ConfirmedCases'], ascending = False)


# In[ ]:


G_cases[:10]


# In[ ]:


G1 = G_cases.drop(['Id'], axis = 1)


# In[ ]:


G1[:10]


# In[ ]:


corrmat = df_train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[ ]:


G1[:10].plot(kind = 'bar')


# # ***Percent change* of Confirmed Cases and Fatalities of max affected Countries and affect of Mitigation like social distancing, shelter in place , Closed down et al.**

# In[ ]:


G1['ConfirmedCases_percent_change'] = G1['ConfirmedCases'].pct_change()
G1['Fatalities_percent_change'] = G1['Fatalities'].pct_change()


# In[ ]:


G2 = G1.drop(['ConfirmedCases', 'Fatalities'], axis = 1)


# In[ ]:


G2[:10].plot(kind = 'line')

x = plt.gca().xaxis

for item in x.get_ticklabels():
    item.set_rotation(70)


# # ***EDA* for *US* Confirmed Cases and Fatalites by most affected *States***

# In[ ]:


us = df_train[df_train['Country_Region'] == 'US'].groupby(['Province_State']).max()


# In[ ]:


us_cases = us.sort_values(['ConfirmedCases'], ascending = False)


# In[ ]:


us_cases.head()


# In[ ]:


us_cases = us_cases.drop(['Id', 'Date','Country_Region'], axis = 1)


# In[ ]:


us_cases[:10].plot(kind = 'bar')


# # ***Percent change *of Confirmed Cases and Fatalities of max affected US states and affect of *Mitigation* - Social Distancing, Shelter in place , Closed down et al.**

# In[ ]:


us_cases['ConfirmedCases_percent_change'] = us_cases['ConfirmedCases'].pct_change()
us_cases['Fatalities_percent_change'] = us_cases['Fatalities'].pct_change()


# In[ ]:


us_cases = us_cases.drop(['ConfirmedCases', 'Fatalities'], axis = 1)


# In[ ]:


us_cases[:10].plot(kind = 'line')
x = plt.gca().xaxis
for item in x.get_ticklabels():
    item.set_rotation(70)


# # ***Forecasting* Cases and Fatalities**

# In[ ]:





# In[ ]:


df_train['Province_State'].fillna("", inplace = True)
df_test['Province_State'].fillna("", inplace = True)


# In[ ]:


df_train['Country_Region'] = df_train['Country_Region'] + ' ' + df_train['Province_State']
df_test['Country_Region'] = df_test['Country_Region'] + ' ' + df_test['Province_State']


# In[ ]:


df_train['Date'] = list(df_train['Date'].dt.strftime('%Y,%m,%d'))
df_test['Date'] = list(df_test['Date'].dt.strftime('%Y,%m,%d'))


# In[ ]:



def createDateFields(df):
    year = []
    month = []
    day = []
    for i in df['Date']:
        i = i.split(",")
    
        year.append(i[0])
        month.append(i[1])
        day.append(i[2])
    df['year'] = year
    df['month'] = month
    df['day'] = day
    
    return df   
  


# In[ ]:



df_train = createDateFields(df_train)
df_test = createDateFields(df_test)


# In[ ]:


df_train.tail()


# In[ ]:


df_test.tail()


# In[ ]:


df_train['ConfirmedCases']= df_train['ConfirmedCases'].apply(int)
df_train['Fatalities'] = df_train['Fatalities'].apply(int)


# In[ ]:


y1 = df_train.ConfirmedCases
y2 = df_train.Fatalities

del df_train['ConfirmedCases']
del df_train['Fatalities']


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


lable_encoder = LabelEncoder()
df_train['Country_Region'] = lable_encoder.fit_transform(df_train['Country_Region'])
df_test['Country_Region'] = lable_encoder.transform(df_test['Country_Region'])


# In[ ]:


features = ['Country_Region', 'month', 'day']


# In[ ]:


X = df_train[features]


# In[ ]:


test_features = ['Country_Region', 'month', 'day']


# In[ ]:


X_test = df_test[test_features]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X.values)
X_test = min_max_scaler.transform(X_test.values)


# In[ ]:


import xgboost as xgb


# In[ ]:


xg_reg_model1 = xgb.XGBRegressor(n_estimators = 1500,random_state = 0, max_depth = 15, learning_rate = 0.1)


# In[ ]:


xg_reg_model1.fit(X,y1);


# In[ ]:


y_pred1 = xg_reg_model1.predict(X_test)


# In[ ]:


#y_pred1


# In[ ]:


xg_reg_model2 = xgb.XGBRegressor (n_estimators = 1500,random_state =0, max_depth = 15, learning_rate = 0.1)
xg_reg_model2.fit(X,y2);


# In[ ]:


y_pred2 = xg_reg_model2.predict(X_test)


# In[ ]:


#y_pred2


# In[ ]:


y_pred1 , y_pred2 = np.round(y_pred1),np.round(y_pred2)

y_pred1[y_pred1 < 0] = 0
y_pred2[y_pred2 < 0] = 0

df_submission['ConfirmedCases'] = y_pred1.astype(int)
df_submission['Fatalities'] = y_pred2.astype(int)


# In[ ]:


df_submission.head()


# In[ ]:


df_submission.to_csv("submission.csv",index = False)


# In[ ]:




