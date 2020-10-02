#!/usr/bin/env python
# coding: utf-8

# # Acknowledgements
# 
# - Built-upon https://www.kaggle.com/ranjithks/19-lines-of-code-result-better-score/notebook
# - Modifications: added LightGBM, blended XGBoost & LightGBM predictions and tweaked hyperparameters

# In[ ]:


import numpy as np 
import pandas as pd 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


PATH_WEEK2='/kaggle/input/covid19-global-forecasting-week-2'
df_train = pd.read_csv(f'{PATH_WEEK2}/train.csv')
df_test = pd.read_csv(f'{PATH_WEEK2}/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.rename(columns={'Country_Region':'Country'}, inplace=True)
df_test.rename(columns={'Country_Region':'Country'}, inplace=True)

df_train.rename(columns={'Province_State':'State'}, inplace=True)
df_test.rename(columns={'Province_State':'State'}, inplace=True)


# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


y1_Train = df_train.iloc[:, -2]
y1_Train.head()


# In[ ]:


y2_Train = df_train.iloc[:, -1]
y2_Train.head()


# In[ ]:


EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state


# In[ ]:


#X_Train = df_train.loc[:, ['State', 'Country', 'Date']]
X_Train = df_train.copy()

X_Train['State'].fillna(EMPTY_VAL, inplace=True)
X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")
X_Train["Date"]  = X_Train["Date"].astype(int)

X_Train.head()


# In[ ]:


#X_Test = df_test.loc[:, ['State', 'Country', 'Date']]
X_Test = df_test.copy()

X_Test['State'].fillna(EMPTY_VAL, inplace=True)
X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")
X_Test["Date"]  = X_Test["Date"].astype(int)

X_Test.head()


# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()


# In[ ]:


X_Train.Country = le.fit_transform(X_Train.Country)
X_Train['State'] = le.fit_transform(X_Train['State'])

X_Train.head()


# In[ ]:


X_Test.Country = le.fit_transform(X_Test.Country)
X_Test['State'] = le.fit_transform(X_Test['State'])

X_Test.head()


# In[ ]:


df_train.head()


# In[ ]:


df_train.loc[df_train.Country == 'Afghanistan', :]


# In[ ]:


df_test.tail()


# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:


le = preprocessing.LabelEncoder()


# In[ ]:


from xgboost import XGBRegressor
import lightgbm as lgb


# In[ ]:


countries = X_Train.Country.unique()

df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
df_out2 = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in countries:
    states = X_Train.loc[X_Train.Country == country, :].State.unique()
    #print(country, states)
    # check whether string is nan or not
    for state in states:
        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]
        
        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']
        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']
        
        X_Train_CS = X_Train_CS.loc[:, ['State', 'Country', 'Date']]
        
        X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)
        X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])
        
        X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), ['State', 'Country', 'Date', 'ForecastId']]
        
        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']
        X_Test_CS = X_Test_CS.loc[:, ['State', 'Country', 'Date']]
        
        X_Test_CS.Country = le.fit_transform(X_Test_CS.Country)
        X_Test_CS['State'] = le.fit_transform(X_Test_CS['State'])
        
        # XGBoost
        model1 = XGBRegressor(n_estimators=2000)
        model1.fit(X_Train_CS, y1_Train_CS)
        y1_pred = model1.predict(X_Test_CS)
        
        model2 = XGBRegressor(n_estimators=2000)
        model2.fit(X_Train_CS, y2_Train_CS)
        y2_pred = model2.predict(X_Test_CS)
        
        # LightGBM
        model3 = lgb.LGBMRegressor(n_estimators=2000)
        model3.fit(X_Train_CS, y1_Train_CS)
        y3_pred = model3.predict(X_Test_CS)
        
        model4 = lgb.LGBMRegressor(n_estimators=2000)
        model4.fit(X_Train_CS, y2_Train_CS)
        y4_pred = model4.predict(X_Test_CS)
        
        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})
        df2 = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y3_pred, 'Fatalities': y4_pred})
        df_out = pd.concat([df_out, df], axis=0)
        df_out2 = pd.concat([df_out2, df2], axis=0)
    # Done for state loop
# Done for country Loop


# In[ ]:


df_out.ForecastId = df_out.ForecastId.astype('int')
df_out2.ForecastId = df_out2.ForecastId.astype('int')


# In[ ]:


df_out['ConfirmedCases'] = (1/2)*(df_out['ConfirmedCases'] + df_out2['ConfirmedCases'])
df_out['Fatalities'] = (1/2)*(df_out['Fatalities'] + df_out2['Fatalities'])


# In[ ]:


df_out['ConfirmedCases'] = df_out['ConfirmedCases'].round().astype(int)
df_out['Fatalities'] = df_out['Fatalities'].round().astype(int)


# In[ ]:


df_out.tail()


# In[ ]:


df_out.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




