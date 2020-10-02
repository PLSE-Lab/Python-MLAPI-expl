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


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


train_df=pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
test_df=pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.tail()
#Overlap between training and test sets


# In[ ]:


test_df.info()


# In[ ]:


train_df[train_df["Country/Region"]=="China"]
#Note that ConfirmedCases and Fatalities are cumulative, hence we should select the row with the last date, for the total count in a country


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


temp_Confirmed = train_df.loc[train_df["Date"]==train_df["Date"][len(train_df)-1]].groupby(['Country/Region'])["ConfirmedCases"].sum().reset_index()
temp_Confirmed.head()


# In[ ]:


Country_Confirmed=pd.DataFrame()
Country_Confirmed["Name"]=temp_Confirmed["Country/Region"]
Country_Confirmed["Confirmed"]=temp_Confirmed["ConfirmedCases"]
Country_Confirmed.head()


# In[ ]:


fig=px.choropleth(Country_Confirmed,locations="Name", locationmode="country names", color="Confirmed" )
fig.update_layout(title="Spread of Corona cases across countries")
fig.show()


# In[ ]:


temp_Fatalities = train_df.loc[train_df["Date"]==train_df["Date"][len(train_df)-1]].groupby(['Country/Region'])["Fatalities"].sum().reset_index()
temp_Fatalities.head()


# In[ ]:


Country_Fatalities=pd.DataFrame()
Country_Fatalities["Name"]=temp_Fatalities["Country/Region"]
Country_Fatalities["Fatalities"]=temp_Fatalities["Fatalities"]
Country_Fatalities.head()


# In[ ]:


fig=px.choropleth(Country_Fatalities,locations="Name", locationmode="country names", color="Fatalities", color_continuous_scale="Viridis" )
fig.update_layout(title="Spread of Corona fatalities across countries")
fig.show()


# In[ ]:


China=train_df[train_df["Country/Region"]=="China"].groupby(["Date"])["ConfirmedCases"].sum().reset_index()
China['Date']=pd.to_datetime(China['Date'])


# In[ ]:


China_Fatalities=train_df[train_df["Country/Region"]=="China"].groupby(["Date"])["Fatalities"].sum().reset_index()
China_Fatalities['Date']=pd.to_datetime(China_Fatalities['Date'])


# In[ ]:


Italy=train_df[train_df["Country/Region"]=="Italy"].groupby(["Date"])["ConfirmedCases"].sum().reset_index()
Italy['Date']=pd.to_datetime(Italy['Date'])


# In[ ]:


Italy_Fatalities=train_df[train_df["Country/Region"]=="Italy"].groupby(["Date"])["Fatalities"].sum().reset_index()
Italy_Fatalities['Date']=pd.to_datetime(Italy_Fatalities['Date'])


# In[ ]:


Other_countries=train_df[~train_df["Country/Region"].isin(['China','Italy'])]
Other_countries['Country/Region']= 'Others'
Other_countries=Other_countries.groupby(["Date"])["ConfirmedCases"].sum().reset_index()
Other_countries['Date']=pd.to_datetime(Other_countries['Date'])


# In[ ]:


Other_Fatalities=train_df[~train_df["Country/Region"].isin(["China","Italy"])]
Other_Fatalities['Country/Region']= 'Others'
Other_Fatalities=Other_Fatalities.groupby(["Date"])["Fatalities"].sum().reset_index()
Other_Fatalities['Date']=pd.to_datetime(Other_Fatalities['Date'])


# In[ ]:


fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=China['Date'], y=China['ConfirmedCases'], name='Cases in China', line_color='blue'))
fig1.add_trace(go.Scatter(x=Italy['Date'], y=Italy['ConfirmedCases'], name='Cases in Italy', line_color='red'))
fig1.add_trace(go.Scatter(x=Other_countries['Date'], y=Other_countries['ConfirmedCases'], name='Cases in other countries', line_color='green'))
fig1.show()


# In[ ]:


fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=China_Fatalities['Date'], y=China_Fatalities['Fatalities'], name='Fatalities in China', line=dict(dash="dot", width=4), line_color='blue'))
fig2.add_trace(go.Scatter(x=Italy_Fatalities['Date'], y=Italy_Fatalities['Fatalities'], name='Fatalities in Italy', line=dict(dash="dot", width=4), line_color='red'))
fig2.add_trace(go.Scatter(x=Other_Fatalities['Date'], y=Other_Fatalities['Fatalities'], name='Fatalities in Other countries', line=dict(dash="dot", width=4), line_color='green'))


# In[ ]:


train_df.isnull().sum()


# In[ ]:


X=train_df[["Date","Lat","Long"]]
X["Date"]=X["Date"].apply(lambda x:x.replace("-",""))
X["Date"]  = X["Date"].astype(int)


# In[ ]:


X.head()


# In[ ]:


X.info()


# In[ ]:


y1=train_df["ConfirmedCases"]
y2=train_df["Fatalities"]


# In[ ]:


#Confirmed Cases
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(X, y1, test_size = .20, random_state = 42)


# In[ ]:


dt1=DecisionTreeRegressor()
dt1.fit(X_train_confirmed, y_train_confirmed)
y_pred_dt_confirmed=dt1.predict(X_test_confirmed)


# In[ ]:


np.sqrt(mean_squared_log_error( y_test_confirmed, y_pred_dt_confirmed ))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfc1=RandomForestRegressor(n_estimators=5, min_samples_split=3, random_state=42)
rfc1.fit(X_train_confirmed, y_train_confirmed)


# In[ ]:


y_pred_rfc_confirmed=rfc1.predict(X_test_confirmed)


# In[ ]:


np.sqrt(mean_squared_log_error( y_test_confirmed, y_pred_rfc_confirmed ))


# In[ ]:


#Fatalities
X_train_fatal, X_test_fatal, y_train_fatal, y_test_fatal = train_test_split(X, y2, test_size = .20, random_state = 42)


# In[ ]:


dt2=DecisionTreeRegressor()


# In[ ]:


dt2.fit(X_train_fatal, y_train_fatal)


# In[ ]:


y_pred_dt_fatal=dt2.predict(X_test_fatal)


# In[ ]:


np.sqrt(mean_squared_log_error( y_test_fatal, y_pred_dt_fatal ))


# In[ ]:


rfc2=RandomForestRegressor(n_estimators=5, min_samples_split=3, random_state=42)
rfc2.fit(X_train_fatal, y_train_fatal)


# In[ ]:


y_pred_rfc_fatal=rfc2.predict(X_test_fatal)


# In[ ]:


np.sqrt(mean_squared_log_error( y_test_fatal, y_pred_rfc_fatal ))


# In[ ]:


test_df.isnull().sum()


# In[ ]:


test_data=test_df[["Date","Lat","Long"]]


# In[ ]:


test_data.info()


# In[ ]:


test_data["Date"]=test_data["Date"].apply(lambda x:x.replace("-",""))
test_data["Date"]  = test_data["Date"].astype(int)


# In[ ]:


test_data.head()


# In[ ]:


#y_confirmed=dt1.predict(test_data)
y_confirmed=rfc1.predict(test_data)


# In[ ]:


#y_fatal=dt2.predict(test_data)
y_fatal=rfc2.predict(test_data)


# In[ ]:


submission=pd.DataFrame({'ForecastId': test_df["ForecastId"], 'ConfirmedCases': y_confirmed, 'Fatalities': y_fatal})


# In[ ]:


submission["ConfirmedCases"]=submission["ConfirmedCases"].astype(int)
submission["Fatalities"]=submission["Fatalities"].astype(int)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

