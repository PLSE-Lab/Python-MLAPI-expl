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


PATH_WEEK4 = '/kaggle/input/covid19-global-forecasting-week-4'

df_Train = pd.read_csv(f'{PATH_WEEK4}/train.csv', parse_dates=["Date"], engine='python')
df_Test = pd.read_csv(f'{PATH_WEEK4}/test.csv')


# In[ ]:


def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state


# In[ ]:


df_Train.rename(columns={'Country_Region':'Country'}, inplace=True)
df_Test.rename(columns={'Country_Region':'Country'}, inplace=True)
#df_Covid19.rename(columns={'Country/Region':'Country', 'ObservationDate': 'Date'}, inplace=True)
#df_Covid19.replace({'Country': 'Mainland China'}, 'China', inplace=True)
#df_Covid19.replace({'Country': 'Taiwan'}, 'Taiwan*', inplace=True)

EMPTY_VAL = "EMPTY_VAL"

df_Train.rename(columns={'Province_State':'State'}, inplace=True)
df_Train['State'].fillna(EMPTY_VAL, inplace=True)
df_Train['State'] = df_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

df_Test.rename(columns={'Province_State':'State'}, inplace=True)
df_Test['State'].fillna(EMPTY_VAL, inplace=True)
df_Test['State'] = df_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

#df_Covid19.rename(columns={'Province/State':'State'}, inplace=True)
#df_Covid19['State'].fillna(EMPTY_VAL, inplace=True)
#df_Covid19['State'] = df_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)
#df_Covid19.replace({'State': 'Taiwan*'}, 'Taiwan*', inplace=True)

#df_Train['Date'] = pd.to_datetime(df_Train['Date'], infer_datetime_format=True) # as pd.read_csv does parsed 'Date' as dates
df_Test['Date'] = pd.to_datetime(df_Test['Date'], infer_datetime_format=True) # dtype('Date') would be object, adnd we need to explicitly convert object to date as we did not use parse_dates
#df_Covid19['Date'] = pd.to_datetime(df_Covid19['Date'], infer_datetime_format=True)


# In[ ]:


df_groupByCountry = df_Train.loc[:, ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().groupby('Country').sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()
df_groupByCountry[:15].style.background_gradient(cmap='viridis_r')


# In[ ]:


import plotly.express as px

countries = df_groupByCountry.Country.unique().tolist()
df_plot = df_Train.loc[(df_Train.Country.isin(countries[:10])) & (df_Train.Date >= '2020-03-11'), ['Date', 'Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country', 'State']).max().reset_index().groupby(['Date', 'Country']).sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()

fig = px.bar(df_plot, x="Date", y="ConfirmedCases", color="Country", barmode="stack")
fig.update_layout(title='Rise of Confirmed Cases around top 10 countries', annotations=[dict(x='2020-03-21', y=150, xref="x", yref="y", text="Coronas Rise exponentially from here", showarrow=True, arrowhead=1, ax=-150, ay=-150)])
fig.show()


# In[ ]:


df_Train.loc[: , ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().nlargest(15, "ConfirmedCases").style.background_gradient(cmap='nipy_spectral')


# In[ ]:


import plotly.express as px

df_plot = df_Train.loc[: , ['Date', 'Country', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country']).max().reset_index()

df_plot.loc[:, 'Date'] = df_plot.Date.dt.strftime("%Y-%m-%d")
df_plot.loc[:, 'Size'] = np.power(df_plot["ConfirmedCases"]+1,0.3)-1 #np.where(df_plot['Country'].isin(['China', 'Italy']), df_plot['ConfirmedCases'], df_plot['ConfirmedCases']*300)

fig = px.scatter_geo(df_plot,
                     locations="Country",
                     locationmode = "country names",
                     hover_name="Country",
                     color="ConfirmedCases",
                     animation_frame="Date", 
                     size='Size',
                     #projection="natural earth",
                     title="Rise of Coronavirus Confirmed Cases")
fig.show()


# In[ ]:


import plotly.express as px

countries = df_groupByCountry.Country.unique().tolist()
df_plot = df_Train.loc[df_Train.Country.isin(countries[:10]), ['Date', 'Country', 'ConfirmedCases']].groupby(['Date', 'Country']).max().reset_index()

fig = px.line(df_plot, x="Date", y="ConfirmedCases", color='Country')
fig.update_layout(title='No.of Confirmed Cases per Day for Top 10 Countries',
                   xaxis_title='Date',
                   yaxis_title='No.of Confirmed Cases')
fig.show()


# In[ ]:


import plotly.express as px

countries = df_groupByCountry.Country.unique().tolist()
df_plot = df_Train.loc[df_Train.Country.isin(countries[:10]), ['Date', 'Country', 'Fatalities']].groupby(['Date', 'Country']).max().reset_index()

fig = px.scatter(df_plot, x="Date", y="Fatalities", color='Country')
fig.update_layout(title='No.of Fatalities per Day for Top 10 Countries',
                   xaxis_title='Date',
                   yaxis_title='No.of Fatalities')
fig.show()


# In[ ]:


MIN_TEST_DATE = df_Test.Date.min()

df_train = df_Train.loc[df_Train.Date < MIN_TEST_DATE, :]
y1_Train = df_train.iloc[:, -2]
y2_Train = df_train.iloc[:, -1]


# In[ ]:


def extractDate(df, colName = 'Date'):
    """
    This function does extract the date feature in to multiple features
    - week, day, month, year, dayofweek
    """
    assert colName in df.columns
    df = df.assign(week = df.loc[:, colName].dt.week,
                   day = df.loc[:, colName].dt.day,
                   month = df.loc[:, colName].dt.month,
                   #year = df.loc[:, colName].dt.year,
                   dayofweek = df.loc[:, colName].dt.dayofweek)
    return df


# In[ ]:


def createNewDataset(df):
    """
    This function does create a new dataset for modelling.
    """
    df_New = df.copy()
    
    #df_New = extractDate(df_New)
    #df_New.loc[:, 'Date_Int'] = (df_New.loc[:, 'Date'].dt.strftime("%m%d")).astype('int16')
    #df_New.drop(columns=['Date'], axis=1, inplace=True)
    
    #df_New.loc[:, 'Country_State'] = df_New.loc[:, 'Country'] + '_' + df_New.loc[:, 'State']
    #df_New.loc[:, 'Country_State'] = df_New[["State", "Country"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)
    #df_New.drop(columns=['Country', 'State'], axis=1, inplace=True)
    
    return df_New


# In[ ]:


X_Train = createNewDataset(df_train)
X_Test = createNewDataset(df_Test)


# In[ ]:


def RMSLE(pred, actual):
    return np.sqrt(np.mean(np.power((np.log(pred + 1) - np.log(actual + 1)), 2)))


# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')


# from xgboost import XGBRegressor
# from sklearn import preprocessing
# #from sklearn.model_selection import ShuffleSplit, cross_val_score
# 
# LEncoder = preprocessing.LabelEncoder()
# #skfold = ShuffleSplit(random_state=7)
# 
# countries = X_Train.Country.unique().tolist()
# 
# df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
# 
# for country in countries:
#     states = X_Train.loc[X_Train.Country == country, :].State.unique().tolist()
#     for state in states:
#         categoricalFeatures = ['Country', 'State']
#         
#         # Train
#         X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), :]
#         #X_Train_CS.loc[:, 'Country_State'] = X_Train_CS.loc[:, ["State", "Country"]].apply(lambda row: row[0] + "_" + row[1],axis=1)
#         
#         y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']
#         y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']
#         X_Train_CS.drop(columns=['Id', 'ConfirmedCases', 'Fatalities'], axis=1, inplace=True)
#         #X_Train_CS.drop(columns=categoricalFeatures, axis=1, inplace=True)
#         
#         X_Train_CS.loc[:, 'Country'] = LEncoder.fit_transform(X_Train_CS.loc[:, 'Country'])
#         X_Train_CS.loc[:, 'State'] = LEncoder.fit_transform(X_Train_CS.loc[:, 'State'])
#         #X_Train_CS.loc[:, 'Country_State'] = LEncoder.fit_transform(X_Train_CS.loc[:, 'Country_State'])
#         
#         # Test
#         X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), :]
#         #X_Test_CS.loc[:, 'Country_State'] = X_Test_CS.loc[:, ["State", "Country"]].apply(lambda row: row[0] + "_" + row[1],axis=1)
# 
#         X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']
#         X_Test_CS.drop(columns=['ForecastId'], axis=1, inplace=True)
#         #X_Test_CS.drop(columns=categoricalFeatures, axis=1, inplace=True)
# 
#         X_Test_CS.loc[:, 'Country'] = LEncoder.fit_transform(X_Test_CS.loc[:, 'Country'])
#         X_Test_CS.loc[:, 'State'] = LEncoder.fit_transform(X_Test_CS.loc[:, 'State'])
#         #X_Test_CS.loc[:, 'Country_State'] = LEncoder.fit_transform(X_Test_CS.loc[:, 'Country_State'])
# 
#         # Model fit & predict
#         model1 = XGBRegressor(n_estimators=1250)
#         #results = cross_val_score(model1, X_Train_CS, y1_Train_CS, cv=skfold)
#         #print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#         
#         model1.fit(X_Train_CS, y1_Train_CS)
#         y1_pred = model1.predict(X_Test_CS)
# 
#         model2 = XGBRegressor(n_estimators=1000)
#         #results = cross_val_score(model2, X_Train_CS, y2_Train_CS, cv=skfold)
#         
#         model2.fit(X_Train_CS, y2_Train_CS)
#         y2_pred = model2.predict(X_Test_CS)
#         
#         # Output Dataset
#         df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})
#         df_out = pd.concat([df_out, df], axis=0)
#     # Done for state loop
# # Done for country Loop

# In[ ]:


from fbprophet import Prophet

df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
i = 0
countries = X_Train.Country.unique().tolist()
for country in countries:
    states = X_Train.loc[X_Train.Country == country, :].State.unique().tolist()
    for state in states:       
        # Train
        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), :]
        confirmed = X_Train_CS.groupby('Date')['ConfirmedCases'].sum().reset_index()
        fatalities = X_Train_CS.groupby('Date')['Fatalities'].sum().reset_index()
        futureDays = (df_Test.Date.max() - X_Train_CS.Date.max()).days
        
        confirmed.columns = ['ds', 'y']
        confirmed['ds'] = pd.to_datetime(confirmed['ds'])
        
        # Confirmed Model fit & predict
        model1 = Prophet(interval_width=0.95)
        model1.fit(confirmed)
        futureConfirmed = model1.make_future_dataframe(periods=futureDays)
        forecastConfirmed = model1.predict(futureConfirmed)
        
        # Fatalities Model fit & predict
        fatalities.columns = ['ds', 'y']
        fatalities['ds'] = pd.to_datetime(fatalities['ds'])
        
        model2 = Prophet(interval_width=0.95)
        model2.fit(fatalities)
        futureFatalities = model2.make_future_dataframe(periods=futureDays)
        forecastFatalities = model2.predict(futureFatalities)
        
        # print(confirmed)
        # print(forecastConfirmed.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        # print(forecastFatalities.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        # model1.plot(forecastConfirmed)
        # model1.plot_components(forecastConfirmed)
        # print(forecastConfirmed[['ds', 'yhat']][(-1 * futureDays):])
        
        # Output Dataset
        X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), :]
        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']
        
        y1_pred = forecastConfirmed['yhat'][(-1 * futureDays):].values
        y2_pred = forecastFatalities['yhat'][(-1 * futureDays):].values
        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})
        df_out = pd.concat([df_out, df], axis=0)
        
        i += 1
        print(f'{i:03}; Predicting the {country} - {state}')
        # print(df)
        # break
    # Done for state loop
    # break
# Done for country Loop


# In[ ]:


df_out.ForecastId = df_out.ForecastId.astype('int')


# In[ ]:


#df_out.iloc[np.r_[42, 45, 97, 143, 175, 267, 327, 350, 420, 450, 540, 590, 680, 730, 2880, 2900, 2960, 3000, 3050, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000], :]
df_out.sample(10)


# In[ ]:


df_out.to_csv('submission.csv', index=False)


# In[ ]:




