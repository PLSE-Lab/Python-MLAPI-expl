#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import pycountry
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.integrate import odeint

# Calculate the Root Mean Squared Logarithmic Error (RMSLE)
from sklearn.metrics import mean_squared_log_error

from fbprophet import Prophet
from warnings import filterwarnings
filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Define the fuction to calculte the "root mean squared logarithmic error"

def rmsle(y_true, y_predict):
    return np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_predict))))
#log1p: natural logarithmic value of x+1


# # Explore the dataset 
# Exploring the dataset to see what information we have.

# In[ ]:


test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
train['unique_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)

print('Total Number of Country in Training Data: ', train['Country_Region'].nunique())
print('Has in total number of Province or States: ', train['Province_State'].nunique())
print('Date range: ', min(train['Date']), max(train['Date']), 'Today number of days: ', train['Date'].nunique())

print('Total Number of Country in Test Data: ', test['Country_Region'].nunique())
print('Has in total number of Province or States: ', test['Province_State'].nunique())
print('Date range: ', min(test['Date']), max(test['Date']), 'Today number of days: ', test['Date'].nunique())

print('For the training dataset, the number of regions on the first day ', min(train['Date']), ' are ', train[train['Date'] == min(train['Date'])]['Country_Region'].nunique())


# The training dataset has all 180 countries confirmed case and fatalities regardless whether that was the first day the country found its first case or not. Our training dataset will be driving by a lot of zero values from the earlier days.
# 
# Lets see globally, how the coronavirus cases number change. I will use plotly here.

# In[ ]:


tot_confirmed = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})
tot_fatalities = train.groupby(['Date']).agg({'Fatalities':['sum']})
tot_case_bydate = tot_confirmed.join(tot_fatalities)
tot_case_bydate.reset_index(inplace = True)
tot_case_bydate.head()

# Later need to put into one figure
fig = px.scatter(tot_case_bydate, x = 'Date', y = 'ConfirmedCases',
                width=800, height=400)
fig.update_layout(title='Global Confirmed Cases - Cumulative')
fig.show()

fig = px.scatter(tot_case_bydate, x = 'Date', y = 'Fatalities',
                width=800, height=400)
fig.update_layout(title='Global Fatalities Cases - Cumulative')
fig.show()


# In[ ]:


df_map = train.copy()
df_map['Date'] = df_map['Date'].astype(str)
df_map = df_map.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()

def get_iso3_util(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_3
    except:
        if 'Congo' in country_name:
            country_name = 'Congo'
        elif country_name == 'Diamond Princess' or country_name == 'Laos' or country_name == 'MS Zaandam':
            return country_name
        elif country_name == 'Korea, South':
            country_name = 'Korea, Republic of'
        elif country_name == 'Taiwan*':
            country_name = 'Taiwan'
        elif country_name == 'Burma':
            country_name = 'Myanmar'
        elif country_name == 'West Bank and Gaza':
            country_name = 'Gaza'
        country = pycountry.countries.search_fuzzy(country_name)
        return country[0].alpha_3

d = {}
def get_iso3(country):
    if country in d:
        return d[country]
    else:
        d[country] = get_iso3_util(country)
    
df_map['iso_alpha'] = df_map.apply(lambda x: get_iso3(x['Country_Region']), axis=1)

df_map['ln(ConfirmedCases)'] = np.log(df_map.ConfirmedCases + 1)
df_map['ln(Fatalities)'] = np.log(df_map.Fatalities + 1)

px.choropleth(df_map, 
              locations="iso_alpha", 
              color="ln(ConfirmedCases)", 
              hover_name="Country_Region", 
              hover_data=["ConfirmedCases"] ,
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.dense, 
              title='Total Confirmed Cases growth(Logarithmic Scale)')


# # Machine Learning Model
# ## 1. Linear Regression
# We start with the basic linear regression model. Because the increase of confirmed cases and fatalities are not linear (we can tell from above graphs), and it is more like exponential growth, we use the log value of confirmed cases and fatalities for the model. Be careful: since the predicted value is log value, do not forget to convert back to nomral number. 

# In[ ]:


test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
test['Province_State'].fillna('', inplace = True)
train['Province_State'].fillna('', inplace = True)
test['Date'] = pd.to_datetime(test['Date'])
train['Date'] = pd.to_datetime(train['Date'])

train['unique_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)


# In[ ]:


days = 11

pred_confirm = []
pred_fatality = []
test_length = test['Date'].nunique()

# Prepare to calculate the loss function:
y_true_c = []
y_predict_c = []
y_true_f = []
y_predict_f = []

# Using Linear Regression Model to Predict Confirmed Cases and Fatalities.
for uid in train['unique_id'].unique():
    df = train[train['unique_id'] == uid]
    
    # use log transformed value
    y_c = df.set_index('Date')['ConfirmedCases'].values.flatten()
    # Append the last 7 days to y_true data for loss function calculation later
    y_true_c.append(y_c[-days:])
    index = len(y_c) - days
    y_confirm = y_c[:index]
    y_confirm = np.log(y_confirm)
    y_confirm[y_confirm == np.inf] = 0
    y_confirm[y_confirm == -np.inf] = 0
    
    y_f = df.set_index('Date')['Fatalities'].values.flatten()
    index = len(y_f) - days
    y_true_f.append(y_f[-days:])
    y_fatality = y_f[:index]
    y_fatality = np.log(y_fatality)
    y_fatality[y_fatality == np.inf] = 0
    y_fatality[y_fatality == -np.inf] = 0
    
    x = np.arange(0, len(y_confirm), 1)
    x = x.reshape(-1,1)
    x_test = np.arange(max(x) + 1, max(x) + test_length + 1, 1)
    x_test = x_test.reshape(-1,1)
    
    lreg = LinearRegression()
    
    lreg.fit(x,y_confirm)
    predict_c = lreg.predict(x_test)
    pred_confirm.append(predict_c)
    y_predict_c.append(predict_c[:days])
    
    lreg.fit(x,y_fatality)
    predict_f = lreg.predict(x_test)
    pred_fatality.append(predict_f)
    y_predict_f.append(predict_f[:days])

pred_confirm = [item for sublist in pred_confirm for item in sublist]
pred_fatality = [item for sublist in pred_fatality for item in sublist]

# Convert log to normal value:
predict_c = np.exp(pred_confirm).astype(int)
predict_f = np.exp(pred_fatality).astype(int)

# Replace any negative value with 0
predict_c[predict_c < 0] = 0
predict_f[predict_f < 0] = 0

submission = pd.DataFrame({'ForecastId': test['ForecastId'], 
                           'ConfirmedCases': predict_c, 
                           'Fatalities': predict_f})
# submission.to_csv('submission.csv', index = False)


####### Calculate the loss function: RMSLE
y_true_c = [item for sublist in y_true_c for item in sublist]
y_predict_c = [item for sublist in y_predict_c for item in sublist]
y_true_f = [item for sublist in y_true_f for item in sublist]
y_predict_f = [item for sublist in y_predict_f for item in sublist]

# Convert log to normal value:
y_predict_c2 = np.exp(y_predict_c).astype(int)
y_predict_f2 = np.exp(y_predict_f).astype(int)

# Replace any negative value with 0
y_predict_c2[y_predict_c2 < 0] = 0
y_predict_c2 = y_predict_c2.tolist()
y_predict_f2[y_predict_f2 < 0] = 0
y_predict_f2 = y_predict_f2.tolist()

Y_true = [*y_true_c, *y_true_f]
Y_predict =[*y_predict_c2, *y_predict_f2]

# Calculate the lost: 1.8981
rmsle(Y_true, Y_predict)


# ## 2. XGBoost Model
# Three classes of boosting: 
# 1. Adaptive Boosting
# 2. Gradient Boosting
# 3. XGBoost: has the tendency to fill in the missing values. It is an advanced version of gradient boosting (extreme gradient boosting). It is faster and more efficient.
#     - It is faster and can build more efficient model compare with Gradient Boosting;
#     - Supports parallelization by creating decision trees;
#     - Weight each input based on what is the most relevant optimal.
# 
# 

# In[ ]:


days = 0

pred_confirm = []
pred_fatality = []
test_length = test['Date'].nunique()

# Prepare to calculate the loss function:
y_true_c = []
y_predict_c = []
y_true_f = []
y_predict_f = []

# Using Linear Regression Model to Predict Confirmed Cases and Fatalities.
for uid in train['unique_id'].unique():
    df = train[train['unique_id'] == uid]
    
    # use log transformed value
    y_c = df.set_index('Date')['ConfirmedCases'].values.flatten()
    # Append the last 7 days to y_true data for loss function calculation later
    y_true_c.append(y_c[-days:])
    index = len(y_c) - days
    y_confirm = y_c[:index]
    
    y_f = df.set_index('Date')['Fatalities'].values.flatten()
    index = len(y_f) - days
    y_true_f.append(y_f[-days:])
    y_fatality = y_f[:index]
    
    x = np.arange(0, len(y_confirm), 1)
    x = x.reshape(-1,1)
    x_test = np.arange(max(x) + 1, max(x) + test_length + 1, 1)
    x_test = x_test.reshape(-1,1)
    
    # xgboost
    model =  XGBRegressor(n_estimators=1000)
    model.fit(x,y_confirm)                
    predict_c = model.predict(x_test)
    pred_confirm.append(predict_c)
    y_predict_c.append(predict_c[:days])
    
    model.fit(x,y_fatality)
    predict_f = model.predict(x_test)
    pred_fatality.append(predict_f)
    y_predict_f.append(predict_f[:days])    
    
pred_confirm = [item for sublist in pred_confirm for item in sublist]
pred_fatality = [item for sublist in pred_fatality for item in sublist]


submission = pd.DataFrame({'ForecastId': test['ForecastId'], 
                           'ConfirmedCases': pred_confirm,
                           'Fatalities': pred_fatality})

submission = submission.round(0)

submission.to_csv('submission.csv', index = False)


# In[ ]:


# ####### Calculate the loss function: RMSLE
# # Calculate the loss
# y_true_c = [item for sublist in y_true_c for item in sublist]
# y_predict_c = [item for sublist in y_predict_c for item in sublist]
# y_predict_c = [round(elem, 2) for elem in y_predict_c]
# y_true_f = [item for sublist in y_true_f for item in sublist]
# y_predict_f = [item for sublist in y_predict_f for item in sublist]
# y_predict_f = [round(elem, 2) for elem in y_predict_f]


# Y_true = [*y_true_c, *y_true_f]
# Y_predict =[*y_predict_c, *y_predict_f]

# # Calculate the loss: 0.9131976
# rmsle(Y_true, Y_predict)


# ## 3. SIR Model
# The resulst for SIR model is not that accurate. We did not use it after all.

# In[ ]:


test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
test['Date'] = pd.to_datetime(test['Date'])
train['Date'] = pd.to_datetime(train['Date'])
train['Province_State'] = train['Province_State'].fillna('None')
train['unique_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)


# In[ ]:


train_hubei = train[train['Province_State']=='Hubei']
train_hubei['Date'] = train_hubei['Date'].astype(str)


# In[ ]:


def deriv(y, t, N,beta,gamma):

            S,I,R = y
            dSdt = -beta * S * I/N
            dIdt = beta * S * I/N  - gamma * I
            dRdt = gamma * I
            return dSdt,dIdt,dRdt
        
def odeint_func(t,N,I0,R0,beta,gamma):
    
    S0 = (N - I0 - R0)
    y0 = S0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))

    return np.ravel(np.vstack((ret[:,1],ret[:,2])))


# In[ ]:


def run_curve_fit_diff_S(y1,y2,N):
   
    I0 = y1[0]
    R0 = y2[0]
    def deriv(y, t,N, beta,gamma,tau):

        S,I,R = y
#         dNdt = -(1/10**tau)*N
        dSdt = -beta * S * I/N
        dIdt = beta * S * I/N  - gamma * I
        dRdt = gamma * I
        return dSdt,dIdt,dRdt
    
    def odeint_func(t,N0,beta,gamma,tau,I0,R0):
        
        S0 = (N - I0 - R0)
        y0 = S0, I0, R0
        ret = odeint(deriv, y0, t, args=(N,beta, gamma,tau))
        return np.ravel(np.vstack((ret[:,1],ret[:,2])))

    t = np.arange(0,len(y1),1)
    y_t = np.vstack((y1,y2))

    values , pcov = curve_fit(lambda t,beta,gamma,tau: odeint_func(t,N,beta,gamma,tau,I0,R0), 
                          t, np.ravel(y_t) ,bounds=((0,0,-np.inf),(1,1,np.inf)),maxfev=999999)
        
    return values[0],values[1],values[2]


# In[ ]:


def get_SIR_data_diff_S(beta,gamma,tau, y_active,N, days):
    
    I0 = y_active[0]
    R0 = 0
    t = np.arange(0,len(y_active)+days,1)
    def deriv(y, t,N, beta,gamma,tau):

        S,I,R = y
#         dNdt = -(1/10**tau)*N
        dSdt = -beta * S * I/N
        dIdt = beta * S * I/N  - gamma * I
        dRdt = gamma * I
        return dSdt,dIdt,dRdt
    
    S0 = (N - I0 - R0)
    y0 = S0, I0, R0
    ret = odeint(deriv, y0, t, args=(N,beta, gamma,tau))
        
    return ret.T


# In[ ]:


import datetime
pop = 58500000
y_conf  = train_hubei.set_index('Date')['ConfirmedCases'].values.flatten()
y_death = train_hubei.set_index('Date')['Fatalities'].values.flatten()

idx = np.argwhere(y_conf>0)[0][0]
    
y_active = y_conf[idx:]
y_deaths = y_death[idx:]
beta,gamma,tau = run_curve_fit_diff_S(y_active,y_deaths,pop)
print(beta,gamma,tau)
fig = plt.figure()
ax = plt.gca()
start_date =  datetime.datetime.strptime(train_hubei['Date'].values[idx], '%Y-%m-%d')
first_date = datetime.datetime.strptime(train_hubei['Date'].values[0], '%Y-%m-%d')

date_line = [first_date + datetime.timedelta(days=x) for x in range(len(train_hubei))]

ax.plot(date_line,y_conf,'r-', label='Active Cases')
ax.plot(date_line,y_death,'g-', label='Active Cases')
N_days = 360
date_line_ext = [first_date + datetime.timedelta(days=x) for x in range(len(y_conf)+N_days)]

S,I,R = get_SIR_data_diff_S(beta,gamma,tau,y_active,pop,N_days)
#N  = np.append(np.ones(idx)*max(N),N)
S = np.append(np.ones(idx)*max(S),S)
I = np.append(np.zeros(idx),I)
R = np.append(np.zeros(idx),R)
#ax.plot(date_line_ext,N,'k--',label='Total Population Fit')

ax.plot(date_line_ext,S,'b--',label='Susceptible Cases (Fit)')
ax.plot(date_line_ext,I,'r--',label='Active Cases (Fit)')
ax.plot(date_line_ext,R,'g--',label = 'Death Cases (Fit)')
plt.yscale('linear')
plt.legend()

#plt.ylim(10,1e8)


# ## 4. Prophet (Base model)
# The loss function is not great for the world prediction (2.7939650495039494), so we did not use this model eventually and it takes forever to run.

# In[ ]:


# test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
# train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
# train['unique_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)
# train['Date'] = pd.to_datetime(train['Date'])
# test['Date'] = pd.to_datetime(test['Date'])


# In[ ]:


# # Using New York as an example
# # I want to predict any date after March 26th
# days = 7

# df = train[train['unique_id'] == 'US_New York']
# index = len(df) - days

# df = df[:index]

# confirmed = df[['Date','ConfirmedCases']]
# fatalities = df[['Date','Fatalities']]

# confirmed.columns = ['ds','y']
# fatalities.columns = ['ds','y']


# In[ ]:


# model = Prophet(interval_width = 0.95)
# model.fit(confirmed)
# furture = model.make_future_dataframe(periods = days)
# furture.tail()


# In[ ]:


# # Set up upper value and lower value (set up tolerance)
# forecast = model.predict(furture)
# forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[ ]:


# confirmed_forecast_plot = model.plot(forecast)


# In[ ]:


# confirmed_forecast_plot = model.plot_components(forecast)


# In[ ]:


# days = 12

# # Prepare to calculate the loss function:
# y_true_c = []
# y_true_f = []
# y_predict_c = []
# y_predict_f = []

# # Using Linear Regression Model to Predict Confirmed Cases and Fatalities.
# for uid in train['unique_id'].unique():
#     df = train[train['unique_id'] == uid]
   
#     # Append the last 7 days to y_true data for loss function calculation later    
#     y_true_c.append(df['ConfirmedCases'][-days:])
#     y_true_f.append(df['Fatalities'][-days:])
    
#     index = len(df) - days
#     df = df[:index]
    
#     confirmed = df[['Date','ConfirmedCases']]
#     fatalities = df[['Date','Fatalities']]

#     confirmed.columns = ['ds','y']
#     fatalities.columns = ['ds','y']
    
#     # Predict the ConfirmedCases
#     model_c = Prophet(interval_width = 0.95)
#     model_c.fit(confirmed) 
#     furture_c = model.make_future_dataframe(periods=days)
#     forecast_c = model.predict(furture_c)
#     y_predict_c.append(forecast_c[-days:]['yhat'])
    
#     # Predict Fatalities
#     model_f = Prophet(interval_width = 0.95)
#     model_f.fit(fatalities)
#     furture_f = model.make_future_dataframe(periods=days)
#     forecast_f = model.predict(furture_f)
#     y_predict_f.append(forecast_f[-days:]['yhat'])    


# In[ ]:


# Y_true = [*y_true_c, *y_true_f]
# Y_predict =[*y_predict_c, *y_predict_f]

# # Calculate the lostt
# rmsle(Y_true, Y_predict)


# In[ ]:


# train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
# test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

# train = train.fillna('NA')
# test = test.fillna('NA')

# train['Date'] = pd.to_datetime(train['Date'])
# train['Date'] = train['Date'].dt.strftime('%m%d')   # Change to only month and day. Because years are all the same. 
# test['Date'] = pd.to_datetime(test['Date'])
# test['Date'] = test['Date'].dt.strftime('%m%d')

# country_list = train['Country_Region'].unique()

# sub = []
# for country in country_list:
    
#     province_list = train.loc[train['Country_Region'] == country].Province_State.unique()  #Get the province list by country
    
#     for province in province_list:
        
#         X_train = train.loc[(train['Country_Region'] == country) & (train['Province_State'] == province),['Date']].astype('int')  # Date
#         Y_train_c = train.loc[(train['Country_Region'] == country) & (train['Province_State'] == province),['ConfirmedCases']]    # Confirmed Cases 
#         Y_train_f = train.loc[(train['Country_Region'] == country) & (train['Province_State'] == province),['Fatalities']]        # Fatalities
#         X_test = test.loc[(test['Country_Region'] == country) & (test['Province_State'] == province), ['Date']].astype('int')     # Date
#         X_forecastId = test.loc[(test['Country_Region'] == country) & (test['Province_State'] == province), ['ForecastId']]       # ForecastId
#         X_forecastId = X_forecastId.values.tolist()      # Put ForecastId into an array (nested)
#         X_forecastId = [v[0] for v in X_forecastId]      # Open the nested array to one list
        
#         # Use XGBRegressor to fit and predict
#         model_c = XGBRegressor(n_estimators=1000)
#         model_c.fit(X_train, Y_train_c)
#         Y_pred_c = model_c.predict(X_test)
        
#         model_f = XGBRegressor(n_estimators=1000)
#         model_f.fit(X_train, Y_train_f)
#         Y_pred_f = model_f.predict(X_test)
        
#         for j in range(len(Y_pred_c)):
#             dic = { 'ForecastId': X_forecastId[j], 'ConfirmedCases': Y_pred_c[j], 'Fatalities': Y_pred_f[j]}
#             sub.append(dic)

# submission = pd.DataFrame(sub)
# submission = submission.round(0)
# # submission[['ForecastId','ConfirmedCases','Fatalities']].to_csv(path_or_buf='submission.csv',index=False)

