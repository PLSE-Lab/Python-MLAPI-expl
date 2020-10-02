#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
import datetime
from matplotlib import dates
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from xgboost import XGBRegressor
import lightgbm as lgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
import scipy.integrate as integrate
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error, mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #  1. Exploratory Data Analysis (EDA)

# ## 1.1. Exploring the covid19 dataset for preliminary findings

# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv", parse_dates=['Date'])
test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv", parse_dates=['Date'])
display(train.tail())
display(test.tail())
display(train.describe())
print("Number of Country_Region: ", train['Country_Region'].nunique())
print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")
print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())


# In[ ]:


train.rename(columns={'Country_Region':'Country'}, inplace=True)
test.rename(columns={'Country_Region':'Country'}, inplace=True)
train.rename(columns={'Province_State':'State'}, inplace=True)
test.rename(columns={'Province_State':'State'}, inplace=True)


# ## 1.2. Visualizations

# In[ ]:


#change to more sophisticated graphs(animations and black graphs) for global stats and option to view each country's visualisation through dash or animation, trend of infection rates from first(or 100th) confirmed case, proportion of population affected

#confirmed_country = train.groupby(['Country/Region', 'Province/State']).agg({'ConfirmedCases':['sum']})
#fatalities_country = train.groupby(['Country/Region', 'Province/State']).agg({'Fatalities':['sum']})
confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})
total_date = confirmed_total_date.join(fatalities_total_date)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
total_date.plot(ax=ax1)
ax1.set_title("Global confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_date.plot(ax=ax2, color='orange')
ax2.set_title("Global deceased cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)


# In[ ]:


## Preparing India specific dataset along with true confirmed cases
india = train[train['Country']=="India"]
india.index = india['Date']

#creating extra columns for graphical purposes
india['log_confirmed'] = np.log(india['ConfirmedCases'])
india['log_confirmed'] = india['log_confirmed'].replace([np.inf, -np.inf], 0)
india['CC_LAG'] = india['ConfirmedCases'].shift(periods=1).fillna(0)
india['Fatalities_LAG'] = india['Fatalities'].shift(periods=1).fillna(0)
india['new_deaths'] = india['Fatalities'] - india['Fatalities_LAG']
india['new_deaths'] = india['new_deaths'].clip(lower=0)
india['log_new_cases'] = np.log(india['ConfirmedCases'] - india['CC_LAG'])
india['log_new_cases'] = india['log_new_cases'].replace([np.inf, -np.inf], 0)
display(india)

#adding true cases column
def actual(t_inc, t_inf, death_rate):      #function to create df with true cases data
    first = india[india['Fatalities'] > 0]['Date'].iloc[0]
    last = india['Date'].iloc[len(india)-1]
    period = t_inc + t_inf
    DD = datetime.timedelta(days=period)
    true_dict = {}
    while first <= last:
        deaths = india[india['Date']==first]['new_deaths'].iloc[0]
        true_cases = deaths/death_rate
        earlier = first - DD
        true_dict[earlier] = true_cases
        first += datetime.timedelta(days=1)
    df = pd.DataFrame.from_dict(true_dict, orient='index', columns=['True cases'])
    df = df.cumsum(axis = 0)    
    return df
    
df_true = actual(5, 15, 0.01)    #new df with only true cases data
df_true.plot()


# In[ ]:


# visualisations for India

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(17,7))

india['ConfirmedCases'].plot(ax=ax1)
ax1.set_title("Total confirmed cases vs time", size=13)
ax1.set_ylabel("Total cases", size=13)
ax1.set_xlabel("Date", size=13)

india['log_confirmed'].plot(ax=ax2, color='orange')
ax2.set_title("Log total vs time", size=13)
ax2.set_ylabel("Log total cases", size=13)
ax2.set_xlabel("Date", size=13)

india['log_new_cases'].plot(ax=ax3, color='green')
ax3.set_title("Log new cases vs time", size=13)
ax3.set_ylabel("Log new cases", size=13)
ax3.set_xlabel("Date", size=13)

india.index = india['log_confirmed']
india['log_new_cases'].plot(ax=ax4, color='red')
ax4.set_title("Log new cases vs Log total cases", size=13)
ax4.set_ylabel("Log new cases", size=13)
ax4.set_xlabel("Log total cases", size=13)




# # 2. Supplementary data

# In[ ]:


#population dataset
pop = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")
pop.head()


# # 3. Modeling

# ## 3.1. SEIR Model

# In[ ]:


## Defining the ODEs

# Susceptible equation
def Susceptible(S, I, R_t, T_inf):
    beta = (R_t / T_inf)
    dS_dt = -beta * I * S
    return dS_dt

# Exposed equation
def Exposed(S, E, I, R_t, T_inf, T_inc):
    beta = (R_t / T_inf)
    gamma = (T_inc**-1)
    dE_dt = beta*S*I - gamma*E
    return dE_dt

# Infected equation
def Infected(I, E, T_inc, T_inf):
    gamma = (T_inc**-1)
    delta = (T_inf**-1)
    dI_dt = gamma*E - delta*I
    return dI_dt

# Recovered/Remove/deceased equation
def Removed(I, T_inf):
    delta = (T_inf**-1)
    dR_dt = delta*I
    return dR_dt

def SEIR(t, y, R_t, T_inf, T_inc):
    
    if callable(R_t):
        reproduction = R_t(t)
    else:
        reproduction = R_t
        
    S, E, I, R = y
    
    dS_dt = Susceptible(S, I, reproduction, T_inf)
    dE_dt = Exposed(S, E, I, reproduction, T_inf, T_inc)
    dI_dt = Infected(I, E, T_inc, T_inf)
    dR_dt = Removed(I, T_inf)
    
    return([dS_dt, dE_dt, dI_dt, dR_dt])


# In[ ]:


def eval_model(param, data, population, inf, inc, return_solution=False, forecast_days=0):
    R_0 = param
    N = population   #total population
    n_infected = data['True cases'].iloc[0]  #number of individuals infected at the beginning of the outbreak
    max_days = len(data) + forecast_days      #number of days to predict for  
    s, e, i, r = (N - n_infected)/ N, 0, n_infected / N, 0     #initial conditions for SEIR model

    def time_varying_reproduction(t):    #functional form for R_0
        if t <= 30:         #pre-lockdown period
            return R_0
        else:
            return R_0*0.3     #reproduction number reduces after enforcing lockdown

    #solving the SEIR differential equation.
    sol = integrate.solve_ivp(SEIR, [0, max_days], [s, e, i, r], args=(time_varying_reproduction, inf, inc),
                    t_eval=np.arange(0, max_days))
    sus, exp, inf, rec = sol.y
    
    y_pred_cases = np.clip((inf + rec) * N , 0, np.inf)      #predicting actual cases
    y_true_cases = data['True cases'].values
    
    optim_days = min(25, len(data))      #number of days to optimise for
    weights = 1 / np.arange(1, optim_days + 1)[::-1]       #giving higher weightage to recent data
    
    #using mean square log error to evaluate
    msle = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[-optim_days:], weights)
    
    if return_solution:
        return msle, sol
    else:
        return msle


# In[ ]:


def fit_model(data, country, inf, inc):
    
    #population of the country
    N = pop[pop['Country (or dependency)']==country]['Population (2020)'].iloc[0]    #total population
        
    ####### Fit the real data by minimize the MSLE #######
    res = minimize(eval_model, np.array([2.5]), bounds=((1, 20),), args=(data, N, inf, inc, False), method='L-BFGS-B')
    print(res)
    print(res.x)
    
    msle, sol = eval_model(res.x, data, N, inf, inc, True, 300)
    sus, exp, inf, rec = sol.y
    
    print(msle)
    print(sol.y)    
    
    # Plotting result
    f = plt.figure(figsize=(16,5))
    ax1 = f.add_subplot(1,2,1)
    ax1.plot(sus, 'b', label='S(t)');
    ax1.plot(exp, 'y', label='E(t)');
    ax1.plot(inf, 'r', label='I(t)');
    ax1.plot(rec, 'c', label='R(t)');
    plt.title("SEIR model")
    plt.xlabel("Days", fontsize=10);
    plt.ylabel("Fraction of population", fontsize=10);
    plt.legend(loc='best');

    ax2 = f.add_subplot(1,2,2)
    preds = np.clip((inf + rec) * N ,0, np.inf)
    ax2.plot(range(len(data)),preds[:len(data)],label = 'Predictions')
    ax2.plot(range(len(data)),data['True cases'])
    plt.title('Predictions vs Actual')
    plt.ylabel("Population", fontsize=10);
    plt.xlabel("Days", fontsize=10);
    plt.legend(loc='best');


# In[ ]:


#plotting results after parameter fitting

country = "India"
T_inf = 10
T_inc = 5

fit_model(df_true, country, T_inf, T_inc)


# In[ ]:


##################################### TEST CODE ##################################


# In[ ]:


# Initial population conditions
country = "India"
N = pop[pop['Country (or dependency)']==country]['Population (2020)'].iloc[0]    #total population
n_infected = df_true['True cases'].iloc[0]  #number of individuals infected at the beginning of the outbreak

# Initial conditions for state variables in proportion terms
S0 = (N - n_infected)/N
E0 = 0.0
I0 = n_infected/N
R0 = 0.0

# Constant values and functional forms for parameters
T_inc = 5.2    # average incubation period
T_inf = 2.9    # average infectious period
R_0, k, L=[ 2.95469597 ,3.1, 15.32328881]    #initial conditions for parameters of decaying R_t

def time_varying_reproduction(t):    #functional form for R_0
    if t <= 30:         #pre-lockdown period
        return R_0
    else:
        return R_0*0.5     #reproduction number halves after enforcing lockdown

# Time vector
#t = [0, 100] 


# In[ ]:


# Result
solution = integrate.solve_ivp(SEIR, [0, 300], [S0, E0, I0, R0], args=(time_varying_reproduction, T_inf, T_inc), t_eval=np.arange(300))
sus, exp, inf, rec = solution.y


# In[ ]:


# Plotting result
f = plt.figure(figsize=(16,5))
ax1 = f.add_subplot(1,2,1)
ax1.plot(sus, 'b', label='S(t)');
ax1.plot(exp, 'y', label='E(t)');
ax1.plot(inf, 'r', label='I(t)');
ax1.plot(rec, 'c', label='R(t)');
plt.title("SEIR model")
plt.xlabel("Days", fontsize=10);
plt.ylabel("Fraction of population", fontsize=10);
plt.legend(loc='best');
    
ax2 = f.add_subplot(1,2,2)
preds = np.clip((inf + rec) * N ,0, np.inf)
ax2.plot(range(len(df_true)),preds[:len(df_true)],label = 'Predictions')
ax2.plot(range(len(df_true)),df_true['True cases'])
plt.title('Predictions vs Actual')
plt.ylabel("Population", fontsize=10);
plt.xlabel("Days", fontsize=10);
plt.legend(loc='best');


# ## 3.2. XGBoost + LightGBM

# In[ ]:


#missing values

EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state

X_Train = train.copy()

X_Train['State'].fillna(EMPTY_VAL, inplace=True)
X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")
X_Train["Date"]  = X_Train["Date"].astype(int)

display(X_Train.head())

X_Test = test.copy()

X_Test['State'].fillna(EMPTY_VAL, inplace=True)
X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")
X_Test["Date"]  = X_Test["Date"].astype(int)

display(X_Test.head())


# In[ ]:


le = preprocessing.LabelEncoder()

countries = X_Train.Country.unique()
df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
df_out2 = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in countries:
    states = X_Train.loc[X_Train.Country == country, :].State.unique()
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

df_out.ForecastId = df_out.ForecastId.astype('int')
df_out2.ForecastId = df_out2.ForecastId.astype('int')

df_out['ConfirmedCases'] = (1/2)*(df_out['ConfirmedCases'] + df_out2['ConfirmedCases'])
df_out['Fatalities'] = (1/2)*(df_out['Fatalities'] + df_out2['Fatalities'])

df_out['ConfirmedCases'] = df_out['ConfirmedCases'].round().astype(int)
df_out['Fatalities'] = df_out['Fatalities'].round().astype(int)

display(df_out)


# In[ ]:


#visualizing predictions

pred = df_out.iloc[4644:4687,-2]
dates_list_num = list(range(0,43))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

ax1.plot(dates_list_num, pred)


# In[ ]:


df_out.to_csv('submission.csv', index=False)


# In[ ]:




