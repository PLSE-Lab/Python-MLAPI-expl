#!/usr/bin/env python
# coding: utf-8

# # SEIR compartments Model
# This is a working example of a [SIER](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model) model with added compartments for hospitalization, social distance and Covid19 test.
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error, mean_squared_error


# ## Parameters used in the model
# `R_t` = reproduction number at time t. 
# 
# **Transition times**
# * `T_inc` = average incubation period.
# * `T_inf` = average infectious period. 
# * `T_hosp` = average time a patient is in hospital before either recovering or becoming critical. 
# * `T_crit` = average time a patient is in a critical state (either recover or die). 
# * `m` = fraction of infections that are asymptomatic or mild. 
# * `c` = fraction of severe cases that turn critical. 
# * `f` = fraction of critical cases that are fatal. 
# * `p` = fraction of tests that are positive. 

# In[ ]:


# Susceptible equation
def dS_dt(S, I, R_t, t_inf):
    return -(R_t / t_inf) * I * S


# Exposed equation
def dE_dt(S, E, I, R_t, t_inf, t_inc):
    return (R_t / t_inf) * I * S - (E / t_inc)


# Infected equation
def dI_dt(I, E, t_inc, t_inf):
    return (E / t_inc) - (I / t_inf)


# Hospialized equation
def dH_dt(I, C, H, t_inf, t_hosp, t_crit, m_a, f_a):
    return ((1 - m_a) * (I / t_inf)) + ((1 - f_a) * C / t_crit) - (H / t_hosp)


# Critical equation
def dC_dt(H, C, t_hosp, t_crit, c_a):
    return (c_a * H / t_hosp) - (C / t_crit)


# Recovered equation
def dR_dt(I, H, t_inf, t_hosp, m_a, c_a):
    return (m_a * I / t_inf) + (1 - c_a) * (H / t_hosp)


# Deaths equation
def dD_dt(C, t_crit, f_a):
    return f_a * C / t_crit

# Positive equation
def dP_dt(C, t_inc, p_a):
    return p_a * C / t_inc

def SEIR_COM_model(t, y, R_t, t_inc=2.9, t_inf=5.2, t_hosp=4, t_crit=14, m_a=0.8, c_a=0.1, f_a=0.3,p_a=0.3):
    """

    :param t: Time step for solve_ivp
    :param y: Previous solution or initial values
    :param R_t: Reproduction number
    :param t_inc: Average incubation period. 
    :param t_inf: Average infectious period. 
    :param t_hosp: Average time a patient is in hospital before either recovering or becoming critical. 
    :param t_crit: Average time a patient is in a critical state (either recover or die). 
    :param m_a: Fraction of infections that are asymptomatic or mild. 
    :param c_a: Fraction of severe cases that turn critical. 
    :param f_a: Fraction of critical cases that are fatal. 
    :param p_a: Fraction of testing cases that are positive.
    :return:
    """
    if callable(R_t):
        reprod = R_t(t)
    else:
        reprod = R_t
        
    S, E, I, R, H, C, D , P= y
    
    S_out = dS_dt(S, I, reprod, t_inf)
    E_out = dE_dt(S, E, I, reprod, t_inf, t_inc)
    I_out = dI_dt(I, E, t_inc, t_inf)
    R_out = dR_dt(I, H, t_inf, t_hosp, m_a, c_a)
    H_out = dH_dt(I, C, H, t_inf, t_hosp, t_crit, m_a, f_a)
    C_out = dC_dt(H, C, t_hosp, t_crit, c_a)
    D_out = dD_dt(C, t_crit, f_a)
    P_out = dP_dt(C, t_inc, p_a)
    return [S_out, E_out, I_out, R_out, H_out, C_out, D_out,P_out]


# In[ ]:


def plot_model(solution, title='SEIR+COM model'):
    sus, exp, inf, rec, hosp, crit, death , positive= solution.y
    
    cases = inf + rec + hosp + crit + death + positive

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
    fig.suptitle(title)
    
    ax1.plot(sus, 'tab:blue', label='Susceptible');
    ax1.plot(exp, 'tab:orange', label='Exposed');
    ax1.plot(inf, 'tab:red', label='Infected');
    ax1.plot(rec, 'tab:green', label='Recovered');
    ax1.plot(hosp, 'tab:purple', label='Hospitalised');
    ax1.plot(crit, 'tab:brown', label='Critical');
    ax1.plot(death, 'tab:cyan', label='Dead');
    ax1.plot(death, 'tab:pink', label='Positive');
    
    
    ax1.set_xlabel("Days", fontsize=10);
    ax1.set_ylabel("Fraction of population", fontsize=10);
    ax1.legend(loc='best');
    
    ax2.plot(cases, 'tab:red', label='Cases');    
    ax2.set_xlabel("Days", fontsize=10);
    ax2.set_ylabel("Fraction of population (Cases)", fontsize=10, color='tab:red');
    
    ax3 = ax2.twinx()
    ax3.plot(death, 'tab:cyan', label='Dead');    
    ax3.set_xlabel("Days", fontsize=10);
    ax3.set_ylabel("Fraction of population (Fatalities)", fontsize=10, color='tab:cyan');


# # Model
# Lets assume that there is some intervention that causes the reproduction number (`R_0`) to fall to a lower value (`R_t`) at a certain time. Assuming reproduction number will deacrease when keeping safe social ditance. 

# # Fitting the model to data
# Parameters in the model:
# * Average incubation period, `t_inc`
# * Average infection period, `t_inf`
# * Average hospitalization period, `t_hosp`
# * Average critital period, `t_crit`
# * The fraction of mild/asymptomatic cases, `m_a`
# * The fraction of severe cases that turn critical, `c_a`
# * The fraction of critical cases that result in a fatality, `f_a`
# * The fraction of positive cases, `p_a`
# * Reproduction number, `R_0` or `R_t`
# 
# Other factors such as:
# * social distance
# * Population demographic of a country (is a significant proportion of the population old?). This is the `a` subscript
# * Heathcare system capacity (hostpital beds per capita)
# * Number of testing kits available
# Using a Hill decay, which has 2 parameters, `k` and `L` (the half decay constant):

# In[ ]:


DATE_BORDER = '2020-04-08'

data_path = Path('/kaggle/input/covid19-global-forecasting-week-3/')

train = pd.read_csv(data_path / 'train.csv', parse_dates=['Date'])
test = pd.read_csv(data_path /'test.csv', parse_dates=['Date'])
submission = pd.read_csv(data_path /'submission.csv', index_col=['ForecastId'])

# Load the population data into lookup dicts
pop_info = pd.read_csv('/kaggle/input/covid19-population-data/population_data.csv')
country_pop = pop_info.query('Type == "Country/Region"')
province_pop = pop_info.query('Type == "Province/State"')
country_lookup = dict(zip(country_pop['Name'], country_pop['Population']))
province_lookup = dict(zip(province_pop['Name'], province_pop['Population']))

# Load the social distance data into lookup dicts
distance_info = pd.read_csv('/kaggle/input/personal-distance-in-42-countries/personal_distance.csv')
distance_info['Country/Region']=distance_info['Country']
distance_pop = distance_info.query('Country == "Country/Region"')
distance_lookup = dict(zip(distance_pop['Country'], distance_pop.iloc[:,1]))
#distance_lookup = dict(zip(province_pop['Name'], province_pop['Population']))


# Fix the Georgia State/Country confusion - probably a better was of doing this :)
train['Province_State'] = train['Province_State'].replace('Georgia', 'Georgia (State)')
test['Province_State'] = test['Province_State'].replace('Georgia', 'Georgia (State)')
province_lookup['Georgia (State)'] = province_lookup['Georgia']

train['Area'] = train['Province_State'].fillna(train['Country_Region'])
test['Area'] = test['Province_State'].fillna(test['Country_Region'])

# https://www.kaggle.com/c/covid19-global-forecasting-week-1/discussion/139172
train['ConfirmedCases'] = train.groupby('Area')['ConfirmedCases'].cummax()
train['Fatalities'] = train.groupby('Area')['Fatalities'].cummax()

# load covid19 tests data
test_info = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/Tests_Conducted_31Mar2020.csv')
test_info['Country']=test_info['Country or region']
test_pop =test_info.query('Country== "Country/Region"')
#tests_lookup = dict(zip(province_pop['Name'], province_pop['Population']))
tests_lookup = dict(zip(test_pop['Country'], test_pop['Tests']))
positive_lookup = dict(zip(test_pop['Country'], test_pop['Positive']))
#positive_lookup = dict(zip(province_pop['Name'], province_pop['Population']))
# Remove the leaking data
train_full = train.copy()
valid = train[train['Date'] >= test['Date'].min()]
train = train[train['Date'] < test['Date'].min()]

# Split the test into public & private
test_public = test[test['Date'] <= DATE_BORDER]
test_private = test[test['Date'] > DATE_BORDER]

# Use a multi-index for easier slicing
train_full.set_index(['Area', 'Date'], inplace=True)
train.set_index(['Area', 'Date'], inplace=True)
valid.set_index(['Area', 'Date'], inplace=True)
test_public.set_index(['Area', 'Date'], inplace=True)
test_private.set_index(['Area', 'Date'], inplace=True)

submission['ConfirmedCases'] = 0
submission['Fatalities'] = 0

train_full.shape, train.shape, valid.shape, test_public.shape, test_private.shape, submission.shape


# The function below evaluates a model with a constant `R` number as well as `t_hosp`, `t_crit`, `m`, `c`, `f`,'p'

# In[ ]:


OPTIM_DAYS = 14  # Number of days to use for the optimisation evaluation


# In[ ]:


# Use a constant reproduction number
def eval_model_const(params, data, population, return_solution=False, forecast_days=0):
    R_0, t_hosp, t_crit, m, c, f ,p= params
    N = population
    n_infected = data['ConfirmedCases'].iloc[0]
    max_days = len(data) + forecast_days
    initial_state = [(N - n_infected)/ N, 0, n_infected / N, 0, 0, 0, 0,0]
    args = (R_0, 5.6, 2.9, t_hosp, t_crit, m, c, f,p)
               
    sol = solve_ivp(SEIR_COM_model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
    
    sus, exp, inf, rec, hosp, crit, deaths,positive = sol.y
    
    y_pred_cases = np.clip(inf + rec + hosp + crit + deaths + positive, 0, np.inf) * population
    y_true_cases = data['ConfirmedCases'].values
    y_pred_fat = np.clip(deaths, 0, np.inf) * population
    y_true_fat = data['Fatalities'].values
    
    optim_days = min(OPTIM_DAYS, len(data))  # Days to optimise for
    weights = 1 / np.arange(1, optim_days+1)[::-1]  # Recent data is more heavily weighted
    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[-optim_days:], weights)
    msle_fat = mean_squared_log_error(y_true_fat[-optim_days:], y_pred_fat[-optim_days:], weights)
    
    msle_final = np.mean([msle_cases, msle_fat])
    
    if return_solution:
        return msle_final, sol
    else:
        return msle_final


# The function below is essentially the same as above, by R is decayed using a Hill decay function. This model requires 2 additional parameters to be optimized, `k` & `L`

# In[ ]:


# Use a Hill decayed reproduction number
def eval_model_decay(params, data, population, return_solution=False, forecast_days=0):
    R_0, t_hosp, t_crit, m, c, f, p, k, L = params  
    N = population
    n_infected = data['ConfirmedCases'].iloc[0]
    max_days = len(data) + forecast_days
    
    # https://github.com/SwissTPH/openmalaria/wiki/ModelDecayFunctions   
    # Hill decay. Initial values: R_0=2.2, k=2, L=50
    def time_varying_reproduction(t): 
        return R_0 / (1 + (t/L)**k)
    
    initial_state = [(N - n_infected)/ N, 0, n_infected / N, 0, 0, 0, 0,0]
    args = (time_varying_reproduction, 5.6, 2.9, t_hosp, t_crit, m, c, f,p)
            
    sol = solve_ivp(SEIR_COM_model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))
    
    sus, exp, inf, rec, hosp, crit, deaths,postive = sol.y
    
    y_pred_cases = np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population
    y_true_cases = data['ConfirmedCases'].values
    y_pred_fat = np.clip(deaths, 0, np.inf) * population
    y_true_fat = data['Fatalities'].values
    
    optim_days = min(OPTIM_DAYS, len(data))  # Days to optimise for
    weights = 1 / np.arange(1, optim_days+1)[::-1]  # Recent data is more heavily weighted
    
    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[-optim_days:], weights)
    msle_fat = mean_squared_log_error(y_true_fat[-optim_days:], y_pred_fat[-optim_days:], weights)
    msle_final = np.mean([msle_cases, msle_fat])
    
    if return_solution:
        return msle_final, sol
    else:
        return msle_final


# In[ ]:


def use_last_value(train_data, valid_data, test_data):
    lv = train_data[['ConfirmedCases', 'Fatalities']].iloc[-1].values
    
    forecast_ids = test_data['ForecastId']
    submission.loc[forecast_ids, ['ConfirmedCases', 'Fatalities']] = lv
    
    if valid_data is not None:
        y_pred_valid = np.ones((len(valid_data), 2)) * lv.reshape(1, 2)
        y_true_valid = valid_data[['ConfirmedCases', 'Fatalities']]

        msle_cases = mean_squared_log_error(y_true_valid['ConfirmedCases'], y_pred_valid[:, 0])
        msle_fat = mean_squared_log_error(y_true_valid['Fatalities'], y_pred_valid[:, 1])
        msle_final = np.mean([msle_cases, msle_fat])

        return msle_final


# In[ ]:


def plot_model_results(y_pred, train_data, valid_data=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
    
    ax1.set_title('Confirmed Cases')
    ax2.set_title('Fatalities')
    
    train_data['ConfirmedCases'].plot(label='Confirmed Cases (train)', color='g', ax=ax1)
    y_pred.loc[train_data.index, 'ConfirmedCases'].plot(label='Modeled Cases', color='r', ax=ax1)
    ax3 = y_pred['R'].plot(label='Reproduction number', color='c', linestyle='-', secondary_y=True, ax=ax1)
    ax3.set_ylabel("Reproduction number", fontsize=10, color='c');
        
    train_data['Fatalities'].plot(label='Fatalities (train)', color='g', ax=ax2)
    y_pred.loc[train_data.index, 'Fatalities'].plot(label='Modeled Fatalities', color='r', ax=ax2)
    
    if valid_data is not None:
        valid_data['ConfirmedCases'].plot(label='Confirmed Cases (valid)', color='g', linestyle=':', ax=ax1)
        valid_data['Fatalities'].plot(label='Fatalities (valid)', color='g', linestyle=':', ax=ax2)
        y_pred.loc[valid_data.index, 'ConfirmedCases'].plot(label='Modeled Cases (forecast)', color='r', linestyle=':', ax=ax1)
        y_pred.loc[valid_data.index, 'Fatalities'].plot(label='Modeled Fatalities (forecast)', color='r', linestyle=':', ax=ax2)
    else:
        y_pred.loc[:, 'ConfirmedCases'].plot(label='Modeled Cases (forecast)', color='r', linestyle=':', ax=ax1)
        y_pred.loc[:, 'Fatalities'].plot(label='Modeled Fatalities (forecast)', color='r', linestyle=':', ax=ax2)
        
    ax1.legend(loc='best')
    


# The function below fits a SEIR-COMPARTMENTS model for each area, either using a constant R or a decayed R, whichever is better. If the total cases/1M pop is below 1 and social distance large than 100cm, then the last value is used.

# In[ ]:


def fit_model_public(area_name, 
                     initial_guess=[3.6, 4, 14, 0.8, 0.1, 0.3, 0.3,2, 50],
                     bounds=((1, 20), # R bounds
                             (0.5, 10), (2, 20), # transition time param bounds
                             (0.5, 1), (0, 1), (0, 1),(0, 1), (1, 10), (1, 100)), # fraction time param bounds
                     make_plot=True):
        
    train_data = train.loc[area_name].query('ConfirmedCases > 0')
    valid_data = valid.loc[area_name]
    test_data = test_public.loc[area_name]  
    
    try:
        population = province_lookup[area_name]
    except KeyError:
        population = country_lookup[area_name]
         
    try:
        distacnce = distance_lookup[area_name]
    except KeyError:
        distacnce = 100
    try:
        positive = tests_lookup[area_name]
    except KeyError:
        positive = 0.3
        
    cases_per_million = train_data['ConfirmedCases'].max() * 10**6 / population
    n_infected = train_data['ConfirmedCases'].iloc[0]
    
    if cases_per_million < 1:
        return use_last_value(train_data, valid_data, test_data)
    
                
    res_const = minimize(eval_model_const, initial_guess[:-2], bounds=bounds[:-2],
                         args=(train_data, population, False),
                         method='L-BFGS-B')
    
    res_decay = minimize(eval_model_decay, initial_guess, bounds=bounds,
                         args=(train_data, population, False),
                         method='L-BFGS-B')
    
    dates_all = train_data.index.append(test_data.index)
    dates_val = train_data.index.append(valid_data.index)
    
    
    # If using a constant R number is better, use that model
    if res_const.fun < res_decay.fun:
        msle, sol = eval_model_const(res_const.x, train_data, population, True, len(test_data))
        res = res_const
        R_t = pd.Series([res_const.x[0]] * len(dates_val), dates_val)
    else:
        msle, sol = eval_model_decay(res_decay.x, train_data, population, True, len(test_data))
        res = res_decay
        
        # Calculate the R_t values
        t = np.arange(len(dates_val))
        R_0, t_hosp, t_crit, m, c, f,p, k, L = res.x  
        R_t = pd.Series(R_0 / (1 + (t/L)**k), dates_val)
        
    sus, exp, inf, rec, hosp, crit, deaths, positive= sol.y
    
    y_pred = pd.DataFrame({
        'ConfirmedCases': np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population,
        'Fatalities': np.clip(deaths, 0, np.inf) * population,
        'R': R_t,
    }, index=dates_all)
    
    y_pred_valid = y_pred.iloc[len(train_data): len(train_data)+len(valid_data)]
    y_pred_test = y_pred.iloc[len(train_data):]
    y_true_valid = valid_data[['ConfirmedCases', 'Fatalities']]
        
    valid_msle_cases = mean_squared_log_error(y_true_valid['ConfirmedCases'], y_pred_valid['ConfirmedCases'])
    valid_msle_fat = mean_squared_log_error(y_true_valid['Fatalities'], y_pred_valid['Fatalities'])
    valid_msle = np.mean([valid_msle_cases, valid_msle_fat])
    
    if make_plot:
        print(f'Validation MSLE: {valid_msle:0.5f}')
        print(f'R: {res.x[0]:0.3f}, t_hosp: {res.x[1]:0.3f}, t_crit: {res.x[2]:0.3f}, '
              f'm: {res.x[3]:0.3f}, c: {res.x[4]:0.3f}, f: {res.x[5]:0.3f},p: {res.x[5]:0.3f}')
        plot_model_results(y_pred, train_data, valid_data)
        
    # Put the forecast in the submission
    forecast_ids = test_data['ForecastId']
    submission.loc[forecast_ids, ['ConfirmedCases', 'Fatalities']] = y_pred_test[['ConfirmedCases', 'Fatalities']].values
    
    return valid_msle
            


# In[ ]:


# Fit a model on the full dataset (i.e. no validation)
def fit_model_private(area_name, 
                      initial_guess=[3.6, 4, 14, 0.8, 0.1, 0.3, 0.3,2, 50],
                      bounds=((1, 20), # R bounds
                              (0.5, 10), (2, 20), # transition time param bounds
                              (0.5, 1), (0, 1), (0, 1), (0, 1),(1, 10), (1, 100)), # fraction time param bounds
                      make_plot=True):
        
    train_data = train_full.loc[area_name].query('ConfirmedCases > 0')
    test_data = test_private.loc[area_name]
    
    try:
        population = province_lookup[area_name]
    except KeyError:
        population = country_lookup[area_name]
        
    cases_per_million = train_data['ConfirmedCases'].max() * 10**6 / population
    n_infected = train_data['ConfirmedCases'].iloc[0]
        
    if cases_per_million < 1:
        return use_last_value(train_data, None, test_data)
                
    res_const = minimize(eval_model_const, initial_guess[:-2], bounds=bounds[:-2],
                         args=(train_data, population, False),
                         method='L-BFGS-B')
    
    res_decay = minimize(eval_model_decay, initial_guess, bounds=bounds,
                         args=(train_data, population, False),
                         method='L-BFGS-B')
    
    dates_all = train_data.index.append(test_data.index)
    
    
    # If using a constant R number is better, use that model
    if res_const.fun < res_decay.fun:
        msle, sol = eval_model_const(res_const.x, train_data, population, True, len(test_data))
        res = res_const
        R_t = pd.Series([res_const.x[0]] * len(dates_all), dates_all)
    else:
        msle, sol = eval_model_decay(res_decay.x, train_data, population, True, len(test_data))
        res = res_decay
        
        # Calculate the R_t values
        t = np.arange(len(dates_all))
        R_0, t_hosp, t_crit, m, c, f, p,k, L = res.x  
        R_t = pd.Series(R_0 / (1 + (t/L)**k), dates_all)
        
    sus, exp, inf, rec, hosp, crit, deaths,positive= sol.y
    
    y_pred = pd.DataFrame({
        'ConfirmedCases': np.clip(inf + rec + hosp + crit + deaths+positive, 0, np.inf) * population,
        'Fatalities': np.clip(deaths, 0, np.inf) * population,
        'R': R_t,
    }, index=dates_all)
    
    y_pred_test = y_pred.iloc[len(train_data):]
    
    if make_plot:
        print(f'R: {res.x[0]:0.3f}, t_hosp: {res.x[1]:0.3f}, t_crit: {res.x[2]:0.3f}, '
              f'm: {res.x[3]:0.3f}, c: {res.x[4]:0.3f}, f: {res.x[5]:0.3f},p: {res.x[5]:0.3f}')
        plot_model_results(y_pred, train_data)
        
    # Put the forecast in the submission
    forecast_ids = test_data['ForecastId']
    submission.loc[forecast_ids, ['ConfirmedCases', 'Fatalities']] = y_pred_test[['ConfirmedCases', 'Fatalities']].values
            


# # Calculate for all countries

# In[ ]:


# Public Leaderboard
validation_scores = []

for c in tqdm(test_public.index.levels[0].values):
    try:
        score = fit_model_public(c, make_plot=False)
        validation_scores.append({'Country': c, 'MSLE': score})
        print(f'{c} {score:0.5f}')
    except IndexError as e:
        print(c, 'has no cases in train')
    except ValueError as e:
        print(c, e)

validation_scores = pd.DataFrame(validation_scores)
print(f'Mean validation score: {np.sqrt(validation_scores["MSLE"].mean()):0.5f}')


# In[ ]:


# Find which areas are not being predicted well
validation_scores.sort_values(by=['MSLE'], ascending=False).head(20)


# In[ ]:


dir_output = '/kaggle/working/'
submission.round().to_csv('submission.csv')


# In[ ]:


submission.join(test.set_index('ForecastId')).query(f'Date > "{DATE_BORDER}"').round().to_csv('forecast.csv')


# In[ ]:




