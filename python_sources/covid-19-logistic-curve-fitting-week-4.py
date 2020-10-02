#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


area_data = train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv')
area_data['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(area_data['Country.Region'], area_data['Province.State'])]
area_data.head()


# In[ ]:


path = "/kaggle/input/covid19-global-forecasting-week-4/"
train_data = pd.read_csv(path+"train.csv")
train_df = train_data
train_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(train_data['Country_Region'], train_data['Province_State'])]
train_df['Date'] = pd.to_datetime(train_df['Date'])
full_data = train_df
full_data.head()


# In[ ]:


today = full_data['Date'].max()+timedelta(days=1) 
# remove date leakage
#today = '2020-03-26'
#train_df = train_df[train_df['Date']<pd.to_datetime(today)]
#train_df.head()


# In[ ]:


def get_country_data(train_df, area, metric):
    country_data = train_df[train_df['area']==area]
    country_data = country_data.drop(['Id','Province_State', 'Country_Region'], axis=1)
    country_data = pd.pivot_table(country_data, values=['ConfirmedCases','Fatalities'], index=['Date'], aggfunc=np.sum) 
    country_data = country_data[country_data[metric]!=0]
    return country_data        


# In[ ]:


area_info = pd.DataFrame(columns=['area', 'cases_start_date', 'deaths_start_date', 'init_ConfirmedCases', 'init_Fatalities'])
for i in range(len(train_df['area'].unique())):
    area = train_df['area'].unique()[i]
    area_cases_data = get_country_data(train_df, area, 'ConfirmedCases')
    area_deaths_data = get_country_data(train_df, area, 'Fatalities')
    cases_start_date = area_cases_data.index.min()
    deaths_start_date = area_deaths_data.index.min()
    if len(area_cases_data) > 0:
        confirmed_cases = max(area_cases_data['ConfirmedCases'])
    else:
        confirmed_cases = 0
    if len(area_deaths_data) > 0:
        fatalities = max(area_deaths_data['Fatalities'])
    else:
        fatalities = 0
    area_info.loc[i] = [area, cases_start_date, deaths_start_date, confirmed_cases, fatalities]
area_info = area_info.fillna(pd.to_datetime(today))
area_info['init_cases_day_no'] = pd.to_datetime(today)-area_info['cases_start_date']
area_info['init_cases_day_no'] = area_info['init_cases_day_no'].dt.days.fillna(0).astype(int)
area_info['init_deaths_day_no'] = pd.to_datetime(today)-area_info['deaths_start_date']
area_info['init_deaths_day_no'] = area_info['init_deaths_day_no'].dt.days.fillna(0).astype(int)
area_info.head()


# In[ ]:


def log_curve(x, k, x_0, ymax):
    return ymax / (1 + np.exp(-k*(x-x_0)))

#test
#log_curve(20, 0.220384, 62.014271, 159970.890625)

def log_fit(train_df, area, metric):
    area_data = get_country_data(train_df, area, metric)
    x_data = range(len(area_data.index))
    y_data = area_data[metric]
    if len(y_data) < 5:
        estimated_k = -1  
        estimated_x_0 = -1 
        ymax = -1
    elif max(y_data) == 0:
        estimated_k = -1  
        estimated_x_0 = -1 
        ymax = -1
    else:
        try:
            popt, pcov = curve_fit(log_curve, x_data, y_data, bounds=([0,0,0],np.inf), p0=[0.3,100,10000], maxfev=1000000)
            estimated_k, estimated_x_0, ymax = popt
        except RuntimeError:
            print(area)
            print("Error - curve_fit failed") 
            estimated_k = -1  
            estimated_x_0 = -1 
            ymax = -1
    estimated_parameters = pd.DataFrame(np.array([[area, estimated_k, estimated_x_0, ymax]]), columns=['area', 'k', 'x_0', 'ymax'])
    return estimated_parameters


# In[ ]:


def get_parameters(metric):
    parameters = pd.DataFrame(columns=['area', 'k', 'x_0', 'ymax'], dtype=np.float)
    for area in train_df['area'].unique():
        estimated_parameters = log_fit(train_df, area, metric)
        parameters = parameters.append(estimated_parameters)
    parameters['k'] = pd.to_numeric(parameters['k'], downcast="float")
    parameters['x_0'] = pd.to_numeric(parameters['x_0'], downcast="float")
    parameters['ymax'] = pd.to_numeric(parameters['ymax'], downcast="float")
    parameters = parameters.replace({'k': {-1: parameters[parameters['ymax']>0].median()[0]}, 
                                     'x_0': {-1: parameters[parameters['ymax']>0].median()[1]}, 
                                     'ymax': {-1: parameters[parameters['ymax']>0].median()[2]}})
    return parameters


# In[ ]:


cases_parameters = get_parameters('ConfirmedCases')
cases_parameters.head(20)


# In[ ]:


deaths_parameters = get_parameters('Fatalities')
deaths_parameters.head(20)


# In[ ]:


# Adjust the fit: the fit can put the inflection far too early - this adjusts the parameters to rectify that

# get completion benchmark from 'China - Hubei' data
hubei_results = area_info.merge(area_data[area_data['area']=='China - Hubei'], on='area', how='inner')
hubei_results['ConfirmedCases_perc'] = 100*hubei_results['init_ConfirmedCases']/hubei_results['Population']
hubei_results['Fatalities_perc'] = 100*hubei_results['init_Fatalities']/hubei_results['Population']
hubei_ConfirmedCases_prop = hubei_results['ConfirmedCases_perc'][0]/100
hubei_Fatalities_prop = hubei_results['Fatalities_perc'][0]/100

# adjust cases parameters based on benchmark
adj_cases_parameters = cases_parameters.merge(area_data, on='area', how='left')
adj_cases_parameters['ymax'] = [min(i,j) for i, j in zip(adj_cases_parameters['ymax'],adj_cases_parameters['Population'])]
adj_cases_parameters['ymax_benchmark'] = hubei_ConfirmedCases_prop*adj_cases_parameters['Population']
adj_cases_parameters.loc[adj_cases_parameters['ymax'] < adj_cases_parameters['ymax_benchmark']*0.01, 'k'] = adj_cases_parameters['k']/(adj_cases_parameters['ymax_benchmark']*0.01/adj_cases_parameters['ymax'])
adj_cases_parameters.loc[adj_cases_parameters['ymax'] < adj_cases_parameters['ymax_benchmark']*0.01, 'x_0'] = adj_cases_parameters['x_0']*adj_cases_parameters['ymax_benchmark']*0.01/adj_cases_parameters['ymax']
adj_cases_parameters.loc[adj_cases_parameters['ymax'] < adj_cases_parameters['ymax_benchmark']*0.01, 'ymax'] = adj_cases_parameters['ymax_benchmark']*0.01
adj_cases_parameters = adj_cases_parameters[['area', 'k', 'x_0', 'ymax', 'ymax_benchmark']]

# adjust deaths parameters based on benchmark
adj_deaths_parameters = deaths_parameters.merge(area_data, on='area', how='left')
adj_deaths_parameters['ymax'] = [min(i,j) for i, j in zip(adj_deaths_parameters['ymax'],adj_deaths_parameters['Population'])]
adj_deaths_parameters['ymax_benchmark'] = hubei_Fatalities_prop*adj_deaths_parameters['Population']
adj_deaths_parameters.loc[adj_deaths_parameters['ymax'] < adj_deaths_parameters['ymax_benchmark']*0.01, 'k'] = adj_deaths_parameters['k']/(adj_deaths_parameters['ymax_benchmark']*0.01/adj_deaths_parameters['ymax'])
adj_deaths_parameters.loc[adj_deaths_parameters['ymax'] < adj_deaths_parameters['ymax_benchmark']*0.01, 'x_0'] = adj_deaths_parameters['x_0']*adj_deaths_parameters['ymax_benchmark']*0.01/adj_deaths_parameters['ymax']
adj_deaths_parameters.loc[adj_deaths_parameters['ymax'] < adj_deaths_parameters['ymax_benchmark']*0.01, 'ymax'] = adj_deaths_parameters['ymax_benchmark']*0.01
adj_deaths_parameters = adj_deaths_parameters[['area', 'k', 'x_0', 'ymax', 'ymax_benchmark']]


# In[ ]:


fit_df = area_info.merge(adj_cases_parameters, on='area', how='left')
fit_df = fit_df.rename(columns={"k": "cases_k", "x_0": "cases_x_0", "ymax": "cases_ymax", "ymax_benchmark": "cases_ymax_benchmark"})
fit_df = fit_df.merge(adj_deaths_parameters, on='area', how='left')
fit_df = fit_df.rename(columns={"k": "deaths_k", "x_0": "deaths_x_0", "ymax": "deaths_ymax", "ymax_benchmark": "deaths_ymax_benchmark"})
fit_df['init_ConfirmedCases_fit'] = log_curve(fit_df['init_cases_day_no'], fit_df['cases_k'], fit_df['cases_x_0'], fit_df['cases_ymax'])
fit_df['init_Fatalities_fit'] = log_curve(fit_df['init_deaths_day_no'], fit_df['deaths_k'], fit_df['deaths_x_0'], fit_df['deaths_ymax'])
fit_df['ConfirmedCases_error'] = fit_df['init_ConfirmedCases']-fit_df['init_ConfirmedCases_fit']
fit_df['Fatalities_error'] = fit_df['init_Fatalities']-fit_df['init_Fatalities_fit']
fit_df.head()


# In[ ]:


test_data = pd.read_csv(path+"test.csv")
test_df = test_data
test_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(test_data['Country_Region'], test_data['Province_State'])]

test_df = test_df.merge(fit_df, on='area', how='left')

test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df['cases_start_date'] = pd.to_datetime(test_df['cases_start_date'])
test_df['deaths_start_date'] = pd.to_datetime(test_df['deaths_start_date'])

test_df['cases_day_no'] = test_df['Date']-test_df['cases_start_date']
test_df['cases_day_no'] = test_df['cases_day_no'].dt.days.fillna(0).astype(int)
test_df['deaths_day_no'] = test_df['Date']-test_df['deaths_start_date']
test_df['deaths_day_no'] = test_df['deaths_day_no'].dt.days.fillna(0).astype(int)

test_df['ConfirmedCases_fit'] = log_curve(test_df['cases_day_no'], test_df['cases_k'], test_df['cases_x_0'], test_df['cases_ymax'])
test_df['Fatalities_fit'] = log_curve(test_df['deaths_day_no'], test_df['deaths_k'], test_df['deaths_x_0'], test_df['deaths_ymax'])

test_df['ConfirmedCases_pred'] = round(test_df['ConfirmedCases_fit']+test_df['ConfirmedCases_error'])
test_df['Fatalities_pred'] = round(test_df['Fatalities_fit']+test_df['Fatalities_error'])

test_df.head()


# In[ ]:


# generate submission
submission = pd.DataFrame(data={'ForecastId': test_df['ForecastId'], 'ConfirmedCases': test_df['ConfirmedCases_pred'], 'Fatalities': test_df['Fatalities_pred']}).fillna(0)
submission['ConfirmedCases'] = submission['ConfirmedCases'].clip(lower=0)
submission['Fatalities'] = submission['Fatalities'].clip(lower=0)
submission.to_csv("/kaggle/working/submission.csv", index=False)


# In[ ]:


# 20 weeks from 20th Jan 2020
predictions = pd.DataFrame(columns = ['area', 'date', 'ConfirmedCases', 'Fatalities', 'Fitted Cases', 'Fitted Fatalities'])
start = pd.to_datetime('20/01/2020')
numdays = 20*7
date_list = [start + datetime.timedelta(days=x) for x in range(numdays)]
for area in ['United Kingdom - nan', 'Italy - nan', 'Iran - nan', 'China - Hubei']: #train_df['area'].unique():    
    cases_data = get_country_data(train_df, area, 'ConfirmedCases')
    offset = (area_info[area_info['area']==area]['cases_start_date'] - start).dt.days.values[0]
    x_data_fit = [i-offset for i in range(numdays)]
    x_data_real = range(1,len(country_data.index)+1)
    parameters = adj_cases_parameters
    estimated_k = parameters[parameters['area']==area]['k']
    estimated_x_0 = parameters[parameters['area']==area]['x_0']
    estimated_ymax = parameters[parameters['area']==area]['ymax']
    cases_fitted = log_curve(x_data_fit, np.float64(estimated_k), np.float64(estimated_x_0), np.float64(estimated_ymax))
    
    deaths_data = get_country_data(train_df, area, 'Fatalities')
    offset = (area_info[area_info['area']==area]['deaths_start_date'] - start).dt.days.values[0]
    x_data_fit = [i-offset for i in range(numdays)]
    x_data_real = range(1,len(country_data.index)+1)
    parameters = adj_deaths_parameters
    estimated_k = parameters[parameters['area']==area]['k']
    estimated_x_0 = parameters[parameters['area']==area]['x_0']
    estimated_ymax = parameters[parameters['area']==area]['ymax']
    deaths_fitted = log_curve(x_data_fit, np.float64(estimated_k), np.float64(estimated_x_0), np.float64(estimated_ymax))
    
    area_predictions = pd.DataFrame(data={'area': [area]*numdays, 'date': date_list}) 
    area_predictions = area_predictions.merge(cases_data, left_on='date', right_index=True, how='left')
    area_predictions['Fitted Cases'] = cases_fitted
    area_predictions['Fitted Fatalities'] = deaths_fitted
    
    predictions = predictions.append(area_predictions)
    


# In[ ]:


# log fit test
x_data = range(numdays)
y_fitted = log_curve(x_data, np.float64(0.1), np.float64(numdays/2), np.float64(0.009))
#steepness (divide)  #inflection on x-axis (multiply)  #max on y-axis (multiply)
fig = plt.figure(figsize=(10,7))
plt.ylim(0,0.01)
plt.ylabel('percentage of population')
plt.xlabel('date')
ax = fig.add_subplot(111)
ax.plot(date_list, y_fitted, '-', label='COVID-19 Cases', color='b')
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

#ax.set_title('COVID-19 Cases')


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,15))
areas = ['United Kingdom - nan', 'Italy - nan', 'Iran - nan', 'China - Hubei']
axs = [ax1, ax2, ax3, ax4]
for i in range(len(areas)):
    area = areas[i]
    axs[i].set_title(area.replace("- nan", ""))
    area_predictions = predictions[predictions['area']==area]
    axs[i].plot(area_predictions['date'], area_predictions['Fitted Cases'], label='Fitted Cases', color='b')
    axs[i].plot(area_predictions['date'], area_predictions['ConfirmedCases'], 'o', label='Actual Cases', color='b')
    axs[i].plot(area_predictions['date'], area_predictions['Fitted Fatalities'], '-', label='Fitted Fatalities', color='r')
    axs[i].plot(area_predictions['date'], area_predictions['Fatalities'], 'o', label='Actual Fatalities', color='r')
    axs[i].legend()


# In[ ]:


## does not work if dates are missing ##
def plot_pred(train_df, area, parameters, metric):
    country_data = get_country_data(train_df, area, metric)
    x_data_fit = range(1,100)
    x_data_real = range(1,len(country_data.index)+1)
    y_data = country_data[metric]
    estimated_k = parameters[parameters['area']==area]['k']
    estimated_x_0 = parameters[parameters['area']==area]['x_0']
    estimated_ymax = parameters[parameters['area']==area]['ymax']
    y_fitted = log_curve(x_data_fit, np.float64(estimated_k), np.float64(estimated_x_0), np.float64(estimated_ymax))
    if metric == 'ConfirmedCases':
        y_error = fit_df[fit_df['area']==area]['ConfirmedCases_error'].values[0]
    elif metric == 'Fatalities':
        y_error = fit_df[fit_df['area']==area]['Fatalities_error'].values[0]
    y_adjusted = [i+y_error for i in y_fitted]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(area+' - '+metric)
    ax.plot(x_data_fit, y_fitted, '--', label='fitted')
    ax.plot(x_data_real, y_data, 'o', label=metric)


# In[ ]:


for area in ['China - Hubei', 'United Kingdom - nan', 'Italy - nan', 'Iran - nan']:    
    plot_pred(train_df, area, adj_cases_parameters, 'ConfirmedCases')


# In[ ]:


for area in adj_cases_parameters[['US' in i for i in adj_cases_parameters['area']]]['area']:    
    plot_pred(train_df, area, adj_cases_parameters, 'ConfirmedCases')

