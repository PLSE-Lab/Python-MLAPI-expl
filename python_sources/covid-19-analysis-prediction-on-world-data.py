#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Analysis & Prediction on World and USA Data

# 
# This notebook visualizes and predicts the spread of the novel coronavirus, also known as SARS-CoV-2. It is a contagious respiratory virus that first started in Wuhan in December 2019. On 11 February 2020, the disease is officially named COVID-19 by the World Health Organization. 
# <br>
# 
# In this project, the COVID-19 data by Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) is used for analysis and visualizations.
# 
# <br>Data: <a href='https://github.com/CSSEGISandData/COVID-19'>https://github.com/CSSEGISandData/COVID-19</a>
# 
# Part of this code was used from https://www.kaggle.com/therealcyberlord/coronavirus-covid-19-visualization-prediction
#     
# Medium Link: https://medium.com/analytics-vidhya/covid-19-data-analysis-e9cb652e8c10    
# ***Note: This data is updated on a regular basis. For updated visualizations please run the notebook* <br>
# *Last Updated: 5th of May, 2020***

# **Import Required Libraries**

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import scipy as sp
import random
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ### Data Preparation

# **Import data**

# In[ ]:


confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-30-2020.csv')
us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/04-30-2020.csv')


# In[ ]:


latest_data.head()


# **View Data Samples**

# In[ ]:


confirmed_df.head()


# In[ ]:


deaths_df.head()


# In[ ]:


recoveries_df.head()


# In[ ]:


us_medical_data.head()


# In[ ]:


cols = confirmed_df.keys()


# **Get the dates and number of cases associated with that date**

# In[ ]:


confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]


# In[ ]:


dates = confirmed.keys()


# The next cell is for storing the data into different lists for making it easy to visualize. Of the different lists, one is for storing mortality rate and another is for storing recovery rate.
# 
# <br>Mortality rate can be defined as the ratio of number of deaths recorded against the total number of cases recorded and this is calculated using the following formula:<br>
# 
# $$mortality\;rate = \frac{no.\:of\:deaths}{no.\:of\:confirmed\:cases}$$
# 
# <br>Recovery rate can be defined as the ratio of number of recovered patients recorded against the total number of cases recorded and this is calculated using the following formula:<br>
# 
# $$recovery\:rate = \frac{no.\:of\:recovered\:cases}{no.\:of\:confirmed\:cases}$$

# In[ ]:


# storing world data
world_cases = [] # to store total cases
total_deaths = [] # to store total deaths
mortality_rate = [] # to store mortality rate
recovery_rate = [] # to store recovery rate
total_recovered = [] 
total_active = [] 

# storing US data
us_cases = [] 
us_deaths = [] 
us_recoveries = []


for i in dates:
    # calculate sums
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()

    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)

    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    # case studies 
    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())    
    us_deaths.append(deaths_df[deaths_df['Country/Region']=='US'][i].sum())
    us_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='US'][i].sum())


# Now that we have all the data of the recorded cases, let us calculate the day-wise change in count of cases, deaths, and recoveries. The function in the following cell helps to calculate this change

# In[ ]:


def get_daily_increase(data):
    '''
    INPUT - a list containing day by day case counts
    
    OUTPUT - a list containing the day by day increment of count 
    
    Function to count the daily increment in figures
    '''
    increment_count = [] 
    for i in range(len(data)):
        if i == 0:
            increment_count.append(data[0])
        else:
            increment_count.append(data[i]-data[i-1])
    return increment_count


# In[ ]:


# confirmed cases
world_daily_increase = get_daily_increase(world_cases)
us_daily_increase = get_daily_increase(us_cases)

# deaths
world_daily_death = get_daily_increase(total_deaths)
us_daily_death = get_daily_increase(us_deaths)

# recoveries
world_daily_recovery = get_daily_increase(total_recovered)
us_daily_recovery = get_daily_increase(us_recoveries)


# In[ ]:


# days from the first day in the dataset i.e. Jan 22, 2020 (1/22/2020)
days = np.array([i for i in range(len(dates))]).reshape(-1, 1)

# reshaping the data
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)


# **Days for Future Forecast**

# In[ ]:


num_days_future = 15
forecast_future = np.array([i for i in range(len(dates)+num_days_future)]).reshape(-1, 1)
adjusted_dates = forecast_future[:-15]


# The dates in dataset are in int64 format. Convert these into date-time format

# In[ ]:


start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
dates_for_forecast_future = []
for i in range(len(forecast_future)):
    dates_for_forecast_future.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# ### Visualizing the data

# In[ ]:


def plot_stats(x_data, y_data, title, y_label):
    '''
    INPUT - data to be plotted on X-axis and Y-axis and a strings for title, and y-axis label
    
    OUTPUT - function doesn't return anything but prints required plots
    '''
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel('Days Since Jan 22, 2020')
    plt.ylabel(y_label)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.show()


# In[ ]:


adjusted_dates = adjusted_dates.reshape(1, -1)[0]


# **World wide COVID-19 data trends**

# In[ ]:


plot_stats(adjusted_dates, world_cases,'Number of COVID-19 Cases Over Time','Number of Cases')


# In[ ]:


plot_stats(adjusted_dates, total_deaths,'Number of COVID-19 Deaths Over Time','Number of Deaths')


# In[ ]:


plot_stats(adjusted_dates, total_recovered, 'Number of COVID-19 Recoveries Over Time', 'Number of Recoveries')


# In[ ]:


plot_stats(adjusted_dates, total_active, 'Number of COVID-19 Active Cases Over Time','Number of  Active Cases')


# In[ ]:


plot_stats(adjusted_dates, world_daily_increase, 'Day-wise plot of Confirmed Cases - Worldwide', 'Number of Cases')


# In[ ]:


plot_stats(adjusted_dates, world_daily_death, 'Day-wise plot of Deaths - Worldwide','Number of Cases')


# In[ ]:


plot_stats(adjusted_dates, world_daily_recovery, 'Day-wise plot of Recoveries - Worldwide', 'Number of Cases')


# **USA wide data trends**

# In[ ]:


plot_stats(adjusted_dates,us_cases,'Number of Confirmed cases - USA','Number of Cases')


# In[ ]:


plot_stats(adjusted_dates,us_daily_increase,'Day-wise plot of Cases - USA','Number of Cases')


# In[ ]:


plot_stats(adjusted_dates,us_daily_death,'Day-wise plot of Deaths - USA','Number of Cases')


# In[ ]:


plot_stats(adjusted_dates,us_daily_recovery,'Day-wise plot of Recoveries - USA','Number of Cases')


# **State wise COVID-19 cases analysis**

# As we obtained and visualized the data of world and the US, the following cells are for plotting the region wise data

# In[ ]:


states =  list(latest_data.loc[latest_data['Country_Region'] == 'US','Province_State'].unique())


# In[ ]:


states_confirmed_cases = []
states_death_cases = [] 
states_recovery_cases = []
states_mortality_rate = [] 

no_cases = [] 
for state in states:
    cases = latest_data[latest_data['Province_State']==state]['Confirmed'].sum()
    if cases > 0:
        states_confirmed_cases.append(cases)
    else:
        no_cases.append(state)
 
# removing the areas with zero cases
if len(no_cases) != 0:
    for area in no_cases:
        states.remove(area)
    
states = [k for k, v in sorted(zip(states, states_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(states)):
    states_confirmed_cases[i] = latest_data[latest_data['Province_State']==states[i]]['Confirmed'].sum()
    states_death_cases.append(latest_data[latest_data['Province_State']==states[i]]['Deaths'].sum())
    states_recovery_cases.append(latest_data[latest_data['Province_State']==states[i]]['Recovered'].sum())
    states_mortality_rate.append(states_death_cases[i]/states_confirmed_cases[i])


# In[ ]:


# number of cases per US state
state_df = pd.DataFrame({'State': states, 'Number of Confirmed Cases': states_confirmed_cases,'Number of Deaths': states_death_cases, 'Mortality Rate': states_mortality_rate})

# number of cases per country/region
state_df.style.background_gradient(cmap='Oranges')


# **Plotting the 10 states with the most confirmed cases. The remaining states are grouped into "others" category**

# In[ ]:


def plot_bar_graphs(x_axis, y_axis, title):
    '''
    INPUT - variables to be plotted on X-axis and Y-axis, and title
    
    OUTPUT - function doesn't return anything but prints required plots
    '''
    plt.figure()
    plt.barh(x_axis, y_axis)
    plt.title(title)
    plt.show()


# In[ ]:


visual_unique_states = [] 
visual_confirmed_cases = []
others = np.sum(states_confirmed_cases[10:])
for i in range(len(states_confirmed_cases[:10])):
    visual_unique_states.append(states[i])
    visual_confirmed_cases.append(states_confirmed_cases[i])

visual_unique_states.append('Others')
visual_confirmed_cases.append(others)


# In[ ]:


plot_bar_graphs(visual_unique_states, visual_confirmed_cases, 'Number of COVID-19 Cases in States')


# **Plotting the same in a pie chart**

# In[ ]:


def plot_pie_charts(x_axis, y_axis, title):
    '''
    INPUT - variables to be plotted on X-axis and Y-axis, and title
    
    OUTPUT - function doesn't return anything but prints required plots
    
    '''
    plt.figure(figsize=(15,15))
    plt.title(title)
    plt.pie(y_axis, colors= random.choices(list(mcolors.CSS4_COLORS.values()),k = len(states)))
    plt.legend(x_axis, loc='best')
    plt.show()


# In[ ]:


plot_pie_charts(visual_unique_states, visual_confirmed_cases, 'Number of COVID-19 Cases in States of USA')


# **Plot Worldwide Mortality and Recovery Rates**

# In[ ]:


def plot_rates(x_axis,y_axis,title,y_label,color):
    '''
    INPUT - data to be plotted on X-axis and Y-axis and a strings for title, legend, and y-axis label
    
    OUTPUT - function doesn't return anything but prints required plots
    '''
    mean_rate = np.mean(y_axis)
    plt.plot(x_axis, y_axis, color = color)
    plt.axhline(y = mean_rate,linestyle='--', color='black')
    plt.title(title)
    plt.legend([y_label, 'y='+str(mean_rate)])
    plt.xlabel('Days Since Jan 22,2020')
    plt.ylabel(y_label)
    plt.show()


# In[ ]:


plot_rates(adjusted_dates, mortality_rate, 'Mortality Rate of COVID-19 patients Over Time', 'Mortality Rate','red')


# In[ ]:


plot_rates(adjusted_dates, recovery_rate, 'Recovery Rate of COVID-19 patients Over Time', 'Recovery Rate','green')


# **Split Data**
# <br>
# Now I'm predicting the number of confirmed cases for the next 15 days. For this first, we have to split the data into train and test sets

# In[ ]:


X_confirmed_train, X_confirmed_test, y_confirmed_train, y_confirmed_test = train_test_split(days, world_cases, test_size=0.30, shuffle=False) 


# **Models for predicting confirmed cases**<br> I used variants of LinearRegression to compare the performance
# <br>*Note that this is just a simple model and the results are not accurate*

# In[ ]:


# define linear regression model
linear_model = LinearRegression(normalize=True, fit_intercept=False)


# In[ ]:


# function to train the model and predict the number of probable cases
def execute_linear_model(X_confirmed_train, y_confirmed_train, X_confirmed_test, forecast_future):
    '''
    INPUT - the split data
    
    OUTPUT - executes linear model and prints MAE & MSE, returns predicted values
    '''
    linear_model.fit(X_confirmed_train, y_confirmed_train)
    test_linear_pred = linear_model.predict(X_confirmed_test)
    linear_pred = linear_model.predict(forecast_future)
    print('MAE:', mean_absolute_error(test_linear_pred, y_confirmed_test))
    print('MSE:',mean_squared_error(test_linear_pred, y_confirmed_test))
    return test_linear_pred, linear_pred


# In[ ]:


# function to plot the models predictions against the test data
def plot_model_predictions(model,y_confirmed_test,test_linear_pred):
    '''
    INPUT - model name, predicted and actual lables
    
    OUTPUT - displays graphs
    '''
    plt.plot(y_confirmed_test)
    plt.plot(test_linear_pred)
    plt.legend(['Test Data',  model+' Regression Predictions'])


# In[ ]:


test_linear_pred, linear_pred = execute_linear_model(X_confirmed_train, y_confirmed_train,X_confirmed_test,forecast_future)


# In[ ]:


plot_model_predictions('Linear',y_confirmed_test,test_linear_pred)


# It can be seen that normal LinearRegression did not work well with the data. Try using Polynomial Regression. <br> For that first transform the data

# In[ ]:


# transform our data for polynomial variants of linear regression
def transform_to_poly(degree=2):
    '''
    INPUT - degree 
    
    OUTPUT - returns the transformed data
    
    Transorms data into required polymonial degree
    '''
    poly = PolynomialFeatures(degree)
    poly_X_confirmed_train = poly.fit_transform(X_confirmed_train)
    poly_X_confirmed_test = poly.fit_transform(X_confirmed_test)
    poly_forecast_future = poly.fit_transform(forecast_future)
    
    return poly_X_confirmed_train, poly_X_confirmed_test, poly_forecast_future


# In[ ]:


poly_2_X_confirmed_train, poly_2_X_confirmed_test, poly_2_forecast_future = transform_to_poly(degree = 2)


# In[ ]:


test_ploy_2_pred, poly_2_pred = execute_linear_model(poly_2_X_confirmed_train,y_confirmed_train,poly_2_X_confirmed_test,poly_2_forecast_future)


# In[ ]:


plot_model_predictions('Polynomial Degree 2',y_confirmed_test,test_ploy_2_pred)


# In[ ]:


poly_3_X_confirmed_train, poly_3_X_confirmed_test, poly_3_forecast_future = transform_to_poly(degree = 3)


# In[ ]:


test_ploy_3_pred, poly_3_pred = execute_linear_model(poly_3_X_confirmed_train,y_confirmed_train,poly_3_X_confirmed_test,poly_3_forecast_future)


# In[ ]:


plot_model_predictions('Polynomial Degree 3',y_confirmed_test,test_ploy_3_pred)


# **Average of predictions of Polynomial Regression models**

# In[ ]:


test_avg_pred = []
for i in range(0,len(test_ploy_2_pred)):
    temp = float(test_ploy_2_pred[i]+test_ploy_3_pred[i])/2
    test_avg_pred.append(temp)


# In[ ]:


plot_model_predictions('Average of Polynomial Degree 2 & 3',y_confirmed_test,test_avg_pred)


# In[ ]:


print('MAE:', mean_absolute_error(test_avg_pred, y_confirmed_test))
print('MSE:',mean_squared_error(test_avg_pred, y_confirmed_test))


# It is clear that the average of predictions of Polynomial Regression models of Degree-2&3 has less MAE and MSE compared to those of the individual models.

# In[ ]:


preds = []
for i in range(0,len(poly_2_pred)):
    temp = float(poly_2_pred[i]+poly_3_pred[i])/2
    preds.append(temp)


# Now, we shall plot predictions of the 3 variants against the confirmed cases

# In[ ]:


def plot_future_predictions(x_data, y_data, predictions, algorithms):
    '''
    INPUT - 
        x_data & y_data: data for x-axis and y-axis
        predictions1,predictions2,predictions3: predictions by te above declared models
        algo_1_name,algo_2_name,algo_3_name: names for algorithms to show in graph legend
        
    OUTPUT - 
        This function doesn't return anything but prints the predictions of models against current confirmed cases
    '''
    
    plt.figure(figsize=(12, 9))
    plt.plot(x_data, y_data)
    for prediction in predictions:
        plt.plot(forecast_future, prediction, linestyle='dashed')
    plt.title('Number of COVID-19 Cases Over Time')
    plt.xlabel('Days Since Jan 22, 2020')
    plt.ylabel('Number of Cases')
    legend = ['Confirmed Cases']
    for algorithm in algorithms:
        legend.append(algorithm)
    plt.legend(legend)
    plt.ticklabel_format(style = 'plain')
    plt.show()


# In[ ]:


pred = [linear_pred, poly_2_pred, poly_3_pred, preds]
algos = ['Linear Regression','Degree 2 Polynomial Regression','Degree 3 Polynomial Regression','Average of Polynomial Models']
plot_future_predictions(adjusted_dates, world_cases,pred,algos)


# It can be seen that the Polynomial variant with degree three performed well compared to other two variants of the Linear Regression model.
# 
# <br> 
# 
# The next cell is for displaying our predictions for the next 15 days that we obtained using the Polynomial model.
# <br>*Note that this is just a simple model and the results are not accurate*

# In[ ]:


poly_3_preds = poly_3_pred.reshape(1,-1)[0]
poly_3_df = pd.DataFrame({'Date': dates_for_forecast_future[-15:], 'Number of worldwide cases predicted': np.round(poly_3_preds[-15:])})
poly_3_df


# In[ ]:


preds_m = np.array(preds).reshape(1,-1)[0]
poly_df = pd.DataFrame({'Date': dates_for_forecast_future[-15:], 'Number of worldwide cases predicted': np.round(preds[-15:])})
poly_df


# It can be seen that the aerage of predictions from Degree-2 and Degree-3 Polynomial regression models showed better results
