#!/usr/bin/env python
# coding: utf-8

# # CORONAVIRUS OUTBREAK PREDICTION BY USING MACHINE LEARNING
# 
# 

# What's in it for you?
# - What is Corona Virus
# - How Covid-19 emerged
# - Symptoms of Corona Virus
# - Global Impact of Corona Virus
# - Corona Virus Outbreak Analysis
# - Safety Precautions

# - Coronaviruses(CoV) are a large family of viruses that cause illness ranging from common cold to more severe diseases such as Middle East Resipiratory Syndrome(MERS-CoV) and Severe Acute Respiratory Syndrome(SARS-CoV)
#   Coronaviruses are zoonotic,i.e it can be transmitted between animals and human beings.
# - COVID-19 is the disease caused by the new coronavirus that emerged in China in December 2019.
#   The source of coronavirus is believed to be a "wet market" in " Wuhan " which sold both dead and live animals incuding fish and birds.
# - COVID-19 symptoms include cough ,fever,shortness of breath,dry cough,headache,sore throat, and pneumonia.COVID-19 can be severe ,and some cases have caused death.
# - The Novel coronavirus is now a public health emergency of international concern,killing more than 11,000 people and infecting more than 200,000 people world wide.
# source follows: https://www.telegraph.co.uk/

# # ANALYSE DETAILS:
# We will analyse the outbreak of coronavirus across various regions,visualize them using charts and graphs and predict 
# the number of upcoming upcoming cases for the next 10 days using linear Regression and Support Vector Machine(SVM) model in Python.

# In[ ]:


# Importing all the important libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
import datetime
import operator
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# loading all the three datasets

confirmed_cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")


# In[ ]:


deaths_reported = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")


# In[ ]:


recovered_cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")


# In[ ]:


# display the head of the Dataset

confirmed_cases.head()


# In[ ]:


deaths_reported.head()


# In[ ]:


recovered_cases.head()


# In[ ]:


# Extracting all the columns using the .keys() function.
cols = confirmed_cases.keys()
cols


# In[ ]:


# Extracting only the dates columns that have information of confirmed,deaths and recovered cases.
confirmed =  confirmed_cases.loc[:, cols[4]:cols[-1]]


# In[ ]:


deaths = deaths_reported.loc[:, cols[4]:cols[-1]]


# In[ ]:


recoveries = recovered_cases.loc[:, cols[4]:cols[-1]]


# In[ ]:


# Check the head of the outbreak cases.
confirmed.head()


# In[ ]:


# Finding the total confirmed cases,death cases and the recovered cases and append them to an 4 empty lists.
# Also, calculate the total mortality rate which is the death sum/confirmed cases.

dates = confirmed.keys()
world_cases = []
total_deaths = []
mortality_rate = []
total_recovered = []

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)


# In[ ]:


# Lets display each of the newly created variables
confirmed_sum


# In[ ]:


death_sum


# In[ ]:


recovered_sum


# In[ ]:


world_cases


# In[ ]:


# Convert all the dates and the cases in the form of a numpy array

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases = np.array(world_cases).reshape(-1,1)
total_deaths = np.array(total_deaths).reshape(-1,1)
total_recovered = np.array(total_recovered).reshape(-1,1)


# In[ ]:


days_since_1_22


# In[ ]:


world_cases


# In[ ]:


total_deaths


# In[ ]:


total_recovered


# In[ ]:


# Future forecasting for the next 10 days

days_in_future = 10
future_forecast = np.array([i for i in range (len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast[:-10]


# In[ ]:


future_forecast


# In[ ]:


# Convert all the integers into datetime for better visualization.
start = '1/22/2020'
start_date = datetime.datetime.strptime(start,'%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forecast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# In[ ]:


# For visualization with the latest data of the 15th of march

latest_confirmed = confirmed_cases[dates[-1]]
latest_deaths = deaths_reported[dates[-1]]
latest_recoveries = recovered_cases[dates[-1]]


# In[ ]:


latest_confirmed


# In[ ]:


latest_deaths


# In[ ]:


latest_recoveries


# In[ ]:


# Find the list of unique countries
unique_countries = list(confirmed_cases['Country/Region'].unique())
unique_countries


# In[ ]:


# The next line of code will basically calculate the total number of confirmed cases by each country.

country_confirmed_cases = []
no_cases = []
for i in unique_countries:
    cases = latest_confirmed[confirmed_cases['Country/Region']==i].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_countries.remove(i)
    
unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()


# In[ ]:


# Number of cases per Country/Region
print('Confirmed Cases by Countries/Regions:')
for i in range(len(unique_countries)):
    print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')


# In[ ]:


# Find the list of unique provinces
unique_provinces =  list(confirmed_cases['Province/State'].unique())
# those are countries, which are not provinces/states.
outliers = ['United Kingdom', 'Denmark', 'France']
for i in outliers:
    unique_provinces.remove(i)


# In[ ]:


# Finding the number of confirmed cases per provinces,state or city.
province_confirmed_cases = []
no_cases = [] 
for i in unique_provinces:
    cases = latest_confirmed[confirmed_cases['Province/State']==i].sum()
    if cases > 0:
        province_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
# remove areas with no confirmed cases
for i in no_cases:
    unique_provinces.remove(i)


# In[ ]:


# Number of cases per province/state/city
for i in range(len(unique_provinces)):
    print(f'{unique_provinces[i]}: {province_confirmed_cases[i]} cases')


# In[ ]:


# Handling nan values if there is any,it is usually a float: float('nan')

nan_indices = [] 

for i in range(len(unique_provinces)):
    if type(unique_provinces[i]) == float:
        nan_indices.append(i)

unique_provinces = list(unique_provinces)
province_confirmed_cases = list(province_confirmed_cases)

for i in nan_indices:
    unique_provinces.pop(i)
    province_confirmed_cases.pop(i)


# In[ ]:


# Plot a bar graph to see the total confirmed cases across different countries.
plt.figure(figsize=(32,32))
plt.barh(unique_countries,country_confirmed_cases)
plt.title('Number of Covid-19 confirmed cases in Countries')
plt.xlabel('Number of Covid19 Confirmed cases')
plt.show()


# In[ ]:


# Plot a bar graph to see the total confirmed cases between mainland china and outside mainland china.
china_confirmed = latest_confirmed[confirmed_cases['Country/Region']=='China'].sum()
outside_mainland_china_confirmed = np.sum(country_confirmed_cases) - china_confirmed
plt.figure(figsize=(16, 9))
plt.barh('Mainland China', china_confirmed)
plt.barh('Outside Mainland China', outside_mainland_china_confirmed)
plt.title('Number of  Confirmed Coronavirus Cases')
plt.show()


# In[ ]:


# Print the total cases in mainland in China and outside of it.
print('Outside Mainland China {} cases:'.format(outside_mainland_china_confirmed))
print('Mainland China: {} cases'.format(china_confirmed))
print('Total: {} cases'.format(china_confirmed+outside_mainland_china_confirmed))


# In[ ]:


# Only show 10 countries with the most confirmed cases, the rest are grouped into the  category named others
visual_unique_countries = [] 
visual_confirmed_cases = []
others = np.sum(country_confirmed_cases[10:])
for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])

visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)


# # Visual Representations (bar charts and pie charts)
# 

# In[ ]:


# Visualize the 10 Countries
plt.figure(figsize=(32, 18))
plt.barh(visual_unique_countries, visual_confirmed_cases)
plt.title('Number of Covid-19 Confirmed Cases in Countries/Regions', size=20)
plt.show()


# In[ ]:


# Create a Pie chart to see the total confirmed cases in 10 different countries.

c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))
plt.figure(figsize=(20,20))
plt.title('Covid-19 Confirmed Cases per Country')
plt.pie(visual_confirmed_cases, colors=c)
plt.legend(visual_unique_countries, loc='best')
plt.show()


# In[ ]:


# Create a pie chart to see the total confirmed cases in 10 different countries outside China
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))
plt.figure(figsize=(20,20))
plt.title('Covid-19 Confirmed Cases in Countries Outside of Mainland China')
plt.pie(visual_confirmed_cases[1:], colors=c)
plt.legend(visual_unique_countries[1:], loc='best')
plt.show()


# In[ ]:


X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)


# In[ ]:


# Building the SVM Model
kernel = ['poly','sigmoid','rbf']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
epsilon = [0.01, 0.1, 1]
shrinking= [True, False]
svm_grid = {'kernel': kernel,'C': c,'gamma': gamma,'epsilon':epsilon,'shrinking':shrinking}
svm = SVR(kernel='poly')
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
svm_search.fit(X_train_confirmed, y_train_confirmed)


# In[ ]:


svm_search.best_params_


# In[ ]:


svm_confirmed = svm_search.best_estimator_
svm_pred = svm_confirmed.predict(future_forecast)


# In[ ]:


svm_confirmed


# In[ ]:


svm_pred


# In[ ]:


# check against testing data
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))


# In[ ]:


# Total number of coronavirus cases over time
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:


# Confirmed vs Predicted Cases
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forecast, svm_pred, linestyle='dashed', color='purple')
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:


# Prediction for the next 10 days using SVM
print('SVM future predictions:')
set(zip(future_forcast_dates[-10:], svm_pred[-10:]))


# In[ ]:


# Using Linear Regression Model to make Predictions
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(normalize = True,fit_intercept = True)
linear_model.fit(X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(X_test_confirmed)
linear_pred = linear_model.predict(future_forecast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))


# In[ ]:


plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)


# In[ ]:


# Graphing the number of confirmed cases, deaths, active cases, and the mortality rate over time, as well as the number of recoveries
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forecast, linear_pred, linestyle='dashed', color='orange')
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Confirmed Cases', 'Linear Regression Predictions'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:


# Prediction for the next 10 days using Linear Regression
print('Linear regression future predictions:')
print(linear_pred[-10:])


# In[ ]:


# Total Deaths over time
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_deaths, color='red')
plt.title('Number of Coronavirus Deaths Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Number of Deaths', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:


mean_mortality_rate = np.mean(mortality_rate)
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, mortality_rate, color='orange')
plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time', size=30)
plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)])
plt.xlabel('Time', size=30)
plt.ylabel('Mortality Rate', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:


# Coronavirus cases Recovered over Time

plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_recovered, color='green')
plt.title('Number of Coronavirus Cases Recovered Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:


# Number of Coronavirus cases recovered vs the number of deaths
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_deaths, color='r')
plt.plot(adjusted_dates, total_recovered, color='green')
plt.legend(['death', 'recoveries'], loc='best', fontsize=20)
plt.title('Number of Coronavirus Cases', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:


# Coronavirus Deaths vs Recoveries
plt.figure(figsize=(20, 12))
plt.plot(total_recovered, total_deaths)
plt.title('Number of Coronavirus Deaths vs. Number of Coronavirus Recoveries', size=30)
plt.xlabel('Total Number of Coronavirus Recoveries', size=30)
plt.ylabel('Total Number of Coronavirus Deaths', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:




