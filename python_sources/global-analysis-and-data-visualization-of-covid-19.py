#!/usr/bin/env python
# coding: utf-8

# # Global Analysis and Data Visualization of COVID-19

# In[ ]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.available
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


# read data from Johns Hopkins github repo
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


# In[ ]:


# shape of dataframe
print(confirmed_df.shape)
print(deaths_df.shape)
print(recoveries_df.shape)


# In[ ]:


# first 5 rows
confirmed_df.head()


# In[ ]:


# checking null values
confirmed_df.isna().sum()


# In[ ]:


# check all unique values
confirmed_df.nunique()


# In[ ]:


# check all unique values
recoveries_df.nunique()


# In[ ]:


# value counts by country
confirmed_df['Country/Region'].value_counts()


# # Cleaning :

# ### 1) Rename columns 'Province/State' & 'Country/Region' & change latest date to 'Current'.  

# In[ ]:


col=confirmed_df.columns[-1]

confirmed_df.rename(columns = {'Province/State' : 'Province', 'Country/Region' : 'Country', col : 'Current'},inplace = True)
deaths_df.rename(columns = {'Province/State' : 'Province', 'Country/Region' : 'Country', col : 'Current'},inplace = True)
recoveries_df.rename(columns = {'Province/State' : 'Province', 'Country/Region' : 'Country', col : 'Current'},inplace = True)
confirmed_df.head(1)


# ### 2) Make new dataframe grouping by unique country

# In[ ]:


# confirmed cases
confirm = pd.DataFrame(confirmed_df.groupby('Country').sum())
confirm.reset_index(inplace = True)
confirm.head(2)


# In[ ]:


# drop Lat & Long columns as they do not give accurate results
col = confirm['Country']
confirm.drop(['Lat','Long'],axis=1,inplace=True)
confirm.head(2)


# In[ ]:


# deaths
deaths= pd.DataFrame(deaths_df.groupby('Country').sum())
deaths.reset_index(inplace = True)

# drop Lat & Long columns as they do not give accurate results
deaths.drop(['Lat','Long'],axis=1,inplace=True)
deaths.head(2)


# In[ ]:


# recovery
recovery= pd.DataFrame(recoveries_df.groupby('Country').sum())
recovery.reset_index(inplace = True)

# drop Lat & Long columns as they do not give accurate results
recovery.drop(['Lat','Long'],axis=1,inplace=True)
recovery.head(2)


# ### 3) Create new dataframe for active cases

# In[ ]:


# active cases dataframe
active= confirm.copy()
for i in active.columns[1:]:
    active[i] =active[i] - recovery[i] - deaths[i]
active.head()


# # Analysis :

# ### 1) Total Cases till date

# In[ ]:


print("Confirmed Cases :" , confirm.iloc[:,-1].sum())
print("Recovered Cases :" , recovery.iloc[:,-1].sum())
print("Deaths :" , deaths.iloc[:,-1].sum())
print("Active Cases :", active.iloc[:,-1].sum())


# ### 2) Top 10 countries

# In[ ]:


confirm_data = confirm[['Country','Current']].sort_values('Current',ascending = False)
deaths_data = deaths[['Country','Current']].sort_values('Current',ascending = False)
recovery_data = recovery[['Country','Current']].sort_values('Current',ascending = False)
active_data = active[['Country','Current']].sort_values('Current',ascending = False)


# In[ ]:


# Confirmed Cases
sns.set(font_scale=1.5)
plt.figure(figsize=(10,5))
fig= sns.barplot(x='Current', y='Country', data=confirm_data[:10], orient='h',color='Blue')
plt.title('Total Confirmed Cases Worldwide')
fig.set(xlabel ='Number of Cases', ylabel ='Country')
plt.show()

# Recovery Cases
plt.figure(figsize=(10,5))
fig= sns.barplot(x='Current', y='Country', data=recovery_data[:10], orient='h',color='Green')
plt.title('Total Recovered Cases Worldwide')
fig.set(xlabel ='Number of Cases', ylabel ='Country')
plt.show()

# Death Cases
plt.figure(figsize=(10,5))
fig= sns.barplot(x='Current', y='Country', data=deaths_data[:10], orient='h',color='Red')
plt.title('Total Deaths Worldwide')
fig.set(xlabel ='Number of Cases', ylabel ='Country')
plt.show()

# Active Cases
plt.figure(figsize=(10,5))
fig= sns.barplot(x='Current', y='Country', data=active_data[:10], orient='h',color='Yellow')
plt.title('Total Active Cases Worldwide')
fig.set(xlabel ='Number of Cases', ylabel ='Country')
plt.show()


# ### 3) Daily Cases for China, Italy, US, Russia, India, Brazil

# In[ ]:


china_confirm = confirm[confirm.Country == 'China'].iloc[:,1:].sum().values.tolist()
us_confirm = confirm[confirm.Country == 'US'].iloc[:,1:].sum().values.tolist()
italy_confirm = confirm[confirm.Country == 'Italy'].iloc[:,1:].sum().values.tolist()
india_confirm = confirm[confirm.Country == 'India'].iloc[:,1:].sum().values.tolist()
russia_confirm = confirm[confirm.Country == 'Russia'].iloc[:,1:].sum().values.tolist()
brazil_confirm = confirm[confirm.Country == 'Brazil'].iloc[:,1:].sum().values.tolist()

china_deaths = deaths[deaths.Country == 'China'].iloc[:,1:].sum().values.tolist()
us_deaths = deaths[deaths.Country == 'US'].iloc[:,1:].sum().values.tolist()
italy_deaths = deaths[deaths.Country == 'Italy'].iloc[:,1:].sum().values.tolist()
india_deaths = deaths[deaths.Country == 'India'].iloc[:,1:].sum().values.tolist()
russia_deaths = deaths[deaths.Country == 'Russia'].iloc[:,1:].sum().values.tolist()
brazil_deaths = deaths[confirm.Country == 'Brazil'].iloc[:,1:].sum().values.tolist()

china_recovery = recovery[recovery.Country == 'China'].iloc[:,1:].sum().values.tolist()
us_recovery = recovery[recovery.Country == 'US'].iloc[:,1:].sum().values.tolist()
italy_recovery = recovery[recovery.Country == 'Italy'].iloc[:,1:].sum().values.tolist()
india_recovery = recovery[recovery.Country == 'India'].iloc[:,1:].sum().values.tolist()
russia_recovery = recovery[recovery.Country == 'Russia'].iloc[:,1:].sum().values.tolist()
brazil_recovery = recovery[recovery.Country == 'Brazil'].iloc[:,1:].sum().values.tolist()


# In[ ]:


# Confirmed Cases
plt.figure(figsize=(16,9))
plt.plot(china_confirm)
plt.plot(italy_confirm)
plt.plot(us_confirm)
plt.plot(india_confirm)
plt.plot(russia_confirm)
plt.plot(brazil_confirm)

plt.title('Confirmed Cases Countrywise', size=25)
plt.xlabel('No. of Days from 1/22/2020', size=20)
plt.ylabel('No. of Cases', size=20)
plt.legend(['China', 'Italy','US','India','Russia','Brazil'])
plt.show()

# Recovered Cases
plt.figure(figsize=(16, 9))
plt.plot(china_recovery)
plt.plot(italy_recovery)
plt.plot(us_recovery)
plt.plot(india_recovery)
plt.plot(russia_recovery)
plt.plot(brazil_recovery)

plt.title('Recovered Cases Countrywise', size=25)
plt.xlabel('No. of Days from 1/22/2020', size=20)
plt.ylabel('No. of Cases', size=20)
plt.legend(['China', 'Italy','US','India','Russia','Brazil'])
plt.show()

# Deaths
plt.figure(figsize=(16, 9))
plt.plot(china_deaths)
plt.plot(italy_deaths)
plt.plot(us_deaths)
plt.plot(india_deaths)
plt.plot(russia_deaths)
plt.plot(brazil_deaths)

plt.title('Deaths Countrywise', size=25)
plt.xlabel('No. of Days 1/22/2020', size=20)
plt.ylabel('No. of Cases', size=20)
plt.legend(['China', 'Italy','US','India','Russia','Brazil'])
plt.show()


# ### Observations :   
# * Although COVID-19originated from China, the number of cases are very less compared to other countries.
# * Number of cases and deaths in USA has been increasing exponentially.    
# * Initially the cases and deaths in Italy were increasing but currently the situation seems under control.     
# * Number of cases and deaths are comparatively less in India.

# ### 4) Increase in Cases Worldwide from 22/1/20

# In[ ]:


confirm_date = confirm.iloc[:,1:].sum().values.tolist()
recovery_date = recovery.iloc[:,1:].sum().values.tolist()
deaths_date = deaths.iloc[:,1:].sum().values.tolist()
active_date = active.iloc[:,1:].sum().values.tolist()


# In[ ]:


plt.figure(figsize=(16,9))
plt.plot(confirm_date,color='Blue')
plt.plot(recovery_date,color='Green')
plt.plot(deaths_date,color='Red')
plt.plot(active_date,color='Yellow')

plt.xlabel('No. of Days from 22/1/2020',size=25)
plt.ylabel('No. of Cases',size=25)
plt.title('Increase in number of Cases',size=25)
plt.legend(['Confirmed','Recovery','Deaths','Active'])
plt.show()


# In[ ]:


days = [ i for i in range(confirm.shape[1] - 1) ] 

plt.figure(figsize=(10,6))
plt.bar(days,confirm_date,color='Blue')
plt.title('Confirmed Cases Worldwide',size=20)
plt.show()

plt.figure(figsize=(10,6))
plt.bar(days,recovery_date,color='Green')
plt.title('Recovered Cases Worldwide',size=20)
plt.show()

plt.figure(figsize=(10,6))
plt.bar(days,active_date,color='Yellow')
plt.title('Active Cases Worldwide',size=20)
plt.show()

plt.figure(figsize=(10,6))
plt.bar(days,deaths_date,color='Red')
plt.title('Death Cases Worldwide',size=20)
plt.show()


# ### Observations :
# * A sharp rise in number of confirmed cases can be seen after 2 months of origin of coronavirus.   
# * The number of deaths are comparatively very less compared to the confirmed cases.    
# * The number of active cases are more compared to recovered cases.

# # Linear Regression :

# In[ ]:


total_confirm = np.array(confirm_date).reshape(-1,1)
total_deaths = np.array(deaths_date).reshape(-1,1)
total_recovery = np.array(recovery_date).reshape(-1,1)
total_active = np.array(active_date).reshape(-1,1)
dates = np.array([i for i in range(len(days))]).reshape(-1, 1)


# In[ ]:


# Visualization function
def linear_plot(x,y,reg,title):
    plt.figure(figsize=(10,6)) 
    plt.scatter(x,y,color='red')
    plt.plot(x,reg)
    plt.title(title)


# In[ ]:


X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(dates[50:], total_confirm[50:], test_size=0.14, shuffle=False) 


# In[ ]:


reg = LinearRegression()
reg.fit(X_train_confirmed, y_train_confirmed);


# In[ ]:


# Plot training set
linear_plot(X_train_confirmed,y_train_confirmed,reg.predict(X_train_confirmed),'Predicting Confirmed Cases Worldwide (Training Set)')

# Plot test set
linear_plot(X_test_confirmed,y_test_confirmed,reg.predict(X_test_confirmed),'Predicting Confirmed Cases Worldwide (Test Set)')


# The test set predictions are not very accurate as training set predictions.      
# As the total confirmed cases has a parabolic curve, trying polynomial linear regression.

# # Polynomial Linear Regression 

# ### Confirmed Cases

# In[ ]:


# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)


# In[ ]:


# polynomial regression
poly_reg = LinearRegression(normalize=True, fit_intercept=False)
poly_reg.fit(poly_X_train_confirmed, y_train_confirmed)


# In[ ]:


# Plot training set
linear_plot(X_train_confirmed,y_train_confirmed,poly_reg.predict(poly_X_train_confirmed),'Predicting Confirmed Cases Worldwide (Training Set)')

# Plot test set
linear_plot(X_test_confirmed,y_test_confirmed,poly_reg.predict(poly_X_test_confirmed),'Predicting Confirmed Cases Worldwide (Test Set)')


# The test set curve shows much better prediction.

# In[ ]:


# Plot total cases
linear_plot(dates,total_confirm,poly_reg.predict(poly.fit_transform(dates)),'Predicting Total Confirmed Cases Worldwide')


# ### Death Cases

# In[ ]:


X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths = train_test_split(dates[60:], total_deaths[60:], test_size=0.14, shuffle=False) 


# In[ ]:


# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths = poly.fit_transform(X_train_deaths)
poly_X_test_deaths = poly.fit_transform(X_test_deaths)

# polynomial regression
poly_reg = LinearRegression(normalize=True, fit_intercept=False)
poly_reg.fit(poly_X_train_deaths, y_train_deaths)


# In[ ]:


# Plot training set
linear_plot(X_train_deaths,y_train_deaths,poly_reg.predict(poly_X_train_deaths),'Predicting Deaths Cases Worldwide (Training Set)')

# Plot test set
linear_plot(X_test_deaths,y_test_deaths,poly_reg.predict(poly_X_test_deaths),'Predicting Deaths Cases Worldwide (Test Set)')

# Plot total cases
linear_plot(dates,total_deaths,poly_reg.predict(poly.fit_transform(dates)),'Predicting Total Death Cases Worldwide')

