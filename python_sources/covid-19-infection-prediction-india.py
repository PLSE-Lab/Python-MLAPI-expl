#!/usr/bin/env python
# coding: utf-8

# # Fitting an Exponential Growth Model for Predicting Infection Spread in India

# ### This note book is just a trial from my end to see if exponetial growth model can be used to predict next few weeks infection growth in India. As the infection spread in India is at initial stage thus it is expected to have a exponential growth rate.

# ### Exponential growth function could be only applicable for intial outbreak as after some time, healed people will not spread the virus anymore and when (almost) everyone is or has been infected, the growth will stop [1]. Thus on such situation logistic growth could be the best representation.

# ### SOURSES: 
# * Individual level data comes from [covid19india](https://www.covid19india.org/)
# * State level data comes from [Ministry of Health & Family Welfare](https://www.mohfw.gov.in/)

# # Content :
# * [Load relavant libraries](#Load-relavant-libraries)
# * [Load dataset](#Load-dataset)
# * [Data Cleaning](#Data-Cleaning)
#  * [Estimate daily count](#Estimate-daily-count)
#  * [Infection Plot](#Infection-Plot)
# * [Model Fitting](#Model-Fitting)
#  * [Model Coefficients](#Model-Coefficients)
# * [Prediction for existing days](#Prediction-for-existing-days)
#  * [Prediction upto present date](#Prediction-upto-present-date)
#  * [Actual vs Predicted Data Frame](#Actual-vs-Predicted-Data-Frame)
#  * [Actual vs Predicted Plot](#Actual-vs-Predicted-Plot)
# * [Predictions for next two week](#Predictions-for-next-two-week)
# * [Prediction Plot](#Prediction-plot)
# 

# # Load relavant libraries

# The first step is to load libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# # Loading COVID-19 dataset

# In[ ]:


India_covid19 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv', parse_dates=['Date'], dayfirst=True)
print(India_covid19.head())
print(India_covid19.tail())


# Checking the column names

# In[ ]:


India_covid19.columns


# # Data Cleaning

# In[ ]:


India_covid19.drop(["Sno"], axis = 1, inplace = True)
India_covid19.rename(columns = {"State/UnionTerritiry": "States"}, inplace=True)
print(India_covid19.head())
print(India_covid19.tail())


# ### Estimate daily count

# In[ ]:


India_per_day = India_covid19.groupby(["Date"])["Confirmed"].sum().reset_index().sort_values("Date", ascending = True)
print(India_per_day.head())
print(India_per_day.tail())
print(India_per_day.shape)


# In[ ]:


India_per_day.shape[0]


# In[ ]:


India_per_day['Date']=pd.to_datetime(India_per_day.Date,dayfirst=True)
India_daily= India_per_day.groupby(['Date'])['Confirmed'].sum().reset_index().sort_values('Date',ascending=True)
India_daily["day_count"] = np.arange(0, India_daily.shape[0])

daily_infection = India_daily.loc[:, ["day_count", "Confirmed"]]
print(daily_infection.head())
print(daily_infection.tail())


# ## Infection Plot

# In[ ]:


plt.scatter(daily_infection["day_count"], daily_infection["Confirmed"], alpha=0.3, c="red")
plt.plot(daily_infection["day_count"], daily_infection["Confirmed"])
plt.title("Daily Infection Plot")
plt.xlabel("Day")
plt.ylabel("Infections")
plt.show()


# > ## Fitting Exponential Growth Model

# For the initial phase of outbrake, epidemiologist usually express the growth as exponetial growth.
# Exponential growth can be expresed by the following formula:
# 
# $x(t) = x_0 + b^t$
# 
# where,
# * x(t) is the number of cases at any given time t
# * x0 is the number of cases at the beginning, also called initial value
# * b is the number of people get infected by each infected person, the growth factor

# To obtain the x0 and b we can fit a linear regression. But due to the exponential nature of the data we need to take logarithm of the confirmed cases and then fitting a linear regression. This kind of transformation is required.

# In[ ]:


# Taking log of dependent variable
daily_infection["logConfirmed"] = np.log(daily_infection.Confirmed)
daily_infection.head(4)


# Transformation makes it linear though error will be present

# In[ ]:


plt.scatter(daily_infection["day_count"], daily_infection["logConfirmed"], alpha=0.3, c="red")
plt.plot(daily_infection["day_count"], daily_infection["logConfirmed"])
plt.title("Daily Infection Plot")
plt.xlabel("Day")
plt.ylabel("log_Infections")
plt.show()


# # Model Fitting

# We can fit linear regression (ordinary least square regression) to estimate the coefficients using the statsmodels package.

# Loading library and adding a constant

# In[ ]:


import statsmodels.api as sm

X = daily_infection.day_count
X = sm.add_constant(X)
y = daily_infection.logConfirmed


# Fitting OLS model the data

# In[ ]:


model = sm.OLS(y, X)
reg = model.fit()
print(reg.summary())


# ## Model Coefficients 

# Estimated coefficients are in log term so we need to take exponent to the actual parameter values

# In[ ]:


x0 = np.exp(reg.params[0])
b = np.exp(reg.params[1])
x0, b


# ## Prediction for existing days

# ### Prediction upto present date

# In[ ]:


t1 = np.arange(India_daily.shape[0])
y = (x0 + b**t1).round()
y


# ### Actual vs Predicted Data Frame

# In[ ]:


upto_now = pd.DataFrame({'day_count': t1, "Actual": daily_infection["Confirmed"], "Predicted": y, })
upto_now


# ### Actual vs Predicted Plot

# In[ ]:


plt.plot(upto_now.day_count, upto_now.Actual, alpha=0.4, c="green")
plt.plot(upto_now.day_count, upto_now.Predicted, alpha=0.4, c="red")

plt.title("Actual vs Predicted Plot")
plt.legend(["Actual Count", "Predicted Count"])
plt.xlabel("Day")
plt.ylabel("Infections")
plt.show()


# ### Prediction rmse

# In[ ]:


from sklearn.metrics import mean_squared_error

mean_squared_error(upto_now.Actual, upto_now.Predicted, squared=False)


# ## Predictions for next two week

# In[ ]:


India_daily.shape[0] + 14


# Setting next two weeks data sequence in "t" varible

# In[ ]:


t = np.arange(India_daily.shape[0], India_daily.shape[0] + 14)
t


# Calculating the next two weeks possible infection count

# In[ ]:


xt = (x0 + b**t).round()
xt


# In[ ]:


next2weeks = pd.DataFrame({'day_count': t, "Confirmed": xt})
next2weeks


# Plotting the projection of next two weeks with exiting data

# In[ ]:


X = daily_infection.day_count
y = daily_infection.Confirmed

X1 = next2weeks.day_count
y1 = next2weeks.Confirmed


# ## Prediction plot

# In[ ]:


plt.scatter(X, y, alpha=0.2, c="blue")
plt.scatter(X1, y1, alpha=0.1, c="blue")
plt.plot(X, y)
plt.plot(X1, y1)
plt.title("Future Infection Prediction")
plt.legend(["Up to the Present Day", "Next 14 Days Prediction"])
plt.xlabel("Day")
plt.ylabel("Infections")
plt.show()


# References
#     
#     [1] Joos Korstanje (2020). "Modeling Exponential Growth", medium.com (TDataScience), https://towardsdatascience.com/modeling-exponential-growth-49a2b6f22e1f.
