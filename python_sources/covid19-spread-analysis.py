#!/usr/bin/env python
# coding: utf-8

# # Covid19 - Spread analysis 
# **#Coronavirus**  
# 
# **Author:** Manish Kumar Mishra
# 
# Explore the trends of corona virus in various countries and see if we can find a pattern in how it spreads over time.  
# The important thing is to understand is the long incubation period of this virus. A person detected today, was actually infected 12-14 days ago and he was acting as a virus carrier since then, causing various other people to get infected who all will be diagnosed after some day once the incubation period is complete.  
# 
# 
# **Objective:** The only objective of this analysis is to make sure that you understand the severity of **#coronavirus** spread trend and how fast it can reach us. We should all **#takecare** of ourselves & our families, and stop the virus to spread any further.  
#   
#  
# ---
# *References:*  
# *1. https://www.who.int/*  
# *2. https://www.kaggle.com*  
# 
# *Few data points may not be accurate as the numbers are changing quickly, but they would be representative. Some data points I have manually updated from different sources.*
# 
# **[Updated with confirmed cases prediction for India after 7 days]**

# ---  
# ### Load modules

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go 
from plotly.offline import init_notebook_mode
import datetime

init_notebook_mode(connected=True)


# In[ ]:


pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 100)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ### Load Data

# In[ ]:


print(os.listdir('../input/covid19-data-16mar20/'))
os.getcwd()

filename = '../input/covid19-data-16mar20/covid19_data_16mar20.csv'


# In[ ]:


# os.listdir()
# filename = "covid19_data_16mar20.csv"
covid19 = pd.read_csv(filename, parse_dates=['observation_date'])
print(covid19.shape)


# In[ ]:


covid19.head(3)


# ### Data duration 

# In[ ]:


min_date = covid19.observation_date.min(); max_date = covid19.observation_date.max();

print("Data start date: ", min_date)
print("Data end date:   ", max_date)
print("Period of data:  ", max_date - min_date)


# > As per WHO it all started from 22 Jan'20. It's been more than 54 days today.

# ---  
# # Country wise spread trend for "covid19"

# In[ ]:


fig = px.line(covid19, x="observation_date", y="confirmed", color="country", #line_group="country", 
              hover_name="country",
              line_shape="spline", render_mode="svg",
              title = 'Country wise confirmed cases - "covid19"')
fig.show()


# > It is clearly visible that China had exponential increase in covid19 cases till 16th Feb'20. Since then decrease in the increment rate can be seen, probably after locking down some cities completely. Now the similar trends can be seen in other countries if we explore them separately.  

# In[ ]:


fig = px.line(covid19, x="observation_date", y="deaths", color="country", #line_group="country", 
              hover_name="country",
              line_shape="spline", render_mode="svg",
              title = 'Country wise death cases - "covid19"')
fig.show()


# > Similar trends can be seen in death cases across countries.  

# # How many countries infected yet?

# In[ ]:


covid19_latest = covid19[covid19.observation_date==max_date]
covid19_latest.shape


# > **Currently more than 148 countries are infected**  
# 
# *I missed to capture few countries, so it's actually more.*

# # Countries with most no of "covid19" confirmed cases?

# In[ ]:


covid19_latest.sort_values(['confirmed'], ascending=False, inplace=True)
covid19_latest.reset_index(drop=True, inplace=True)
covid19_latest.head(10)


# ---  
# 
# # Spread pattern for few most affected countries (top 10) 
# 
# We saw the pattern for china. Let's see for some other most affected countries.  
# I am sure we would see similar exponentially increasing trends in all of them. Let's check out.  

# In[ ]:


# Exclude the top infected country - China 
for country in covid19_latest.country[0:10]:
#     print(country)
    # Visualize trend for selected country
    df_plot = covid19[covid19.country==country]
    # print(df_plot)
    fig = go.Figure()

    fig.add_trace(go.Line(x=df_plot["observation_date"], y=df_plot["confirmed"], name="confirmed"))
    fig.add_trace(go.Line(x=df_plot["observation_date"], y=df_plot["deaths"], name="deaths"))

    fig.update_layout(template='none', title={ 'text': 'Confirmed "covid19" cases - [' + country + ']'}
                      , xaxis_title = 'Date', yaxis_title='Counts')

    fig.show()


# # Where is India today?

# In[ ]:


covid19_latest[covid19_latest.country=='India']


# > India is at 40th position with 114 "covid19" confirmed cases & 2 deaths.

# In[ ]:


df_plot = covid19[covid19.country=='India']
# print(df_plot)
fig = go.Figure()

fig.add_trace(go.Line(x=df_plot["observation_date"], y=df_plot["confirmed"], name="confirmed"))
fig.add_trace(go.Line(x=df_plot["observation_date"], y=df_plot["deaths"], name="deaths"))

fig.update_layout(template='none', title={ 'text': 'Confirmed "covid19" cases - [India]'}
                  , xaxis_title = 'Date', yaxis_title='Counts')

fig.show()


# > Clearly the exponential trends for these countries tells the story and those who are not affected yet, must act now to stay that way.

# # Dataset as of 16th Mar'20

# In[ ]:


covid19_latest.head(10)


# ---
# ---
# 
# # ML Data processing

# In[ ]:


# process ml ready dataset to predict for n days
# WAP for below tasks & process each country one by one.
#1. select a country
#2. get min_date & max_date for this country
#3. get train dates for min_date to (max_date - n days)
#4. get train labels for (min_date + n days) to max_date
#5. convert train dates to numeric days (number), considering each country will have a different start day (day 1).

#6. Include countries which are infected from more than 30 days.


# ## How many countries are infected from more than 30 days?

# In[ ]:


# How many countries are infected from more than 30 days
country_wise_count = pd.DataFrame(covid19.groupby(['country']).nunique()['observation_date'].reset_index())
country_30_days = country_wise_count[country_wise_count.observation_date >= 30]
country_30_days= country_30_days.sort_values('observation_date', ascending = False).reset_index(drop=True)
print(country_30_days.shape)


# > There are 30 such countries affected for more than 30 days.

# ## Function to create training dataset

# In[ ]:


# Function to create the training dataset with data from countries affected for more than 30 days.
def fn_create_train_dataset(covid_df, country_name = 'China', prediction_period = 7):
    # Filter for given country
    covid_df = covid_df[covid_df.country== country_name]

    # Get min and max dates
    min_date = covid_df.observation_date.min()
    max_date = covid_df.observation_date.max()
    
    # Add response date to dataset
    covid_df['response_date'] = covid_df.observation_date + datetime.timedelta(days=prediction_period)
    
    # Add numeric date index
    # Since we only have one record for each day, adding an index would do.
    covid_df = covid_df.reset_index(drop=True).reset_index()
    covid_df =covid_df.rename(columns={'index':'date_index'})
    
    # Create response dataset
    covid_df_response = covid_df[['observation_date',
                                  'confirmed']].rename(columns={'observation_date':'response_date',
                                                                'confirmed':'future_cases'})
    
    # Create training dataset
    df_train = pd.merge(covid_df, covid_df_response, on ='response_date')
    
    # drop additional features
    del [df_train['observation_date'], df_train['response_date'], df_train['country']]
    
    return(df_train)


# In[ ]:


# Test above function
# fn_creat_train_dataset(covid_df = covid19, country_name='China').head(2)


# # Training dataset
# 
# **Prediction Period:** 7 Days

# Creating 3 different training dataset to experiment with:  
# 1. With countries affected for more than 30 days
# 2. With all countries data
# 3. With a specific country 

# In[ ]:


df_train = pd.DataFrame()

for country in country_30_days.country:
    df_temp = fn_create_train_dataset(covid_df = covid19, 
                                     country_name = country)
    df_train = df_train.append(df_temp)
    
print(df_train.shape)
# df_train.head()


# ### Aggregate training dataset on countries affeceted for >30 days

# In[ ]:


df_train_agg_30days = df_train.groupby('date_index').mean()[['confirmed', 'future_cases']].reset_index()

X_30days = df_train_agg_30days[['confirmed']]
y_30days = df_train_agg_30days['future_cases']


# In[ ]:


# Visualize
df_train_agg_30days.plot(x = 'date_index', y = 'confirmed')
plt.xlabel('Days')
plt.ylabel('Avg Confirmed cases')
plt.title('Aggregated Spread trend - Countries effected for > 30 days')
plt.show()


# ### Aggregate training dataset on all countries

# In[ ]:


df_train1 = pd.DataFrame()
for country in covid19['country'].unique():
    df_temp = fn_create_train_dataset(covid_df = covid19, 
                                     country_name = country)
    df_train1 = df_train1.append(df_temp)

# aggregate on all available countries
df_train_agg_all = df_train1.groupby('date_index').mean()[['confirmed', 'future_cases']].reset_index()
# print(df_train_agg_all.shape)
# print(df_train_agg_all.head(2))

X_all = df_train_agg_all[['confirmed']]
y_all = df_train_agg_all['future_cases']


# How does the aggregated training dataset look?

# In[ ]:


# Visualize
df_train_agg_all.plot(x = 'date_index', y = 'confirmed')
plt.xlabel('Days')
plt.ylabel('Avg Confirmed cases')
plt.title('Aggregated Spread trend - All countries')
plt.show()


# ###  Country specific training data

# In[ ]:


country_name = 'Germany'
# Get data for a specific country
df_train_country= fn_create_train_dataset(covid_df = covid19, 
                                     country_name = country_name)

# Train on specific country
X_country = df_train_country[['confirmed']]
y_country = df_train_country['future_cases']


# In[ ]:


# Visualize
df_train_country.plot(x = 'date_index', y = 'confirmed')
plt.xlabel('Days')
plt.ylabel('Avg Confirmed cases')
plt.title('Country specific spread trend - ' + country_name)
plt.show()


# # Train ML Model

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


########## CHOOSE A DATASET HERE #######################
# Fit polynomial regression to the train_30days dataset
# X = X_30days
# y = y_30days

# Fit polynomial regression to the train_all dataset
# X = X_all
# y = y_all

# Fit polynomial regression to country specific dataset
X = X_country
y = y_country

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)


# Visualize
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Current vs predicted')
plt.xlabel('Current Count')
plt.ylabel('Predicted Count')
plt.show()


# ## Predict confirmed cases in India after 7 days from 16th Mar'20

# In[ ]:


df_india =covid19[covid19.country == 'India']
# X_india = df_india[['confirmed']]


# In[ ]:


X_india = df_india.sort_values(['observation_date'], ascending=False).iloc[0:1, 2:3]
lin_reg.predict(poly_reg.fit_transform(X_india))


# > After running various experiments on few country based and aggregated datasets, it looks like we are likely to see a minimum of **750 to 1000 cases by 23rd Mar'20**.  

# I would like to be wrong in predicting above, and hope numbers stay on lesser side.   
# I also think that these are not big numbers, and an explosion may occur sooner if we don't act now to take care of ourselves and our families.
# 
# These numbers may also have effects of upcoming summer season and measure taken by Indian govt.   
# **Fingers crossed** 
# 
# #TakeCare #Coronavirus #Covid19

# ---
