#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


covid_19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_19.head()


# In[ ]:


# We will first try to prepare the data for exploration and then will look to do some predictions

covid_19 = covid_19.drop(['SNo'],axis=1)

# Defining a function to check the NULL Values

def check_null_values(dataframe_to_check):
    null_value  = dataframe_to_check.isnull().sum()
    null_value_precent = round(100*(dataframe_to_check.isnull().sum()/len(dataframe_to_check.index)),2)
    df_to_return  = pd.DataFrame({'Missing Value Count': null_value,'Missing Value Percentage':null_value_precent})
    print(df_to_return)
    
print(covid_19.shape)

# Checking the Missing Value Distribution
check_null_values(covid_19)


# In[ ]:


# Now we will look at the Distribution of the State and Provinces to find the Most Affected Places. We can similarly use this technique of Mode to Impute the missing value so that we donot loose that 40% of Data.

covid_19['Province/State'].value_counts()


# Till Now we see that the Only column which is having Null Values is the **State** column which means around 40% of Cases have not been disclosed with the Location which sounds fishy, but we know that **Gansu** has the most number of Cases, hence we will impute all such Cases with **Gansu** Location.   
# 
# This is our call of imputation. We could have also imputed with **Location Not Available**, which sounds logically correct as we can keep track of Such Cases. 
# Ok so let's impute them.  

# In[ ]:


covid_19['Province/State'] = covid_19['Province/State'].fillna('Location Not Disclosed',axis=0)

check_null_values(covid_19)

# Now just for some Verification we will check if we have the Same Count of Missing Value or Not

print(covid_19[covid_19['Province/State'] == 'Location Not Disclosed'].shape)

# Hence We have 2462 Intact Places, Let's plot Them Now!


# In[ ]:





# In[ ]:


fig = px.scatter(covid_19, y="Deaths",x = "Recovered", color="Country/Region",
                 size='Confirmed', hover_data=['Province/State','Confirmed','Deaths','Recovered'],log_x=True,log_y=True)
fig.show()


# We can see that **Hubei** is the Location where the maximum number of Cases are getting recorded, but in a Scientific way, same place is getting the number of cases updated, hence we cannot look on it in a Unique Fashion. No problem, what we can do is create a dummy DataFrame separately for **Confirmed,Deaths and Recovered** situations and sum them individually. Let me show you, what i am talking about!

# In[ ]:


confirmed_df = covid_19.groupby(['Country/Region','Province/State']).agg({'Confirmed':sum}).reset_index()

confirmed_df.shape


# In[ ]:


print('Uniqe Countries in our Summed up Data Set:',confirmed_df['Country/Region'].nunique())

print('Unique Countries in our Original Data Set:',covid_19['Country/Region'].nunique())


# In[ ]:


# Similarly

deaths_df = covid_19.groupby(['Country/Region','Province/State']).agg({'Deaths':sum}).reset_index()
recovered_df = covid_19.groupby(['Country/Region','Province/State']).agg({'Recovered':sum}).reset_index()

print('Uniqe Countries in our Summed up Data Set:',deaths_df['Country/Region'].nunique())

print('Unique Countries in our Original Data Set:',covid_19['Country/Region'].nunique())


# In[ ]:


# Next we will merge all these to form the correct dataframe

covid_19_1 = confirmed_df.merge(deaths_df,how='inner',on=['Country/Region','Province/State'])
covid_19_df = covid_19_1.merge(recovered_df,how='inner',on=['Country/Region','Province/State'])

check_null_values(covid_19_df)


# In[ ]:


# Let's have a look at Data

covid_19_df.head(15)


# In[ ]:


# Now let's plot it again
df_df = covid_19_df.groupby(['Country/Region']).agg({'Deaths':sum,'Confirmed':sum,'Recovered':sum}).reset_index()

fig = px.scatter(df_df, y="Deaths",x = "Recovered", color="Country/Region",
                 size='Confirmed', hover_data=['Confirmed','Deaths','Recovered'])
fig.update_yaxes(nticks=20)
fig.update_xaxes(nticks=50)

fig.show()


# We can see the top 5 Countries which have a Gloabal Crisis in terms of **Corona Virus**. These are:
#     1. China
#     2. Italy
#     3. Iran
#     4. South Korea
#     5. Spain
#     
# This Kernel is too late to tell you about these Countries as everyone is already aware of these countries being in crisis. But what is important that how quicky which country got it's count of patients up. 
# 
# That's what we are going to do next. We have already cleaned our DataFrame and for ease of Understanding, we will be only taking these Countries and will do a Analysis, using the **Last Update** column to see week by week or maybe Month by Month analysis for these Countries.
# 
# Let's do this!

# In[ ]:


covid_19_time_analysis = covid_19.loc[covid_19['Country/Region'].isin(['Mainland China','Italy','Iran','South Korea','Spain'])]

covid_19_time_analysis.shape


# In[ ]:


print(covid_19_time_analysis.info())

# We will drop Province as we donot need that column, since we will be having the Country in common for those Provinces

covid_19_time_analysis = covid_19_time_analysis.drop(['Province/State'],axis=1)
covid_19_time_analysis.rename({'Country/Region':'Country'},inplace=True,axis=1)
covid_19_time_analysis.head()


# Now the plan goes like this:
# 
#     1. We will be looking at the Observation Date as that will give us the Month for that Patient.
#     2. Next we will be building a new feature which will be looking at the time duration and based on a 24 hour format, we  can look at different situations.
#     3. One of the sitauation could be the Deaths happening within how much time difference and on which date. Things like this.
#     4. Then we can even have a Ditribution of Countries, Monthly wise.
#     
# Let's do this:

# In[ ]:


covid_19_time_analysis['Month of Observation'] = pd.to_datetime(covid_19_time_analysis['ObservationDate']).dt.strftime('%B')
#covid_19_time_analysis['Month of Observation'] = covid_19_time_analysis['Month of Observation'].dt.strftime('%B')
covid_19_time_analysis['Year of Observation'] = pd.to_datetime(covid_19_time_analysis['ObservationDate']).dt.year


# In[ ]:


print(covid_19_time_analysis.describe())

# We will also look for the Unique values in the created columns

print('Unique Values in Month of Observation',covid_19_time_analysis['Month of Observation'].nunique())
print('Unique Values in Month of Observation',covid_19_time_analysis['Year of Observation'].nunique())

# This means that we have data from January to March and for the Year of 2020 only.


# In[ ]:


covid_19_time_analysis['Hour of Observation'] = pd.to_datetime(covid_19_time_analysis['Last Update']).dt.hour


# In[ ]:


covid_19_time_analysis.head(10)


# In[ ]:


fig = px.bar(covid_19_time_analysis.groupby('Country').get_group('Mainland China'), x='Hour of Observation', y='Confirmed', color='Month of Observation')
fig.update_xaxes(nticks=24)
fig.show()


# In[ ]:


fig = px.bar(covid_19_time_analysis, x="Hour of Observation", y="Confirmed", facet_col="Country",color='Month of Observation',log_y=True)
fig.update_xaxes(nticks=6)
#fig.update_yaxes(nticks=20)
fig.show()


# We can see rright from the Chart that for Countries like **China** and **South Korea** got a lot of patients in the Month of **January Only** and it got worsen for the Countries like Iran and Italy in the month of **March**.  
# They had 2 whole months to take action and prevent it, but they failed it seems. We still haven't looked at the Ditribution of Deaths, and next we are going to do the same!

# In[ ]:


fig = px.bar(covid_19_time_analysis, x="Hour of Observation", y="Deaths", facet_col="Country",color='Month of Observation',log_y=True,hover_data=[
    'Country','Confirmed','Hour of Observation','Month of Observation'
])
fig.update_xaxes(nticks=6)
#fig.update_yaxes(nticks=50)
fig.show()


# The Cases kept on Piling up and the Deaths keep happening!

# In[ ]:


fig = px.bar(covid_19_time_analysis, x="Hour of Observation", y="Recovered", facet_col="Country",color='Month of Observation',log_y=True,hover_data=[
    'Country','Confirmed','Hour of Observation','Month of Observation','Deaths','Recovered'],title='Looking at the Countries based on Recovered Rate and Hourly Cases for different Months!'
)
fig.update_xaxes(nticks=6)
#fig.update_yaxes(nticks=50)
fig.show()


# In[ ]:


fig = px.scatter(covid_19_time_analysis, x='ObservationDate', y='Deaths', color='Country',title='Looking at the Countries based on Observation Rate and Death Cases!')
fig.update_yaxes(nticks=20)
fig.show()


# Interpreting this Graph makes me more comfortable as in China in the Hour of **14, 360 people recovered  from 452 cases with only 8 deaths recorded** and that too in the month of March.  
# This inference in it's own stands that It took them 3 months to fight this whole Pandemic and somehow they are coming out of it.
# 
# One scientific look i want to make if the **Percentage of Deaths/Confirmeness of Cases and Cases of Recovering**. 
# 
# Giving you the Glimpse of What we will be doing!!
# Let's look at the percentage distribution.

# In[ ]:


fig = px.scatter(covid_19_time_analysis, x='ObservationDate', y='Recovered', color='Country',title='Looking at the Countries based on Observation Rate and Recovered Cases!')
fig.update_yaxes(nticks=20)
fig.show()


# In[ ]:


fig = px.scatter(covid_19_time_analysis, x='ObservationDate', y='Confirmed', color='Country',title='Looking at the Countries based on Observation Rate and Confirmed Cases!')
fig.update_yaxes(nticks=20)
fig.show()


# These three Scatter Plots do give us the Glipmse of What pattern which we are looking for!!.
# 
# 
# Adding my Partners!! Let's rock and Roll!

# In[ ]:


covid_19_time_analysis.head(10)


# In[ ]:


city_wise = covid_19_time_analysis.groupby('Country').sum()
city_wise['Death Rate'] = city_wise['Deaths'] / city_wise['Confirmed'] * 100
city_wise['Recovery Rate'] = city_wise['Recovered'] / city_wise['Confirmed'] * 100
city_wise['Active'] = city_wise['Confirmed'] - city_wise['Deaths'] - city_wise['Recovered']
city_wise = city_wise.sort_values('Deaths', ascending=False).reset_index()
city_wise


# In[ ]:


px.scatter(city_wise,y = 'Recovery Rate',color='Country',x='Active',size='Confirmed',title='Looking at the Countries based on Recovery Rate and Active Cases!')


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=city_wise['Recovery Rate'], y=city_wise['Country'],
                    mode='lines+markers',
                    name='Recovery rate'))
fig.add_trace(go.Scatter(x=city_wise['Death Rate'], y=city_wise['Country'],
                    mode='lines+markers',
                    name='Death Rate'))
fig.show()


# What do we infer from Here?
# 
# **Italy is the Country which is having a Very High Death Rate and Having Low Recovery rate** which is very concerning for the Country!

# In[ ]:


# Now we will try to Look for the 
from plotly.subplots import make_subplots

confirm_death_recovery_cases = covid_19_time_analysis.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum().reset_index()

plot = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered"))

subPlot1 = go.Scatter(
                x=confirm_death_recovery_cases['ObservationDate'],
                y=confirm_death_recovery_cases['Confirmed'],
                name="Confirmed",
                line=dict(color='royalblue', width=4, dash='dot'),
                opacity=0.8)

subPlot2 = go.Scatter(
                x=confirm_death_recovery_cases['ObservationDate'],
                y=confirm_death_recovery_cases['Deaths'],
                name="Deaths",
                line=dict(color='red', width=4, dash='dot'),
                opacity=0.8)

subPlot3 = go.Scatter(
                x=confirm_death_recovery_cases['ObservationDate'],
                y=confirm_death_recovery_cases['Recovered'],
                name="Recovered",
                line=dict(color='firebrick', width=4, dash='dash'),
                opacity=0.8)

plot.append_trace(subPlot1, 1, 1)
plot.append_trace(subPlot2, 1, 2)
plot.append_trace(subPlot3, 1, 3)
plot.update_layout(template="ggplot2", title_text = '<b><i>Global Spread of the Covid 19 Over Time</i></b>',xaxis_title='Observation Dates',
                   yaxis_title='Count')

plot.show()


# We will just Draw two more **Dynamic Pie Charts** which will be our Final Inference and that same comparison we can use when we will be exploring the India's CoronaVirus Dataset!!

# In[ ]:


fig = px.pie(city_wise, values='Recovery Rate', names='Country')
fig.show()


# In[ ]:


fig = make_subplots(rows=1, cols=2,specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=city_wise['Country'], values=city_wise['Death Rate'], name="Death Rate"),
              1, 1)
fig.add_trace(go.Pie(labels=city_wise['Country'], values=city_wise['Recovery Rate'], name="Recovery Rate"),
              1, 2)
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Death and Recovery rate for Different Countries!",
    annotations=[dict(text='Death Rate', x=0.18, y=0.5, font_size=12, showarrow=False),
                 dict(text='Recovery Rate', x=0.82, y=0.5, font_size=12, showarrow=False)])
fig.show()


# My Main Motive is to Explore more and find the Loopholes which can be actually covered as Corona is on Stage 3 in India and this is where it becomes Clumsy!
# 
# Stay Safe, Stay Indoor, Keep Kaggling and hit a **Upvote** if you like this Kernel. It will be an encouragement for us!!

# In[ ]:


recovered_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
confirmed_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
deaths_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
confirmed_df.head()


# In[ ]:


cols = confirmed_df.keys()
print(cols)
# gettingg all the Dates

confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recovered_df.loc[:, cols[4]:cols[-1]]


# 
# Ok, let me explain to you what I am onto Next.  
# 
# I want to build a predictive Model which will be predicting on a **7 day Gap** of Confirmed Cases for these, irrespective of the Countries, bcoz now we are looking on the Data with a Global Eye.  
# 
# I intend to use a SVM model with all the possible kernels and then get the best Kernel performing and then make the predictions too!. I was initially thinking of making a predictive model, but was in dillema as to start from which one and then going by multiple notebooks, found that if we have a very Sparse Data, it is suitable for SVM to give a try as it will obviously cover your **Linear and Polynomial** model. Have any suggestions, let me know in the comment.
# 
# Till then let's do this!

# ## Pattern of Confirmed Cases!

# In[ ]:


over_time_date = confirmed.keys()

# Number of cases Over time
world_cases = []
# Number of deaths Over time
total_deaths = [] 
# The Rate at which Death id occuring
mortality_rate = []
# Number of people Recovered Over Time
total_recovered = [] 

for i in over_time_date:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)

# Converting them into Array for Modelling

from_starting = np.array([i for i in range(len(over_time_date))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)

moving_day_span = 7
single_week_prediction = np.array([i for i in range(len(over_time_date)+moving_day_span)]).reshape(-1, 1)
modified_date = single_week_prediction[:-7]

start_date_data = '1/22/2020'
start_date = datetime.datetime.strptime(start_date_data, '%m/%d/%Y')
in_future_dates = []
for i in range(len(single_week_prediction)):
    in_future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(from_starting, world_cases, test_size=0.15, shuffle=False)

print(X_train_confirmed.shape)
print(X_test_confirmed.shape)
print(y_train_confirmed.shape)
print(y_test_confirmed.shape)


# In[ ]:


kernel = ['poly', 'sigmoid', 'rbf']
c = [0.01, 0.1, 1,10]
gamma =[0.01, 0.1, 1,10]
epsilon = [0.01, 0.1, 1,10]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}

svm = SVR()
best_model = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=20, verbose=1)
best_model.fit(X_train_confirmed, y_train_confirmed)

# Earlier we were using CV = 5 which was creating 200 folds, but reduced to 3 as the notebook was taking too much time to execute.

print(best_model.best_params_)

# Now comes, what we have been waiting for, make those predictions

best_estimator_from_model = best_model.best_estimator_
make_predictions = best_estimator_from_model.predict(single_week_prediction)


# In[ ]:


plt.figure(figsize=(20, 8))
plt.plot(modified_date, world_cases)
plt.plot(single_week_prediction, make_predictions, linestyle='dashed', color='red')
plt.title('# of Coronavirus Cases Over Time', size=15)
plt.xlabel('Days Since 1/22/2020', size=10)
plt.ylabel('Count of Cases', size=10)
plt.legend(['Confirmed Cases', 'Model Predictions'])
plt.xticks(size = 20)
plt.show()


# ## Pattern of Death Occuring Cases!

# In[ ]:


# Now you people can see the precition till 22nd March 2020. Similarly, we would be doing some prediction for Deaths and recoveries

over_time_deaths = deaths.keys()
from_starting_deaths = np.array([i for i in range(len(over_time_deaths))]).reshape(-1, 1)
death_cases = np.array(total_deaths).reshape(-1, 1)

moving_day_span = 7
single_week_prediction = np.array([i for i in range(len(over_time_deaths)+moving_day_span)]).reshape(-1, 1)
modified_date = single_week_prediction[:-7]

start_date_data = '1/22/2020'
start_date = datetime.datetime.strptime(start_date_data, '%m/%d/%Y')
in_future_dates = []
for i in range(len(single_week_prediction)):
    in_future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

X_train_confirmed_deaths, X_test_confirmed_deaths, y_train_confirmed_deaths, y_test_confirmed_deaths = train_test_split(from_starting_deaths, death_cases, test_size=0.15, shuffle=False)

print(X_train_confirmed_deaths.shape)
print(X_test_confirmed_deaths.shape)
print(y_train_confirmed_deaths.shape)
print(y_test_confirmed_deaths.shape)


# In[ ]:


kernel = ['poly', 'sigmoid', 'rbf']
c = [0.01, 0.1, 1,10]
gamma =[0.01, 0.1, 1,10]
epsilon = [0.01, 0.1, 1,10]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}

svm = SVR()
best_model = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
best_model.fit(X_train_confirmed_deaths, y_train_confirmed_deaths)

print(best_model.best_params_)

# Now comes, what we have been waiting for, make those predictions

best_estimator_from_model = best_model.best_estimator_
make_predictions = best_estimator_from_model.predict(single_week_prediction)


# In[ ]:


plt.figure(figsize=(20, 8))
plt.plot(modified_date, death_cases)
plt.plot(single_week_prediction, make_predictions, linestyle='dashed', color='red')
plt.title('# of Coronavirus Death Cases Over Time', size=15)
plt.xlabel('Days Since 1/22/2020', size=10)
plt.ylabel('Count of Death Cases', size=10)
plt.legend(['Death Cases', 'Model Predictions'])
plt.xticks(rotation = 90)
plt.show()


# We tried a Predictive Modelling of Identifying the Pattern and We have done this from the Dataset of **Confirmed Cases and Death Cases**.

# ## Pattern of Recovered Cases!

# In[ ]:


# Now you people can see the precition till 22nd March 2020. Similarly, we would be doing some prediction for Deaths and recoveries

over_time_recoveries = recoveries.keys()
from_starting_recoveries = np.array([i for i in range(len(over_time_recoveries))]).reshape(-1, 1)
recovered_cases = np.array(total_recovered).reshape(-1, 1)

moving_day_span = 7
single_week_prediction = np.array([i for i in range(len(over_time_recoveries)+moving_day_span)]).reshape(-1, 1)
modified_date = single_week_prediction[:-7]

start_date_data = '1/22/2020'
start_date = datetime.datetime.strptime(start_date_data, '%m/%d/%Y')
in_future_dates = []
for i in range(len(single_week_prediction)):
    in_future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

X_train_confirmed_recoveries, X_test_confirmed_recoveries, y_train_confirmed_recoveries, y_test_confirmed_recoveries = train_test_split(from_starting_recoveries, recovered_cases, test_size=0.15, shuffle=False)

print(X_train_confirmed_recoveries.shape)
print(X_test_confirmed_recoveries.shape)
print(y_train_confirmed_recoveries.shape)
print(y_test_confirmed_recoveries.shape)


# In[ ]:


kernel = ['poly', 'sigmoid', 'rbf']
c = [0.01, 0.1, 1,10]
gamma =[0.01, 0.1, 1,10]
epsilon = [0.01, 0.1, 1,10]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}

svm = SVR()
best_model = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
best_model.fit(X_train_confirmed_recoveries, y_train_confirmed_recoveries)

print(best_model.best_params_)

# Now comes, what we have been waiting for, make those predictions

best_estimator_from_model = best_model.best_estimator_
make_predictions = best_estimator_from_model.predict(single_week_prediction)


# In[ ]:


plt.figure(figsize=(20, 8))
plt.plot(modified_date, recovered_cases)
plt.plot(single_week_prediction, make_predictions, linestyle='dashed', color='red')
plt.title('# of Coronavirus Recovered Cases Over Time', size=15)
plt.xlabel('Days Since 1/22/2020', size=10)
plt.ylabel('Count of Recovery Cases', size=10)
plt.legend(['Recovered Cases', 'Model Predictions'])
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


covid_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
covid_india.head()


# In[ ]:


covid_india.info()

covid_india_new = covid_india.copy()


# In[ ]:


# We can see that there are no null values, since 0 is also treated as a Numnber

# Let's have a,look at the columns

print(covid_india_new.describe())

print(covid_india_new['State/UnionTerritory'].value_counts())


# In[ ]:


covid_india_new['State/UnionTerritory'].replace('Chattisgarh','Chhattisgarh',inplace=True)

covid_india_new.head(10)


# In[ ]:


# So we will plot the Ditribution of States based on the number of Cases Recieved/Occurence,as this will tell us the top states which should be in high alert.

# The thing to note is that we are not looking at the number of Cases, but at the occurence of each state. We will be displaying the number of Cases also in the same.

covid_india_new['Confirmed'] = covid_india_new.ConfirmedIndianNational + covid_india_new.ConfirmedForeignNational
covid_india_new_summed = covid_india_new.groupby('State/UnionTerritory').agg({'Deaths':max,'Cured':max,'Confirmed':max}).reset_index()

fig = px.pie(covid_india_new_summed, values='Confirmed', names='State/UnionTerritory'
             ,color_discrete_sequence=px.colors.sequential.RdBu,title='The Distribution of States with the increase in count of Confirmed Cases.!')
fig.update_traces(textposition='inside', textinfo='value+label')
fig.show()

# So we can see that when we reached 620 cases, the Kerala was the State to have the 620th case indeed!

# We will look at the Distribution of the States also.


# In[ ]:


print('Total Confirmed Cases: ', covid_india_new_summed.Confirmed.sum())

print('Total Deaths occured: ', covid_india_new_summed.Deaths.sum())

print('Total Recovered cases: ', covid_india_new_summed.Cured.sum())


# In[ ]:


# 
covid_india_new_sorted = covid_india_new_summed.sort_values(by='Confirmed')

fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_india_new_sorted['Confirmed'], y=covid_india_new_sorted['State/UnionTerritory'],hoverinfo=['all'],
                         mode='lines+markers',
                    name='The Line of Increasing Cases'))
fig.add_trace(go.Scatter(x=covid_india_new_sorted['Deaths'], y=covid_india_new_sorted['State/UnionTerritory'],hoverinfo=['all'],
                         mode='lines+markers',
                    name='The Line of Deaths faced!'))
fig.add_trace(go.Scatter(x=covid_india_new_sorted['Cured'], y=covid_india_new_sorted['State/UnionTerritory'],hoverinfo=['all'],
                         mode='lines+markers',
                    name='The Line of Recovered Cases'))
fig.update_layout(
    title="Which State gets to see the highest number of Confirmed cases??",
    yaxis_title="States",
    xaxis_title="Count of Cases",
    autosize=True,
    height=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="darkblue"
    )
)
fig.show()


# In[ ]:


# We cannot do any modelling because we donot have a day by day data as far, but we would like to make some analysis based on the Individual Data, that has been provided to us.
# Like, what age of people are getting affected most,is gender impacting the situation somewhere and other questions we would like to answer

individual_details = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

individual_details.head(10)


# In[ ]:


individual_details = individual_details.rename(columns=lambda x: x.strip())

cols_to_drop = ['Unique id','ID','Government id','Detected city pt','Notes','Current location','Current location pt','Created on','Updated on','Contacts']

filter_data = individual_details.drop(cols_to_drop,axis=1)

filter_data.head()


# ### Ok, now we have a Idea. Given certain details like, Age, Diagnosed Date,Detetcted State, we would like to know what could be the **Status of Patient** using Clustering technique.
# 
# ### I am thinking of a either EM Clustering or Birch clustering as i would like to check the probability of a data point falling in a Cluster. This can acutally help in focussing on the patients which can Recover fast, so that the patients which fall in **Hospitalized** or **Deceased** cluster can be given more **focus** and a significant **improvement** can be achieved.
# 
# ## This is just one of the ideas which we would like to explore, I mean, ofcourse i want to make a change by doing some work which actually helps somewhere and i urge people to come up with ideas and we would like to try them out.
# 
# # Stay Safe, keep washing hands and look after each other!
# 
