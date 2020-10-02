#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 0. Load Data

# In[ ]:


submission = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv", parse_dates=['Date'])
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv", parse_dates=['Date'])


# # 1. Exploratory Data Analysis

# ## 1.1 Input

# ### 1.1.1 train

# In[ ]:


display(train.head(5))
display(train.describe())
print(train.dtypes)
print("\n")
print("Number of Country_Region: ", train['Country_Region'].nunique())
print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")
print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())


# ### 1.1.2 test

# In[ ]:


display(test.head(5))
display(test.describe())
print("Number of Country_Region: ", test['Country_Region'].nunique())
print("Dates go from day", max(test['Date']), "to day", min(test['Date']), ", a total of", test['Date'].nunique(), "days")
print("Countries with Province/State informed: ", test[test['Province_State'].isna()==False]['Country_Region'].unique())


# ### 1.1.3 Submission

# In[ ]:


display(submission.head(5))
display(submission.describe())


# ## 1.2 Cases by Region over Time

# ### 1.2.1 geo_id = Country_Region + Province_State
#     

# In[ ]:


train['geo_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)
train.head()


# In[ ]:


test['geo_id'] = test['Country_Region'].astype(str) + '_' + test['Province_State'].astype(str)
test.head()


# ### 1.2.2 combined plot

# In[ ]:


fig, ax = plt.subplots()
train.sort_values(by="Date").groupby('geo_id').plot.line(x='Date', y='ConfirmedCases', ax=ax, legend=False)


# ### 1.2.3 single plots

# In[ ]:


#train.groupby('geo_id').plot.line(x='Date', y='ConfirmedCases')


# ## 1.3 China

# In[ ]:


train[train.Country_Region == "China"].Province_State.unique()


# In[ ]:


train[train.Country_Region == "China"].groupby("Province_State").sum()


# ### 1.3.1 Hubei, China

# In[ ]:


train_hubei = train[train['geo_id'] == "China_Hubei"]
train_hubei


# In[ ]:


fig, ax = plt.subplots()
train_hubei.plot.line(x='Date', y='ConfirmedCases', ax=ax, legend=False)


# # 2 Copy Hubei, China
# Idea: Scale number from Hubei, China by population and translate by date of first infection.

# ## 2.1 Feature Preparation

# ### 2.1.1 Population Data for Country_Regions
# > https://www.kaggle.com/koryto/countryinfo

# In[ ]:


area_features = train[['geo_id', 'Country_Region', 'Province_State']].drop_duplicates().set_index('geo_id')
area_features


# In[ ]:


countryinfo = pd.read_csv("../input/countryinfo/covid19countryinfo.csv", thousands=',')


# In[ ]:


print(countryinfo.dtypes)
display(countryinfo.head(5))
display(countryinfo.describe())


# In[ ]:


# extract population country data
pop_data = countryinfo[['country', "pop"]].rename(columns={'pop': 'pop_country'}).groupby("country").max()
pop_data


# In[ ]:


# left join pop_country to area_features
area_features = area_features.join(pop_data, how='left', on="Country_Region")
area_features


# In[ ]:


print("Number of countries with population: ", area_features[area_features['pop_country'].isna()==False].Country_Region.nunique())
print("Number of countries without population: ", area_features[area_features['pop_country'].isna()==True].Country_Region.nunique())
print("Countries without population: ", area_features[area_features['pop_country'].isna()==True].Country_Region.unique())


# In[ ]:


# fill country population (pop_country) NA with 100 000
area_features['pop_country'] = area_features['pop_country'].fillna(100000)


# ### 2.1.2 Population Data for Province_States

# In[ ]:


print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())


# In[ ]:


# add column num_states per country
num_states = area_features[['Country_Region', 'Province_State']].fillna("").groupby('Country_Region').count().rename(columns={'Province_State': "num_states"})
area_features = area_features.join(num_states, on="Country_Region")
area_features


# In[ ]:


# fill province_state population (pop) with pop_country / num_states
area_features['pop'] = area_features['pop_country'] / area_features['num_states']
area_features


# ### 2.1.3 Date of First Infection per Province_State

# In[ ]:


date_of_first_infection = train[train['ConfirmedCases'] > 0].groupby(['geo_id']).agg({'Date': 'min'}).rename(columns={'Date': 'date_of_first_infection'})
date_of_first_infection


# In[ ]:


area_features = area_features.join(date_of_first_infection, on="geo_id")
area_features


# ### 2.1.4 Date Delta to Hubei

# In[ ]:


hubei_curve = train[(train['Country_Region'] == 'China') & (train['Province_State'] == 'Hubei')]
hubei_curve = hubei_curve[['Date', 'ConfirmedCases', 'Fatalities']].set_index('Date')
hubei_curve


# In[ ]:


hubei_curve.plot.line()


# In[ ]:


import datetime
date_start_hubei = datetime.datetime(2019, 12, 15) #hubei_curve.index.min()
date_start_hubei


# In[ ]:


area_features['date_delta_hubei'] = area_features['date_of_first_infection'] - date_start_hubei
area_features


# ### 2.1.5 Population Scale to Hubei

# In[ ]:


population_hubei = 58.5 * 10**6
population_hubei


# In[ ]:


area_features['pop_scale_hubei'] = area_features['pop'] / population_hubei
area_features


# ## 2.2 Set Hubei S-Curve for All Areas

# In[ ]:


area_features


# In[ ]:


data = train[['geo_id', 'Date']]
data


# In[ ]:


# add Hubei Curve to all Areas
data = data.join(hubei_curve, on='Date')
data


# ## 2.3 Translate Hubei S-[](http://)Curve
# Translating Hubei curve to date of first infection of each Province_State

# In[ ]:


data = data.join(area_features[['date_delta_hubei']], on="geo_id")
data


# In[ ]:


# translate by date_delta_hubei
data['Date'] = data['Date'] + data['date_delta_hubei']
data


# ## 2.4 Scale Hubei S-Curve
# Scale Hubei curve to population by each Province_State****

# In[ ]:


data = data.join(area_features[['pop_scale_hubei']], on="geo_id")
data


# In[ ]:


# scale by pop_scale_hubei
data['ConfirmedCases'] = data['ConfirmedCases'] * data['pop_scale_hubei']
data['Fatalities'] = data['Fatalities'] * data['pop_scale_hubei']
data


# ## 2.5 Generate Submission

# In[ ]:


# drop unneeded columns
data = data[['geo_id', 'Date', 'ConfirmedCases', 'Fatalities']]
data


# In[ ]:


fig, ax = plt.subplots()
data.sort_values(by="Date").groupby('geo_id').plot.line(x='Date', y='ConfirmedCases', ax=ax, legend=False)


# In[ ]:


test


# In[ ]:


# join 
submission = pd.merge(test, data, how="left", on=["geo_id", 'Date'])
submission


# In[ ]:


submission = submission[['ForecastId', 'ConfirmedCases', 'Fatalities']]
submission


# In[ ]:


submission.fillna(0, inplace=True)
submission


# In[ ]:


submission.describe()


# In[ ]:


submission.to_csv('submission.csv', index=False)

