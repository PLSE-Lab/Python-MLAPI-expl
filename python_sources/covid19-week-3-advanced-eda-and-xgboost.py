#!/usr/bin/env python
# coding: utf-8

# ## <a>Loading libraries and data</a>
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
submission = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')

print(train.shape, test.shape, submission.shape)


# ## <a>Exploratory Data Analysis</a>

# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


train.info()


# We have nan values in Province_State column, rest all columns have no missing values. Let's see how many unique countries we have in train.

# In[ ]:


train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])


# In[ ]:


print(train.Country_Region.nunique())


# In[ ]:


countries = train.Country_Region.unique()
countries


# Now the countries for which we have Province_State data.

# In[ ]:


countries_with_provinces = train[~train['Province_State'].isna()].Country_Region.unique()
countries_with_provinces


# In[ ]:


countries_no_province = [i for i in countries if i not in countries_with_provinces]
len(countries_no_province)


# Let's plot the cumulative Fatalities and ConfirmedCases vs Date for the countries with no Province_State data.

# In[ ]:


_, ax = plt.subplots(10, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries_no_province[:20])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')    
#     ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(10, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries_no_province[20:40])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')    
    ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(10, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries[40:60])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')
    ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(10, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries_no_province[60:80])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')    
    ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(10, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries_no_province[80:100])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')  
    ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(10, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries_no_province[100:120])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(10, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries_no_province[120:140])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')    
    ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(10, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries_no_province[140:160])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')    
    ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(6, 2, figsize=(20, 80))
ax=ax.flatten()
for k,i in tqdm(enumerate(countries_no_province[160:172])):
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=train[train['Country_Region'] == i].Date, y=train[train['Country_Region'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# Here,
# 1. Unfortunately, in all plots cumulative ConfirmedCases and Fatalities are increasing with time.
# 2. For most countries, the rate of increase in ConfirmedCases(slope of line) also increased with time.
# 3. For S.Korea and Diamond Princess, rate of increase of ConfirmedCases decreased with time.
# 
# Now, Let's plot for countries with provinces.

# In[ ]:


train['Province_State'] = train['Province_State'].fillna('unknown')
test['Province_State'] = test['Province_State'].fillna('unknown')


# In[ ]:


train[train['Country_Region'].isin(countries_with_provinces)].groupby(['Country_Region']).agg({'Province_State':'nunique'})


# Let's start with Australia.

# In[ ]:


_, ax = plt.subplots(4,2, figsize=(20, 32))
ax = ax.flatten()
temp = train[train['Country_Region'] == 'Australia']
for k,i in tqdm(enumerate(temp.Province_State.unique())):
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# Canada

# In[ ]:


_, ax = plt.subplots(6,2, figsize=(20, 48))
ax = ax.flatten()
temp = train[train['Country_Region'] == 'Canada']
for k,i in tqdm(enumerate(temp.Province_State.unique())):
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# China

# In[ ]:


_, ax = plt.subplots(17,2, figsize=(20, 136))
ax = ax.flatten()
temp = train[train['Country_Region'] == 'China']
for k,i in tqdm(enumerate(temp.Province_State.unique())):
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# Surprisingly, in most of Chinese Provinces, cumulative ConfirmedCases have stopped increasing. Found a cure?

# Denmark

# In[ ]:


_, ax = plt.subplots(2,2, figsize=(20, 16))
ax = ax.flatten()
temp = train[train['Country_Region'] == 'Denmark']
for k,i in tqdm(enumerate(temp.Province_State.unique())):
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# France

# In[ ]:


_, ax = plt.subplots(5,2, figsize=(20, 40))
ax = ax.flatten()
temp = train[train['Country_Region'] == 'France']
for k,i in tqdm(enumerate(temp.Province_State.unique())):
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# In[ ]:


_, ax = plt.subplots(2,2, figsize=(20, 16))
ax = ax.flatten()
temp = train[train['Country_Region'] == 'Netherlands']
for k,i in tqdm(enumerate(temp.Province_State.unique())):
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# United Kingdom

# In[ ]:


_, ax = plt.subplots(5,2, figsize=(20, 40))
ax = ax.flatten()
temp = train[train['Country_Region'] == 'United Kingdom']
for k,i in tqdm(enumerate(temp.Province_State.unique())):
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# US

# In[ ]:


_, ax = plt.subplots(27, 2, figsize=(20, 216))
ax = ax.flatten()
temp = train[train['Country_Region'] == 'US']
for k,i in tqdm(enumerate(temp.Province_State.unique())):
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].Fatalities, label='Fatalities', lw=2, ax=ax[k])
    sns.lineplot(x=temp[temp['Province_State'] == i].Date, y=temp[temp['Province_State'] == i].ConfirmedCases, label='ConfirmedCases', lw=2, ax=ax[k])
    ax[k].set_title(f'Cumulative Fatalies and Confirmed Cases vs Date ({i})')   
    ax[k].set(xlabel='Month')


# Our observations hold for countries and provinces except China. Let's see Top 20 countries with most fatalities till now.

# In[ ]:


most_fatalities = train[train['Country_Region'].isin(countries_no_province)].groupby(['Country_Region']).Fatalities.max().sort_values(ascending=False)
plt.figure(figsize=(20,6))
sns.barplot(most_fatalities[:20].index, most_fatalities[:20].values)


# Italy and Spain have the most no. of fatalities till now. 

# In[ ]:


most_confirmedCases = train[train['Country_Region'].isin(countries_no_province)].groupby(['Country_Region']).ConfirmedCases.max().sort_values(ascending=False)
plt.figure(figsize=(20,6))
sns.barplot(most_confirmedCases[:20].index, most_confirmedCases[:20].values)


# Now, let's see the no of new Confirmed Cases and Fatalities per day for few countries. 

# In[ ]:


train['ConfirmedCases_diff'] = train.groupby(['Country_Region', 'Province_State'])['ConfirmedCases'].diff()
train['Fatalities_diff'] = train.groupby(['Country_Region', 'Province_State'])['Fatalities'].diff()
train = train.fillna(0)


# In[ ]:


plt.figure(figsize=(20,6))
plt.title('No of New Confirmed Cases per day')
sns.lineplot(x=train[train['Country_Region'] == 'India'].Date, y=train[train['Country_Region'] == 'India'].ConfirmedCases_diff, label='India')
sns.lineplot(x=train[train['Country_Region'] == 'Spain'].Date, y=train[train['Country_Region'] == 'Spain'].ConfirmedCases_diff, label='Spain')
sns.lineplot(x=train[train['Country_Region'] == 'Germany'].Date, y=train[train['Country_Region'] == 'Germany'].ConfirmedCases_diff, label='Germany')
sns.lineplot(x=train[train['Country_Region'] == 'Italy'].Date, y=train[train['Country_Region'] == 'Italy'].ConfirmedCases_diff, label='Italy')
sns.lineplot(x=train[train['Country_Region'] == 'Iran'].Date, y=train[train['Country_Region'] == 'Iran'].ConfirmedCases_diff, label='Iran')
sns.lineplot(x=train[train['Country_Region'] == 'Russia'].Date, y=train[train['Country_Region'] == 'Russia'].ConfirmedCases_diff, label='Russia')


# In[ ]:


plt.figure(figsize=(20,6))
plt.title('No of New Fatalities per day')
sns.lineplot(x=train[train['Country_Region'] == 'India'].Date, y=train[train['Country_Region'] == 'India'].Fatalities_diff, label='India')
sns.lineplot(x=train[train['Country_Region'] == 'Spain'].Date, y=train[train['Country_Region'] == 'Spain'].Fatalities_diff, label='Spain')
sns.lineplot(x=train[train['Country_Region'] == 'Germany'].Date, y=train[train['Country_Region'] == 'Germany'].Fatalities_diff, label='Germany')
sns.lineplot(x=train[train['Country_Region'] == 'Italy'].Date, y=train[train['Country_Region'] == 'Italy'].Fatalities_diff, label='Italy')
sns.lineplot(x=train[train['Country_Region'] == 'Iran'].Date, y=train[train['Country_Region'] == 'Iran'].Fatalities_diff, label='Iran')
sns.lineplot(x=train[train['Country_Region'] == 'Russia'].Date, y=train[train['Country_Region'] == 'Russia'].Fatalities_diff, label='Russia')


# ## <a> Model </a>
# 

# In[ ]:


train['Date'] = train['Date'].dt.strftime("%m%d")
train['Date'] = train['Date'].astype(int) 

test['Date'] = test['Date'].dt.strftime("%m%d")
test['Date'] = test['Date'].astype(int) 


# In[ ]:


train['Province_State'] = train['Province_State'].fillna('unknown')
test['Province_State'] = test['Province_State'].fillna('unknown')


# In[ ]:


train['Province_State'] = train['Province_State'].astype('category')
train['Country_Region'] = train['Country_Region'].astype('category')

test['Province_State'] = test['Province_State'].astype('category')
test['Country_Region'] = test['Country_Region'].astype('category')
train


# In[ ]:


FEATURES = ['Date']
submission = pd.DataFrame(columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])

for i in tqdm(train.Country_Region.unique()):
    z_train = train[train['Country_Region'] == i]
    z_test = test[test['Country_Region'] == i]
    for k in z_train.Province_State.unique():
        p_train = z_train[z_train['Province_State'] == k]
        p_test = z_test[z_test['Province_State'] == k]
        x_train = p_train[FEATURES]
        y1 = p_train['ConfirmedCases']
        y2 = p_train['Fatalities']
        model = xgb.XGBRegressor(n_estimators=1000)
        model.fit(x_train, y1)
        ConfirmedCasesPreds = model.predict(p_test[FEATURES])
        model.fit(x_train, y2)
        FatalitiesPreds = model.predict(p_test[FEATURES])
        
        p_test['ConfirmedCases'] = ConfirmedCasesPreds
        p_test['Fatalities'] = FatalitiesPreds
        submission = pd.concat([submission, p_test[['ForecastId', 'ConfirmedCases', 'Fatalities']]], axis=0)


# In[ ]:


submission.to_csv('submission.csv', index=False)

