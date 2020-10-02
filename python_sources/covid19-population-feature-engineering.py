#!/usr/bin/env python
# coding: utf-8

# ## This notebook scrapes population data for the countries in the covid-19 dataset and combines datasets to make one set containing covid-19 data as well as population data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Import Data

# Create dataframes and country/province lists.

# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

submission


# In[ ]:


# Format date
train["Date2"] = train["Date"].apply(lambda x: x.replace("-",""))
train["Date2"]  = train["Date2"].astype(int)

# drop nan's

#train = train.dropna()
#train.isnull().sum()

# Do same to Test data
test["Date2"] = test["Date"].apply(lambda x: x.replace("-",""))
test["Date2"]  = test["Date2"].astype(int)


# In[ ]:





# In[ ]:


test.sample(10)


# ## Training Set Pre-Processing and Feature Engineering

# In[ ]:


# some pre-processing, training set
train['Date'] = (pd.to_datetime(train['Date']))
test['Date'] = (pd.to_datetime(test['Date']))
sLength = len(train['Id'])
tLength = len(test['ForecastId'])
train['daysFirstCase'] = pd.Series(np.zeros(sLength,dtype=int), index=train.index)
train['daysFirstDeath'] = pd.Series(np.zeros(sLength,dtype=int), index=train.index)
train['CountryID'] = pd.Series(np.zeros(sLength,dtype=int), index=train.index)
test['daysFirstCase'] = pd.Series(np.zeros(tLength,dtype=int), index=test.index)
test['daysFirstDeath'] = pd.Series(np.zeros(tLength,dtype=int), index=test.index)
test['CountryID'] = pd.Series(np.zeros(tLength,dtype=int), index=test.index)
province_list = train['Province_State'].unique()[1:] # first entry is nan so skip it
country_list = train['Country_Region'].unique()
if len(test['Country_Region'].unique()) != len(country_list):
    raise Exception('ERROR: Different number of unique countries in train/test sets!')

# Create days since first case column.****************************************************

provinceFirstCaseList = []
for province in province_list:
    loop_date = (train[train['Province_State']==province].tail(1)['Date'].values)[0]
    for index, row in train[train['Province_State']==province].iterrows():
        if row['ConfirmedCases'] > 0:
            loop_date = row['Date']
            break
    provinceFirstCaseList.append(loop_date)
            
            
for index, row in train.iterrows():
    loop_province = row['Province_State']
    if type(loop_province) == float:
        continue
    loop_date = provinceFirstCaseList[np.where(province_list == loop_province)[0][0]]
    train.loc[index,'daysFirstCase'] = (row['Date']-loop_date).days

countryFirstCaseList = []
for country in country_list:
    loop_date = (train[train['Country_Region']==country].tail(1)['Date'].values)[0]
    for index, row in train[train['Country_Region']==country].iterrows():
        if row['ConfirmedCases'] > 0:
            loop_date = row['Date']            
            break
    countryFirstCaseList.append(loop_date)

for index, row in train.iterrows():
    loop_province = row['Province_State']
    if type(loop_province) != float:
        continue
    loop_country = row['Country_Region']
    loop_date = countryFirstCaseList[np.where(country_list == loop_country)[0][0]]
    train.loc[index,'daysFirstCase'] = (row['Date']-loop_date).days

# Create days since first death column. **************************************************

provinceFirstDeathList = []
for province in province_list:
    loop_date = (train[train['Province_State']==province].tail(1)['Date'].values)[0]
    for index, row in train[train['Province_State']==province].iterrows():
        if row['Fatalities'] > 0:
            loop_date = row['Date']
            break    
    provinceFirstDeathList.append(loop_date)
       
for index, row in train.iterrows():
    loop_province = row['Province_State']
    if type(loop_province) == float:
        continue
    loop_date = provinceFirstDeathList[np.where(province_list == loop_province)[0][0]]
    train.loc[index,'daysFirstDeath'] = (row['Date']-loop_date).days

countryFirstDeathList = []
for country in country_list:
    loop_date = (train[train['Country_Region']==country].tail(1)['Date'].values)[0]
    for index, row in train[train['Country_Region']==country].iterrows():
        if row['Fatalities'] > 0:
            loop_date = row['Date']
            break
    countryFirstDeathList.append(loop_date)

for index, row in train.iterrows():
    loop_province = row['Province_State']
    if type(loop_province) != float:
        continue
    loop_country = row['Country_Region']
    loop_date = countryFirstDeathList[np.where(country_list == loop_country)[0][0]]
    train.loc[index,'daysFirstDeath'] = (row['Date']-loop_date).days

# Import world population data for migration metric
worldpop = pd.read_csv("/kaggle/input/worldpop-utf8/world_population_countryID0.csv")#, encoding = "ISO-8859-1", engine='python')
print(len(worldpop['CountryPop']))
print(len(country_list))
worldpop.sample(10)

for index, row in train.iterrows():
    loop_country = row['Country_Region']
    loop_countryID = np.where(country_list == loop_country)[0][0]
    train.loc[index,'CountryID'] = loop_countryID

#Attempt some merging of world pop
merged = train.join(worldpop.set_index('CountryID'), on='CountryID', how='left')
merged['MigPerc'] = 100*merged['CountryMigration']/merged['CountryPop']

# Normalize CountryPop and CountryMigration
merged['CountryPop'] = merged['CountryPop']/merged['CountryPop'].max()
merged['CountryMigration'] = merged['CountryMigration']/merged['CountryMigration'].max()
merged['Lag'] = merged['daysFirstCase']-merged['daysFirstDeath']
merged.sample(5)


# 

# ## Test Set Pre-Processing and Feature Engineering
# 
# 

# Notice, when you look at a sample of the test data below, that in order to test our model, we need to test on the same features we train on. So we need to make the table blow have the same columns as the table above excluding the ConfirmedCases and Fatalities as these are the test targets.

# In[ ]:


for index, row in test.iterrows():
    loop_province = row['Province_State']
    if type(loop_province) != float:
        continue
    loop_country = row['Country_Region']
    loop_date = countryFirstCaseList[np.where(country_list == loop_country)[0][0]]
    test.loc[index,'daysFirstCase'] = (row['Date']-loop_date).days

for index, row in test.iterrows():
    loop_province = row['Province_State']
    if type(loop_province) != float:
        continue
    loop_country = row['Country_Region']
    loop_date = countryFirstDeathList[np.where(country_list == loop_country)[0][0]]
    test.loc[index,'daysFirstDeath'] = (row['Date']-loop_date).days
    
for index, row in test.iterrows():
    loop_country = row['Country_Region']
    loop_countryID = np.where(country_list == loop_country)[0][0]
    test.loc[index,'CountryID'] = loop_countryID

    
    
test.sample(5)


# In[ ]:


# Make merged_test
#Attempt some merging of world pop
merged_test = test.join(worldpop.set_index('CountryID'), on='CountryID', how='left')
merged_test['MigPerc'] = 100*merged_test['CountryMigration']/merged_test['CountryPop']

# Normalize CountryPop and CountryMigration
merged_test['CountryPop'] = merged_test['CountryPop']/merged_test['CountryPop'].max()
merged_test['CountryMigration'] = merged_test['CountryMigration']/merged_test['CountryMigration'].max()
merged_test['Lag'] = merged_test['daysFirstCase'] - merged_test['daysFirstDeath']
merged_test.sample(7)

