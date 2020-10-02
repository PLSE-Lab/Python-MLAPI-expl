#!/usr/bin/env python
# coding: utf-8

# ## Data Load

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


route = pd.read_csv("/kaggle/input/coronavirusdataset/route.csv")
trend = pd.read_csv("/kaggle/input/coronavirusdataset/trend.csv")
time = pd.read_csv("/kaggle/input/coronavirusdataset/time.csv")
patient = pd.read_csv("/kaggle/input/coronavirusdataset/patient.csv")


# ## Data Explore

# ## Plot route

# ### Sort the number of people confirmed by date

# Remove year from date.
# Print top route dataframe by date.
# 

# In[ ]:


route['date'] = route['date'].str.split('2020-').str.join('')
route = route.sort_values(by=['date'])
route.head()


# To view the number of confirmed by date, use "groupby" by "date" and aggregation by "count".

# In[ ]:


route_date = route.groupby('date').agg('count')
route_date.head()


# ### Using Seaborn, barplot x=date, y=counted id.

# Then we can see the confirmed people from 01-19 to 02-19

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(route_date.index, route_date['id'])
plt.show()


# ### Sort the number of people confirmed by province

# To view the number of confirmed by province, use "groupby" by "province" and aggregation by "count".

# In[ ]:


route_prov = route.groupby('province').agg('count')
route_prov.head()


# ### Using Seaborn, barplot x=province, y=counted id.

# Then we can see the confirmed people from "Daegu" to "Seoul"

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(route_prov.index, route_prov['id'])
plt.show()


# ### Population density : [thousand / km^2] chart by province (2018)

# Using the population density from Korean Government Site (Following Link), make density dataframe

# http://www.index.go.kr/potal/stts/idxMain/selectPoSttsIdxMainPrint.do?idx_cd=1007&board_cd=INDX_001

# In[ ]:


density = pd.DataFrame()
density['province'] = route_prov.index
density['pop_density'] = [2773, 90, 2980, 1279, 141, 2764, 226, 145, 16034]

density


# ### Using Seaborn, barplot x=province, y=population density

# Then we can see the population density from "Daegu" to "Seoul"

# In[ ]:


plt.figure(figsize=(20, 5))
sns.barplot(route_prov.index, route_prov['id'])
plt.figure(figsize=(20, 5))
sns.barplot(density['province'], density['pop_density'])
plt.show()


# ### Simple Normalizing Function

# In[ ]:


def normalize(v):
    norm = np.sum(v)
    if norm == 0: 
       return v
    return v / norm


# ### Find Correlation between Confirmed People and Population Density by Province

# It might be positive or negative correlation between Confirmed People and Population Density by Province.

# In[ ]:


corr = pd.DataFrame()
corr['infected'] = normalize(np.array(route_prov['id']))
corr['density'] = normalize(np.array(density['pop_density']))

corr.index = route_prov.index

corr_coef = pd.DataFrame(np.corrcoef(np.array(corr[['infected', 'density']])))
corr_coef.columns = corr.index
corr_coef.index = corr.index
corr_coef


# ### Return Correlated Indices Function

# In[ ]:


def return_corr_reason(df, col):
    return df[df[col] == 1.0].index


# ### Using Seaborn.heatmap, Find Regularity among Province

# In[ ]:


sns.heatmap(corr_coef)
plt.show()


# Then we can find lattice regularity, so classify as 2 groups. And group1 and group2 is anti-correlated.

# In[ ]:


print("first correlation group :", return_corr_reason(corr_coef, 'Daegu').tolist())
print("second correlation group :", return_corr_reason(corr_coef, 'Seoul').tolist())


# ### From Correlated Groups, Calculate Impact Factor by "the normalized total number of confirmed people"

# And print province which has the biggest impact factor.

# In[ ]:


col1 = return_corr_reason(corr_coef, 'Daegu').tolist()
col2 = return_corr_reason(corr_coef, 'Seoul').tolist()

print("first correlation group's impact :", sum(list(corr.loc[col1, 'infected'])))
print()
print("maximum of first correlation group's impact")
print(corr[corr['infected'] == max(list(corr.loc[col1, 'infected']))]['infected'])
print()
print()
print("second correlation group's impact :", sum(list(corr.loc[col2, 'infected'])))
print()
print("maximum of second correlation group's impact")
print(corr[corr['infected'] == max(list(corr.loc[col2, 'infected']))]['infected'])


# In[ ]:


print("first observation date :", route.head(1)['date'])
print("final observation date :", route.tail(1)['date'])


# Thus, from 01-19 to 02-19, the impacts of first group which contains "Daegu" and second group which contains "Seoul" are almost same. And the most infected province is "Gyeonggi-do" in first group, "Seoul" in second group.

# ## Plot trend

# ### Print top 5 of trend Dataframe

# In[ ]:


trend.head()


# ### Correlation between [cold, flu, pneumonia] and coronavirus through Seaborn.lmplot

# In[ ]:


sns.lmplot(x='cold', y='coronavirus', data=trend)
sns.lmplot(x='flu', y='coronavirus', data=trend)
sns.lmplot(x='pneumonia', y='coronavirus', data=trend)
plt.show()


# Thus, there is a meaningful positive correlation between cold and coronavirus in time Series.

# ### Total trend Chart

# Plot the total trend chart by matplotlib.pyplot in time series.

# In[ ]:


plt.figure(figsize=(20, 5))
plt.plot(trend['date'], trend.drop('date', axis=1))
plt.xlabel('date')
plt.ylabel('volume')
plt.xticks('')
plt.legend(['cold', 'flu', 'pneumonia', 'coronavirus'])
plt.show()


# ## Plot Time

# ### Print top 5 of time DataFrame

# In[ ]:


time['date'] = time['date'].str.split('2020-').str.join('')
time.head()


# ### Plot new confirmed, new released and new deceased

# In[ ]:



plt.figure(figsize=(20, 5))
plt.title("New Confirmed by Date")
sns.barplot(time['date'], time['new_confirmed'])
plt.xticks([])


plt.figure(figsize=(20, 5))
plt.title("New Released by Date")
sns.barplot(time['date'], time['new_released'])
plt.xticks([])


plt.figure(figsize=(20, 5))
plt.title("New Deceased by Date")
sns.barplot(time['date'], time['new_deceased'])
plt.xticks([])
plt.show()


# ### Plot the Number of Infected People by Date

# the Number of Infected People = the Number of Accumulated Confirmed People - the Number of Accumulated Released People.

# In[ ]:


time['acc_infected'] = np.array(time['acc_confirmed']) - np.array(time['acc_released'])

plt.figure(figsize=(20, 5))
plt.title("Infected People by Date")
sns.barplot(time['date'], time['acc_infected'])
plt.xticks([])
plt.show()


# As time goes, the increase of the number of accumulated infected people is decreased.

# ## Plot Patient

# ### Print top 5 of patient DataFrame

# In[ ]:


patient.head()


# ### Patient group by birth_year(Age)

# groupby "birth year" and aggregate by "count".

# In[ ]:


patient_birth = patient.groupby('birth_year').agg('count')
patient_birth.head()


# ### Plot patient by age

# To calculate age, 2020 - patient's birth year.

# In[ ]:


patient_birth.index = list(map(int, np.array(2020) - np.array(patient_birth.index)))
                           
plt.figure(figsize=(30, 5))
sns.barplot(patient_birth.index, patient_birth['id'])
plt.show()


# ### Patient group by infection_reason

# View Patient's number by infection_reason.

# In[ ]:


patient_reason = patient.groupby('infection_reason').agg('count')
patient_reason.head()


# ### Plot the number of patient by reason

# In[ ]:


plt.figure(figsize=(40, 5))
sns.barplot(patient_reason.index, patient_reason['id'])
plt.show()


# ### Contact_number group by infection_reason

# View Patient's Contact Number by infection_reason.

# In[ ]:


patient_contact = patient.groupby('infection_reason').agg('sum')
patient_contact.head()


# ### Plot the contact number by reason

# In[ ]:


plt.figure(figsize=(40, 5))
sns.barplot(patient_contact.index, patient_contact['contact_number'])
plt.show()


# ### Correlation with patient's number and patient's contact_number by reason

# Calculate the Pearson Correlation between patient's number and patient's contact number.

# In[ ]:


corr = pd.DataFrame()
corr['id'] = normalize(patient_reason['id'])
corr['contact_number'] = normalize(patient_contact['contact_number'])
corr_coef = pd.DataFrame(np.corrcoef(np.array(corr[['id', 'contact_number']])))
corr_coef.columns = corr.index
corr_coef.index = corr.index
corr_coef


# ### Using seaborn.heatmap, Find Regularity among reasons.

# In[ ]:


sns.heatmap(corr_coef)
plt.show()


# ### Classify as 2 groups from the regularity.

# In[ ]:


print("first correlation group :", return_corr_reason(corr_coef, 'visit to Wuhan').tolist())
print("second correlation group :", return_corr_reason(corr_coef, 'contact with patient').tolist())


# group1 and group2 is anti-correlated.

# ### Calculate the impact factor of each groups.

# impact factor is also normalized the number of infected people.

# In[ ]:


col1 = return_corr_reason(corr_coef, 'visit to Wuhan').tolist()
col2 = return_corr_reason(corr_coef, 'contact with patient').tolist()

print("first correlation group's impact :", sum(list(corr.loc[col1, 'id'])))
print()
print("maximum of first correlation group's impact")
print(corr[corr['id'] == max(list(corr.loc[col1, 'id']))]['id'])
print()
print()
print("second correlation group's impact :", sum(list(corr.loc[col2, 'id'])))
print()
print("maximum of second correlation group's impact")
print(corr[corr['id'] == max(list(corr.loc[col2, 'id']))]['id'])


# As a result, the impact of *second correlation group* is bigger than *first group*.
# And the most reason of *second group* is "**contact with patien**t".
