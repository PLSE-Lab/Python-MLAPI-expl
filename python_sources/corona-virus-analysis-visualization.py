#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# <font color = 'red'>
#     
# Content:
# 
# 1. [Load and Check Data](#1)
# 2. [Variable Description](#2)
# 3. [Basic Data Analysis](#3)
# 4. [Visualization](#4)
# 5. [Linear Regression](#5)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import pycountry
py.init_notebook_mode(connected=True)



import seaborn as sns
sns.set()

from collections import Counter

import plotly.express as px
import folium
from folium import plugins

from sklearn.linear_model import LinearRegression

from datetime import date

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = '1'> </a>
# 
# # Load and Check Data

# In[ ]:


data_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
data_df .head()


# In[ ]:


data_df.columns


# In[ ]:


data_df.describe


# In[ ]:


data_df.info()


# In[ ]:



#data_df.drop(['Sno'],axis=1,inplace=True)
data_df.head()


# <a id = '2'></a>
# 
# # Variable Description
# 
# 
# 
# 

# > 1. Province/State - City of virus suspected cases.
# 2. Country/Region - Country of virus suspected cases.
# 3. last update - Date of update of patient infected
# 4. Confirmed - Confirmation by doctors that this patient is infected with deadly virus
# 5. Suspected - Number of cases registered
# 6. Recovered - Recovery of the patient
# 7. Deaths - Death of the patient .

# In[ ]:


data_df['Country'].replace({'Mainland China':'China'},inplace=True)
country = data_df['Country'].unique().tolist()
print(country)
print("\n Affected contries by virus: ", len(country))


# <a id = '3'></a>
# # Basic Data Analysis

# In[ ]:


union_deneme = []

union_deneme = data_df[["Country","Confirmed"]].groupby(["Country"], as_index = False).sum().sort_values(by="Confirmed", ascending = False).reset_index(drop=True)
union_deneme


# * The cases were reported mostly from China and its neighbors.

# In[ ]:


death_data = data_df[['Country', 'Deaths']].groupby(["Country"], as_index = False).sum().sort_values(by="Deaths", ascending=False).reset_index(drop=True)
death_data = death_data[death_data['Deaths']>0]
death_data


# * Outside of China, the number of deaths caused by the virus is low.

# In[ ]:


import datetime
data_df['Last Update'] = pd.to_datetime(data_df['Last Update']) 
data_df['Date'] = [datetime.datetime.date(d) for d in data_df ['Last Update']]
data_df['Time'] = [datetime.datetime.time(d) for d in data_df['Last Update']]


# In[ ]:


data_df['Date'] = data_df['Date'].astype(str)
day = data_df["Date"].values
day = [my_str.split("-")[2] for my_str in day]
data_df["Date"] = day


# In[ ]:


dates = data_df['Date'].unique()
dates = np.flipud(dates) 

dates


# <a id = '4'></a>
# # Visualization

# In[ ]:


all_cases = []

for i in dates:
    all_cases.append(data_df[data_df['Date']==i].Confirmed.sum())

plt.figure(figsize=(15, 10));
plt.plot(dates, all_cases);
plt.title('Daily Case Numbers', size=15);
plt.xlabel('Days', size= 10)
plt.ylabel('Number of Cases', size=15);
plt.show();


# In[ ]:


all_cases = []

for i in dates:
    all_cases.append(data_df[data_df['Date']==i].Deaths.sum())

plt.figure(figsize=(15, 10));
plt.plot(dates, all_cases);
plt.title('Daily Death Numbers', size=15);
plt.xlabel('Days', size= 10)
plt.ylabel('Number of Cases', size=15);
plt.show();


# ## Countries Where Cases Spread

# In[ ]:



fig = px.scatter_geo(data_df, locations= union_deneme["Country"], locationmode='country names', 
                     color= union_deneme["Confirmed"], hover_name= union_deneme["Country"], range_color= [0, 500], projection="natural earth",
                    title='Countries Where Cases Spread')
fig.show()


# ## Countries where Deaths Occurred

# In[ ]:



fig = px.scatter_geo(data_df, locations= death_data["Country"], locationmode='country names', 
                     color= death_data["Deaths"], hover_name= death_data["Country"], range_color= [0, 500], projection="natural earth",
                    title='Countries where Deaths Occurred')
fig.show()


# ## Number of Recovered of Cases

# In[ ]:


f, ax = plt.subplots(figsize=(10, 16))


sns.barplot(x="Confirmed", y="Province/State", data=data_df[1:],
            label="Confirmed", color="b")


sns.barplot(x="Recovered", y="Province/State", data=data_df[1:],
            label="Recovered", color="g")

ax.legend(ncol=6, loc="lower right", frameon=True)
ax.set(xlim=(0, 1800), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)


# In[ ]:


stateCountry = data_df.groupby(['Country', 'Province/State']).size().unstack()


# In[ ]:


plt.figure(figsize=(15,10))
sc = sns.heatmap(stateCountry,square=True, cbar_kws={'fraction' : 0.01}, cmap='afmhot_r', linewidth=1)


# <a id = '5'></a>
# ## Linear Regression

# In[ ]:


data_df.shape


# In[ ]:


data_df.describe()


# In[ ]:


data_df.isnull().sum()


# * we have empty values in province/state field
# 

# In[ ]:


data_df.drop(['Sno'], axis=1, inplace=True)


# In[ ]:


data_df['Last Update'] =data_df['Last Update'].apply(pd.to_datetime)


# In[ ]:


data_df.tail()


# In[ ]:


data_df['Province/State'].value_counts()


# ## Implementing Data Exploration

# In[ ]:


countries = data_df['Country'].unique().tolist()
print(countries)

print("\nTotal countries affected by virus: ",len(countries))


# ## Latest Data  On nCoV Virus

# In[ ]:


data_df


# In[ ]:


# Latest Numbers

print('Confirmed Cases around the globe : ',data_df['Confirmed'].sum())
print('Deaths Confirmed around the globe: ',data_df['Deaths'].sum())
print('Recovered Cases around the globe : ',data_df['Recovered'].sum())


# In[ ]:


tempState = data_df['Province/State'].mode()
print(tempState)


# In[ ]:


## Countries Currently affected by it.
allCountries = data_df['Country'].unique().tolist()
print(allCountries)

print("\nTotal countries affected by virus: ",len(allCountries))


# In[ ]:


CountryWiseData = pd.DataFrame(data_df.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum())
CountryWiseData['Country'] = CountryWiseData.index
CountryWiseData.index = np.arange(1, len(allCountries)+1)

CountryWiseData = CountryWiseData[['Country','Confirmed', 'Deaths', 'Recovered']]

#formatted_text('***Country wise Analysis of ''Confirmed'', ''Deaths'', ''Recovered'' Cases***')
CountryWiseData


# In[ ]:


data_df.plot(subplots=True,figsize=(18,18))
plt.show()

