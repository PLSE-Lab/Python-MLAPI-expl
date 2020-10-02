#!/usr/bin/env python
# coding: utf-8

# # COVID 19 Analysis in Ghana - What are The Trends? 

# ![Image](https://www.furman.edu/covid-19/wp-content/uploads/sites/177/2020/03/CoronaVirusHeader-Final-3.jpg)

# # Data Mining Analysis using Data Mining Techniques and Algorithms

# Using Data Mining techniques and algorithms to find interesting patterns (knownledge) from the COVID-19 raw data derived from Ghana.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # mathematical operations and linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization library
import seaborn as sns # Fancier visualizations
import statistics # fundamental stats package
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import scipy.stats as stats # to calculate chi-square test stat
from datetime import date
import plotly.graph_objects as go
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Load raw data from url and use first row as column names

rawData = pd.read_csv('../input/jhu-csse-rawdata/coronavirus.csv', sep=',',
                           header=0, encoding='ascii', engine='python')
df = rawData


# In[ ]:


df


# Preprocessing the raw data involves cleaning, reducing and transforming the data to a specifies dataset which comprises of Ghana only.

# In[ ]:


df.sort_values(['country','type','date'],inplace = True)
df.reset_index(drop=True, inplace=True) # reset index from 0 to -1
print(df)


# In[ ]:


df.info() #Display Data Types and Columns


# The "date" attribute was set to an object datatype by default, therefore, it needs to be changed to a datetime64 datatype.
# 
# The "province" attribute has some missing values; missig values can cause inconsistency, so there they need to be replaced.

# In[ ]:


df['date'] = df['date'].astype('datetime64')
df = df.fillna('unknown')
df.info()


# In[ ]:


df.columns # get column names 


# In[ ]:


df_conf_ttl = df[df.type == 'confirmed'].cases.sum()
df_deat_ttl = df[df.type == 'death'].cases.sum()
df_rcvd_ttl = df[df.type == 'recovered'].cases.sum()
ObservationDate = df['date'].max() #Latest date
df_ac_ttl = df_conf_ttl  - (df_deat_ttl + df_rcvd_ttl) # Active cases

labels = ["Last Update","Confirmed","Active cases","Recovered","Deaths"]
fig = go.Figure(data=[go.Table(header=dict(values=labels),
                 cells=dict(values=[ObservationDate,df_conf_ttl,df_ac_ttl,df_rcvd_ttl,df_deat_ttl]))
                     ])
fig.update_layout(
    title='Total Number of COVID 19 Cases in The world',
)
fig.show()


# Number of days COVID-19 has been tracked in Ghana as of Jan 22, 2020.

# In[ ]:


df['date'].max()-df['date'].min() # number of tracked day data has been tracked


# # Data reduction by Country: Ghana

# In[ ]:


dfghana = df[df.country == 'Ghana']
dfghana = dfghana.reset_index(drop=True, inplace=None) #reduce to 74 x 4
dfghana = dfghana[['date','type','cases']]
dfghana


# The raw data has been reduced to only entries for Ghana.

# In[ ]:


# Confirmed Cases in Ghana
dfghana_conf_ttl = dfghana[dfghana.type =='confirmed']
dfghana_conf_ttl = dfghana_conf_ttl[['date','cases']]
dfghana_conf_ttl


# In[ ]:


# Death Cases in Ghana
dfghana_deat_ttl = dfghana[dfghana.type =='death']
dfghana_deat_ttl = dfghana_deat_ttl[['date','cases']]
dfghana_deat_ttl


# In[ ]:


# Recovered Cases in Ghana
dfghana_rcvd_ttl = dfghana[dfghana.type =='recovered']
dfghana_rcvd_ttl =  dfghana_rcvd_ttl[['date','cases']]
dfghana_rcvd_ttl


# In[ ]:


#Actice Cases in Ghana
dfghana_actv_ttl = (dfghana_conf_ttl['cases'].sum() - (dfghana_deat_ttl['cases'].sum() 
                    + dfghana_rcvd_ttl['cases'].sum()))

dfghana_actv_ttl


# In[ ]:


dfghana_merged = pd.merge(dfghana_conf_ttl,dfghana_deat_ttl, on = 'date', how = 'right')
dfghana_merged = pd.merge(dfghana_merged,dfghana_rcvd_ttl, on = 'date', how = 'right')
dfghana_merged


# In[ ]:


#rename columns
dfghana_merged = dfghana_merged.rename(columns ={'cases_x':'confirmed',
                                          'cases_y': 'death',
                                          'cases':'recovered'})


# In[ ]:


dfghana_merged


# In[ ]:


dfghana_merged.describe()


# In[ ]:


#Frequency and five number summary boxplot
dfghana_merged[['confirmed', 'death', 'recovered']].hist(layout=(1,3), sharex=False, sharey=False, figsize=(15, 5), bins=20) 
plt.show()

dfghana_merged[['confirmed', 'recovered', 'death']].plot(kind = 'box',subplots=True, layout=(1,3), sharex=False, sharey=False, figsize=(15,5))
plt.show()

There are a lot of outerliers in all cases. This dataset is not adequate enough to perform other statistical calculations without normalizing the data. 
# In[ ]:


print('-----------Skewness-------------')
print(dfghana_merged.skew(axis = 0, skipna = True))
print('\n-----------Kurtosis-------------')
print(dfghana_merged.kurtosis(skipna = True))


# All cases are positively skewed and heavy-tailed relative to a normal distribution.

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=dfghana_merged['date'],y=dfghana_merged['confirmed'],
             mode='lines+markers',
             name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=dfghana_merged['date'],y=dfghana_merged['death'],
             mode='lines+markers',
             name='Death Cases'))
fig.add_trace(go.Scatter(x=dfghana_merged['date'],y=dfghana_merged['recovered'],
             mode='lines+markers',
             name='Recovery Cases'))

fig.update_xaxes(
    rangeslider_visible=True
)

fig.show()

