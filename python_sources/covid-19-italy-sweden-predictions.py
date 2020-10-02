#!/usr/bin/env python
# coding: utf-8

#  thanks to https://www.kaggle.com/imdevskp/covid-19-analysis-viz-prediction-comparisons

# ### Import

# In[ ]:


# essential libraries
import json
import random
from urllib.request import urlopen

# storing and anaysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()   

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# html embedding
from IPython.display import Javascript
from IPython.core.display import display
from IPython.core.display import HTML


# # Dataset

# In[ ]:


# list files
 get_ipython().system('ls ../input/corona-virus-report')


# In[ ]:


# importing datasets
full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 
                         parse_dates=['Date'])
full_table.head()


# In[ ]:


# dataframe info
full_table.info()


# In[ ]:


# checking for missing value
 full_table.isna().sum()


# # Preprocessing

# In[ ]:


full_table['Country/Region'].nunique()


# In[ ]:


print('these stats are updated to:', full_table['Date'].max())


# In[ ]:


full_table['Country/Region'].value_counts().head(10)


# ### Cleaning Data

# In[ ]:


# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case = confirmed - deaths - recovered
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']


# filling missing values 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
full_table[cases] = full_table[cases].fillna(0)


# In[ ]:


print ('list of countries', full_table['Country/Region'].unique())


# ### Derived Tables

# In[ ]:


# cases in the ships
ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]

# china and the row
china = full_table[full_table['Country/Region']=='China']
row = full_table[full_table['Country/Region']!='China']
sweden = full_table[full_table['Country/Region']=='Sweden']
italy = full_table[full_table['Country/Region']=='Italy']
uk = full_table[full_table['Country/Region']=='United Kingdom']
france = full_table[full_table['Country/Region']=='France']
brazil = full_table[full_table['Country/Region']=='Brazil']
spain = full_table[full_table['Country/Region']=='Spain']
russia = full_table[full_table['Country/Region']=='Russia']
poland = full_table[full_table['Country/Region']=='Poland']
US= full_table[full_table['Country/Region']=='US']
netherlands = full_table[full_table['Country/Region']=='Netherlands']
germany = full_table[full_table['Country/Region']=='Germany']
# latest
full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
china_latest = full_latest[full_latest['Country/Region']=='China']
Sweden_latest = full_latest[full_latest['Country/Region']=='Sweden']
Italy_latest = full_latest[full_latest['Country/Region']=='Italy']
UK_latest = full_latest[full_latest['Country/Region']=='United Kingdom']
France_latest = full_latest[full_latest['Country/Region']=='France']
Belgium_latest = full_latest[full_latest['Country/Region']=='Belgium']
row_latest = full_latest[full_latest['Country/Region']!='China']

# latest condensed
full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()


# ## latest raw data for Sweden and Italy

# # Latest Data

# ### Latest Complete Data

# In[ ]:


temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
# temp.style.background_gradient(cmap='Reds')
temp.tail()


# # Cases over the time

# In[ ]:


thresold = 1000 # thresold value of confirmed cases based on US initial count


# # Sweden

# In[ ]:



df_sweden = sweden.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_sweden = df_sweden[df_sweden['Confirmed'] > thresold].reset_index()

y_sweden = df_sweden['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_sweden = np.diff(y_sweden) # now we can get the derivative as a new numpy array

output_sweden = np.transpose(deriv_y_sweden)
#now add the numpy array to our dataframe
df_sweden['ContagionRate'] = pd.Series(output_sweden)


# In[ ]:


plt.figure(figsize= (5,10))
plt.subplot(211)
plt.plot(df_sweden['Date'],df_sweden['Confirmed'], color = 'g') #trend cases
plt.title('Cases over time')
plt.ylabel('number of cases')
plt.xticks(df_sweden['Date']," ")
plt.subplot(212)
plt.plot(df_sweden['Date'],df_sweden['ContagionRate'], color = 'r') #trend deaths
plt.title('Spread rate over time')
plt.ylabel('Rate (cases % increase)')
plt.xticks(rotation=90)

plt.suptitle('Virus spread over time Sweden')
plt.show()


# # Italy

# In[ ]:


df_italy = italy.groupby(['Date']).mean().reset_index()
df_italy = df_italy[df_italy['Confirmed'] > thresold].reset_index()

y_italy = df_italy['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_italy = np.diff(y_italy) # now we can get the derivative as a new numpy array

output_italy = np.transpose(deriv_y_italy)
#now add the numpy array to our dataframe
df_italy['ContagionRate'] = pd.Series(output_italy)


# In[ ]:


plt.figure(figsize= (5,10))
plt.subplot(211)
plt.plot(df_italy['Date'],df_italy['Confirmed'], color = 'g') #trend cases
plt.title('Cases over time')
plt.ylabel('number of cases')
plt.xticks(df_italy['Date']," ")
plt.subplot(212)
plt.plot(df_italy['Date'],df_italy['ContagionRate'], color = 'r') #trend deaths
plt.title('Spread rate over time')
plt.ylabel('Rate (cases % increase)')
plt.xticks(rotation=90)

plt.suptitle('Virus spread over time Italy')
plt.show()


# ## More countries for an international comparison

# ## China

# In[ ]:


df_china = china.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_china = df_china[df_china['Confirmed'] > thresold].reset_index()

y_china = df_china['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_china = np.diff(y_china) # now we can get the derivative as a new numpy array

output_china = np.transpose(deriv_y_china)
#now add the numpy array to our dataframe
df_china['ContagionRate'] = pd.Series(output_china)
df_china = df_china[df_china['ContagionRate'] < 4500] # clean the chinese data from the suspicious "spike" of 12/2


# # Poland

# In[ ]:


df_poland = poland.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_poland = df_poland[df_poland['Confirmed'] > thresold].reset_index()
y_poland = df_poland['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_poland = np.diff(y_poland) # now we can get the derivative as a new numpy array

output_poland = np.transpose(deriv_y_poland)
#now add the numpy array to our dataframe
df_poland['ContagionRate'] = pd.Series(output_poland)


# ## France

# In[ ]:


df_france = france.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_france = df_france[df_france['Confirmed'] > thresold].reset_index()
y_france = df_france['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_france = np.diff(y_france) # now we can get the derivative as a new numpy array

output_france = np.transpose(deriv_y_france)
#now add the numpy array to our dataframe
df_france['ContagionRate'] = pd.Series(output_france)
population_france = 67064000


# # United Kingdom

# In[ ]:


df_uk = uk.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_uk = df_uk[df_uk['Confirmed'] > thresold].reset_index()
y_uk = df_uk['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_uk = np.diff(y_uk) # now we can get the derivative as a new numpy array

output_uk = np.transpose(deriv_y_uk)
#now add the numpy array to our dataframe
df_uk['ContagionRate'] = pd.Series(output_uk)


# ## United States

# In[ ]:


df_us = US.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_us = df_us[df_us['Confirmed'] > thresold].reset_index()

y_us = df_us['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_us = np.diff(y_us) # now we can get the derivative as a new numpy array

output_us = np.transpose(deriv_y_us)
#now add the numpy array to our dataframe
df_us['ContagionRate'] = pd.Series(output_us)


# # Germany

# In[ ]:


df_germany = germany.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_germany = df_germany[df_germany['Confirmed'] > thresold].reset_index()

y_germany = df_germany['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_germany = np.diff(y_germany) # now we can get the derivative as a new numpy array

output_germany = np.transpose(deriv_y_germany)
#now add the numpy array to our dataframe
df_germany['ContagionRate'] = pd.Series(output_germany)


# ## Brazil

# In[ ]:


df_brazil = brazil.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_brazil = df_brazil[df_brazil['Confirmed'] > thresold].reset_index()

y_brazil = df_brazil['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_brazil = np.diff(y_brazil) # now we can get the derivative as a new numpy array

output_brazil = np.transpose(deriv_y_brazil)
#now add the numpy array to our dataframe
df_brazil['ContagionRate'] = pd.Series(output_brazil)


# ## The Netherlands

# In[ ]:


df_netherlands = netherlands.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_netherlands = df_netherlands[df_netherlands['Confirmed'] > thresold].reset_index()

y_netherlands = df_netherlands['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_netherlands = np.diff(y_netherlands) # now we can get the derivative as a new numpy array

output_netherlands = np.transpose(deriv_y_netherlands)
#now add the numpy array to our dataframe
df_netherlands['ContagionRate'] = pd.Series(output_netherlands)


# ## Russia

# In[ ]:


df_russia = russia.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_russia = df_russia[df_russia['Confirmed'] > thresold].reset_index()

y_russia = df_russia['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_russia = np.diff(y_russia) # now we can get the derivative as a new numpy array

output_russia = np.transpose(deriv_y_russia)
#now add the numpy array to our dataframe
df_russia['ContagionRate'] = pd.Series(output_russia)


# ## Spain

# In[ ]:


df_spain = spain.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df_spain = df_spain[df_spain['Confirmed'] > thresold].reset_index()

y_spain = df_spain['Confirmed'].values # transform the column to differentiate into a numpy array

deriv_y_spain = np.diff(y_spain) # now we can get the derivative as a new numpy array

output_spain = np.transpose(deriv_y_spain)
#now add the numpy array to our dataframe
df_spain['ContagionRate'] = pd.Series(output_spain)


# # Final Comparison and model

# 
# 
# The most complete dataset at the time Covid-19 started to spread in Europe is that of China. We fit a model to this data, that closely resemble a bell curve
# 

# ## Raw data comparison

# In[ ]:


X_spain = df_spain.index.values
y_spain = df_spain['Confirmed'].values
#X_china = X_china[0:10]
#y_china = y_china[0:10]
#y_china


# In[ ]:


X_us = df_us.index.values
y_us = df_us['Confirmed'].values

X_uk = df_uk.index.values
y_uk = df_uk['Confirmed'].values


# In[ ]:



from scipy.optimize import curve_fit



# I define the exponential function
def func(x, a, b, c): 
    return a * np.exp(-b * x) + c

#do the fit!
#popt_us, pcov_us = curve_fit(func, X_us, y_us)
#popt_spain, pcov_spain = curve_fit(func, X_spain, y_spain, maxfev=1200)
#popt_uk, pcov_uk = curve_fit(func, X_uk, y_uk)


# ### Thresold Dates by country
# The day the country reached at least 1000 cases

# In[ ]:


th_china = df_china['Date'].iloc[0]
print('China:', th_china )
th_italy = df_italy['Date'].iloc[0]
print('Italy:', th_italy )
th_germany = df_germany['Date'].iloc[0]
print('Germany:', th_germany )
th_us = df_us['Date'].iloc[0]
print('USA:', th_us )
th_uk = df_uk['Date'].iloc[0]
print('UK:', th_uk )
th_sweden = df_sweden['Date'].iloc[0]
print('Sweden:', th_sweden )
th_brazil = df_brazil['Date'].iloc[0]
print('Brazil:', th_brazil )
th_spain = df_spain['Date'].iloc[0]
print('Spain:', th_spain )
th_russia = df_russia['Date'].iloc[0]
print('Russia:', th_russia )
th_nl = df_netherlands['Date'].iloc[0]
print('Netherlands:', th_nl )
th_pol = df_poland['Date'].iloc[0]
print('Poland:', th_pol) 
th_fr = df_france['Date'].iloc[0]
print('France:', th_fr)
thdates = [th_china, th_sweden, th_italy, th_uk,th_pol, th_germany, th_brazil, th_us, th_spain, th_nl, th_fr]


# In[ ]:


population_russia = 145934462


# In[ ]:


plt.figure(figsize=(18, 18))

plt.subplot(221)

plt.plot(df_china.index, df_china['Confirmed'], label = 'China (2020-01-25)')
plt.plot(df_sweden.index, df_sweden['Confirmed'], label ='Sweden (2020-03-15)')

plt.plot(df_italy.index, df_italy['Confirmed'], label = 'Italy (2020-02-29)')
plt.plot(df_uk.index, df_uk['Confirmed'], label = 'UK (2020-03-14)')
plt.plot(df_germany.index, df_germany['Confirmed'], label ='Germany (2020-03-08)')
plt.plot(df_spain.index, df_spain['Confirmed'], label = 'Spain (2020-03-09)')
plt.plot(df_russia.index, df_russia['Confirmed'], label ='Russia (1000 reached: 2020-03-27)')
plt.plot(df_brazil.index, df_brazil['Confirmed'], label = 'Brazil (1000 reached: 2020-03-21)')
plt.plot(df_us.index, df_us['Confirmed'], color = 'k', label = 'USA (1000 reached: 2020-03-11)')
#plt.plot(X_spain, func(X_spain, *popt_spain), '--', label='Spain fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_spain))
#plt.plot(X_us, func(X_us, *popt_us), '--', label='US fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_us))
#plt.plot(X_uk, func(X_uk, *popt_uk), '--', label='UK fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_uk))
plt.xlim(0, 150)
plt.ylim(0, 3500000)
plt.ylabel('Number of cases', fontsize=14)
plt.xlabel('Days from reaching the thresold (1000 cases)', fontsize = 14)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.subplot(222)
plt.plot(df_sweden.index, df_sweden['Confirmed']/10088474, label ='Sweden (1000 reached: 2020-03-15)')
plt.plot(df_italy.index, df_italy['Confirmed']/60483973,'--', label = 'Italy (1000 reached: 2020-02-29)')
plt.plot(df_russia.index, df_russia['Confirmed']/population_russia,'--', label = 'Russia (1000 reached: 2020-03-27)')
#plt.plot(df_poland.index, df_poland['Confirmed']/37115000 ,':', label = 'Poland(1000 reached: 2020-03-25)')
plt.plot(df_france.index, df_france['Confirmed']/population_france ,':', label = 'France(1000 reached: 2020-03-08)')
plt.plot(df_uk.index, df_uk['Confirmed']/66575226,':', label = 'UK (1000 reached: 2020-03-14)')
plt.plot(df_us.index, df_us['Confirmed']/328953020, color = 'k', label = 'USA (1000 reached: 2020-03-11)')
plt.plot(df_brazil.index, df_brazil['Confirmed']/212434518,color = 'orange', label = 'Brazil (1000 reached: 2020-03-21)')
#plt.plot(X_spain, func(X_spain, *popt_spain), '--', label='Spain fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_spain))
#plt.plot(X_us, func(X_us, *popt_us), '--', label='US fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_us))
#plt.plot(X_fr, func(X_fr, *popt_fr), '--', label='France fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_fr))
plt.xlim(0, 150)
#plt.ylim(0, 250000)
#plt.xscale('log')
plt.yscale('log')
plt.ylabel('Log num. cases/ population', fontsize=14)
plt.xlabel('Days from reaching the thresold (1000 cases)', fontsize = 14)
plt.legend

plt.title('International comparison of cases growth', fontsize = 20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


plt.show()


# In[ ]:


plt.figure(figsize=(6, 6))

plt.plot(df_sweden.index, df_sweden['ContagionRate'], label ='Sweden (1000 reached: 2020-03-15)')
plt.plot(df_italy.index, df_italy['ContagionRate'], label = 'Italy (1000 reached: 2020-02-29)')
#plt.plot(df_uk.index, df_uk['ContagionRate'], label = 'UK (1000 reached: 2020-03-14)')
plt.plot(df_spain.index, df_spain['ContagionRate'], label = 'Spain (1000 reached: 2020-03-09)')
#plt.plot(df_germany.index, df_germany['ContagionRate'], label ='Germany (2020-03-08)')
#plt.plot(df_netherlands.index, df_netherlands['ContagionRate'], label ='Netherlands (1000 reached: 2020-03-15)')
#plt.plot(df_us.index, df_us['ContagionRate'], color = 'k', label = 'USA (1000 reached: 2020-03-11)')
#plt.plot(X_spain, func(X_spain, *popt_spain), '--', label='Spain fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_spain))
#plt.plot(X_us, func(X_us, *popt_us), '--', label='US fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_us))
#plt.plot(X_fr, func(X_fr, *popt_fr), '--', label='France fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_fr))
#plt.xlim(0, 50)
#plt.ylim(0, 250000)
#plt.xscale('log')
#plt.yscale('log')
plt.ylabel('Daily new cases (COVID-19 spread rate)', fontsize=14)
plt.ylim(0, 10000)
plt.xlabel('Days from reaching the thresold (1000 cases)', fontsize = 14)
plt.legend

plt.title('International comparison of contagion rates', fontsize = 20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


plt.show()


# In[ ]:


df_italy_clean = df_italy[df_italy['ContagionRate'] > 100] 
df_uk_clean = df_uk[df_uk['ContagionRate'] > 100] 
df_us_clean = df_us[df_us['ContagionRate'] > 100] 
df_spain_clean = df_spain[df_spain['ContagionRate'] > 100] 
df_sweden_clean = df_sweden[df_sweden['ContagionRate'] > 100] 
df_france_clean = df_germany[df_germany['ContagionRate'] > 100] 


# In[ ]:


plt.figure(figsize=(6, 6))

plt.plot(df_sweden_clean['Confirmed'], df_sweden_clean['ContagionRate'], label ='Sweden (1000 reached: 2020-03-15)')
plt.plot(df_italy_clean['Confirmed'], df_italy_clean['ContagionRate'], label = 'Italy (1000 reached: 2020-02-29)')
#plt.plot(df_uk_clean['Confirmed'], df_uk_clean['ContagionRate'], label = 'UK (1000 reached: 2020-03-14)')
plt.plot(df_france_clean['Confirmed'], df_france_clean['ContagionRate'], label = 'France (1000 reached: 2020-03-08)')
#plt.plot(df_netherlands.index, df_netherlands['ContagionRate'], label ='Netherlands (1000 reached: 2020-03-15)')
plt.plot(df_us_clean['Confirmed'], df_us_clean['ContagionRate'], color = 'k', label = 'USA (1000 reached: 2020-03-11)')
#plt.plot(X_spain, func(X_spain, *popt_spain), '--', label='Spain fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_spain))
#plt.plot(X_us, func(X_us, *popt_us), '--', label='US fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_us))
#plt.plot(X_fr, func(X_fr, *popt_fr), '--', label='France fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_fr))
#plt.xlim(0, 50)
#plt.ylim(0, 250000)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Daily new cases', fontsize=14)
plt.xlabel('Total cases', fontsize = 14)
plt.legend

plt.title('International comparison of contagion rates', fontsize = 20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


plt.show()


# > ## All countries overview

# In[ ]:




temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.to_csv('generaltable.csv')


# In[ ]:


get_ipython().system('pip install imgkit')


# 
# ## Current statistics for countries with over 1000 COVID-19 cases

# In[ ]:



temp_f['%mortality'] = (temp_f['Deaths']/ temp_f['Confirmed'])*100
temp_f['%recovered'] = (temp_f['Recovered']/ temp_f['Confirmed'])*100
temp_f = temp_f[temp_f['Confirmed'] > thresold]
temp_f = temp_f.sort_values(by='%recovered', ascending=False) 
temp_f = temp_f.reset_index(drop=True)
colortable = temp_f.style.background_gradient(cmap='GnBu')

colortable


# # Analysis on similar epidemics

# https://www.kaggle.com/imdevskp/mers-outbreak-analysis  
# https://www.kaggle.com/imdevskp/sars-2003-outbreak-analysis  
# https://www.kaggle.com/imdevskp/western-africa-ebola-outbreak-analysis
# 
