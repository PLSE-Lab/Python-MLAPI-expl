#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Import-libraries-and-load-data" data-toc-modified-id="Import-libraries-and-load-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import libraries and load data</a></span></li><li><span><a href="#Overview-of-data-information" data-toc-modified-id="Overview-of-data-information-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Overview of data information</a></span></li><li><span><a href="#Processing" data-toc-modified-id="Processing-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Processing</a></span><ul class="toc-item"><li><span><a href="#Selecting-single-country-and-data-for-research" data-toc-modified-id="Selecting-single-country-and-data-for-research-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Selecting single country and data for research</a></span></li><li><span><a href="#Add-and-changing-index-for-better-useful" data-toc-modified-id="Add-and-changing-index-for-better-useful-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Add and changing index for better useful</a></span></li><li><span><a href="#Visualising-dynamic-of-new-cases" data-toc-modified-id="Visualising-dynamic-of-new-cases-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Visualising dynamic of new cases</a></span></li><li><span><a href="#Finding-the-illness-coefficient" data-toc-modified-id="Finding-the-illness-coefficient-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Finding the illness coefficient</a></span></li><li><span><a href="#Finding-the-dynamic-of-illness-coefficient" data-toc-modified-id="Finding-the-dynamic-of-illness-coefficient-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Finding the dynamic of illness coefficient</a></span></li><li><span><a href="#Forecast-date-when-only-1%-of-cases-will-remain" data-toc-modified-id="Forecast-date-when-only-1%-of-cases-will-remain-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Forecast date when only 1% of cases will remain</a></span></li><li><span><a href="#Forecast-date-when-last-case-will-remain" data-toc-modified-id="Forecast-date-when-last-case-will-remain-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Forecast date when last case will remain</a></span></li></ul></li><li><span><a href="#Summary" data-toc-modified-id="Summary-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Summary</a></span></li></ul></div>

# ## Introduction
# 
# This research focused on forecast the end date of coronavirus pandemic.
# Official data, prepared by World Helth Organization (WHO) is used:
# * https://data.humdata.org/organization/c021f6be-3598-418e-8f7f-c7a799194dba,
# * https://covid19.who.int/
# 
# The model of bacterial growth is used as mathematical model for finding the growth coefficient of cases and forecast the end date.
# 
# $$N = N_0 \cdot \exp^{k T},$$
# 
# where: $N$ - number of cases at the end of period $T$, $N_0$ - number of cases at start of period $T$, $k$ - the growth coefficient of cases

# ## Import libraries and load data

# In[ ]:


import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Read the data of Coronavirus (COVID-19) cases and deaths, prepared by the World Health Organization

# In[ ]:


url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSe-8lf6l_ShJHvd126J-jGti992SUbNLu-kmJfx1IRkvma_r4DHi0bwEW89opArs8ZkSY5G2-Bc1yT/pub?gid=0&single=true&output=csv"
df = pd.read_csv(url, index_col = 'ADM0_NAME', parse_dates = ['date_epicrv'], dayfirst=True)


# ## Overview of data information

# In[ ]:


df.info()


# ## Processing

# ### Selecting single country and data for research

# In[ ]:


df_SingleCountry = df.loc['Russian Federation'][['date_epicrv', 'NewCase','CumCase']]


# In[ ]:


df_SingleCountry


# ### Add and changing index for better useful

# In[ ]:


df_SingleCountry['index'] = range(len(df_SingleCountry))


# In[ ]:


df_SingleCountry.info()


# In[ ]:


df_SingleCountry.set_index('index', inplace = True)


# In[ ]:


df_SingleCountry.info()


# ### Visualising dynamic of new cases

# Deleting the last row of dataset, because current date is still going.

# In[ ]:


df_SingleCountry.drop(len(df_SingleCountry)-1, inplace = True)


# Visualization of the dynamics of new cases on the chart

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 7))
ax.set_title("Dynamic of new cases", fontsize=16)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Cases per day", fontsize=14)

ax.plot_date(df_SingleCountry['date_epicrv'], df_SingleCountry['NewCase'], fmt='.-')
plt.show()


# ### Finding the illness coefficient

# In[ ]:


t = len(df_SingleCountry)-1
No = df_SingleCountry.at[0,'CumCase']
N = df_SingleCountry.at[t,'CumCase']
k = math.log(N/No)/t
print("Current illness coefficient is {}".format(k))


# It decreases at time if pandemic also decreased.

# ### Finding the dynamic of illness coefficient

# Calculating illness coefficient for each day

# In[ ]:


df_k = pd.DataFrame()

for i in range(t):
    if(i > 0):
        N = df_SingleCountry.at[i,'CumCase']
        df_k.at[i, 1] = math.log(N/No)/i
    else:
        df_k.at[i, 1] = 0
    df_k.at[i, 0] = df_SingleCountry.at[i, 'date_epicrv']


# Visualising illness coefficient on chart

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 7))
ax.set_title("Dynamic of illness coefficient", fontsize=16)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Coefficient value", fontsize=14)

ax.plot_date(df_k[0], df_k[1], fmt='.-')
plt.show()


# Forecast percent of illness people that will be at the end of July

# In[ ]:


to_end_July = datetime.date(2020, 7, 31) - datetime.date.today() - datetime.timedelta(days = 1)

if to_end_July.days > 0:
    Percent = round(math.exp(-k * to_end_July.days) * 100, 1)
    print("Forecast percent of illness people that will be at the end of July is {} %".format(Percent))
else:
    print('July has gone...')


# ### Forecast date when only 1% of cases will remain

# In[ ]:


T = round(math.log((N*0.01)/N)/-k, 1)
end_date_to1per = datetime.date.today() + datetime.timedelta(days=T)
print("Forecast date when only 1% of cases will remain is {}".format(end_date_to1per.strftime("%d.%m.%Y")))


# ### Forecast date when last case will remain

# In[ ]:


T = round(math.log(1/N)/-k, 1)
end_date = datetime.date.today() + datetime.timedelta(days=T)
print("Forecast date when last case will remain is {}".format(end_date.strftime("%d.%m.%Y")))


# ## Summary

# In[ ]:


print("This calculation was completed on {}".format(datetime.date.today().strftime("%d.%m.%Y")))
print("Current illness coefficient is {}".format(k))
if to_end_July.days > 0:
    print("Forecast percent of illness people that will be at the end of July is {} %".format(Percent))
print("Forecast date when only 1% of cases will remain is {}".format(end_date_to1per.strftime("%d.%m.%Y")))
print("Forecast date when last case will remain is {}".format(end_date.strftime("%d.%m.%Y")))


# In[ ]:


fig, ax1 = plt.subplots(figsize=(15, 5))

ax1.set_title("Dynamic of new cases", fontsize=16)
ax1.set_ylabel("Cases per day", fontsize=14)
ax1.plot_date(df_SingleCountry['date_epicrv'], df_SingleCountry['NewCase'], fmt='.-')
ymin = 0
ymax = 12000
ax1.vlines(datetime.date(2020, 5, 1), ymin, ymax, linestyles='dotted', label="Start May holidays", color='r')
ax1.vlines(datetime.date(2020, 5, 22), ymin, ymax, linestyles='dotted', label="End of self-isolation in Moscow region", color='g') 
ax1.vlines(datetime.date(2020, 6, 8), ymin, ymax, linestyles='dotted', label="End of self-isolation in Moscow", color='b')
ax1.vlines(datetime.date(2020, 7, 1), ymin, ymax, linestyles='dotted', label="Voiting")
plt.legend(loc='upper left')


fig, ax2 = plt.subplots(figsize=(15, 5))
ax2.set_title("Dynamic of illness coefficient", fontsize=16)
#ax2.set_xlabel("Date", fontsize=14)
ax2.set_ylabel("Coefficient value", fontsize=14)
ax2.plot_date(df_k[0], df_k[1], fmt='.-')


# View period since 01.05.2020
df2_k = df_k[df_k[0] >= datetime.date(2020, 5, 1)]

fig, ax3 = plt.subplots(figsize=(15, 5))
ax3.set_title("Dynamic of illness coefficient since May", fontsize=16)
ax3.set_xlabel("Date", fontsize=14)
ax3.set_ylabel("Coefficient value", fontsize=14)
ax3.plot_date(df2_k[0], df2_k[1], fmt='.-')
ymin = 0
ymax = 0.12
ax3.vlines(datetime.date(2020, 5, 1), ymin, ymax, linestyles='dotted', label="Start May holidays", color='r')
ax3.vlines(datetime.date(2020, 5, 22), ymin, ymax, linestyles='dotted', label="End of self-isolation in Moscow region", color='g') 
ax3.vlines(datetime.date(2020, 6, 8), ymin, ymax, linestyles='dotted', label="End of self-isolation in Moscow", color='b')
ax3.vlines(datetime.date(2020, 7, 1), ymin, ymax, linestyles='dotted', label="Voiting")

plt.show()

