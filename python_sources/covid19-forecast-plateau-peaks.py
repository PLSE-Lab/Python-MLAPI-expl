#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import math as math

from matplotlib.colors import LogNorm
import seaborn as sns

from bokeh.plotting import figure
from bokeh.models import FactorRange,ColumnDataSource
from bokeh.io import  show,output_notebook
from scipy.signal import find_peaks


# In[ ]:


confirmed_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
#rename column name
confirmed_df = confirmed_df.rename(columns={'Province/State':'State','Country/Region':'Country'})

death_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
#rename column name
death_df = death_df.rename(columns={'Province/State':'State','Country/Region':'Country'})

recovered_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
recovered_df = recovered_df.rename(columns={'Province/State':'State','Country/Region':'Country'})


# In this Notebook, I have analyzed the country-wise comparison of number of confirmed cases, number of recoveries and number of deaths till date. I used this data to detect the patterns, peaks and plateau. Since it originally started with China - we can utilize the patterns in the data from China to project the forecast for other countries. I have done the projection till May 2020.  Here are few findings from the analysis:
# 
#     1. For most of the countries, we can expect to see plateau around May 2020. 
#     2. Asia pacific countries tend to see less number of waves than North America and European countries 
#     3. Forecast analysis shows Italy will reach ~8 million and US will reach ~3.8 million confirmed cased by May 2020. 
#     4. US and UK have seen exactly same number of waves till date => Pattern is very much similar. 
#     5. One steep peak is expected at the beginning of April.
#     
#     
# Here is the graph from CDC - found it interesting :) 
# 
# ![curve](https://storage.googleapis.com/kagglesdsdata/datasets/566092/1027853/curve.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1585151850&Signature=ZHAgg3%2FPIBPFUGtphCvDwRSCmFV6t1Vmlw2%2FOVKuXecmO40426XolmsHiN501m2JMU8UNq21%2BfaWv2a0upqyp6UASzNpf3hZTtF7lqElf5b43DkkBArpl9pxnGWXAwHTVF3R6dDx%2BXF6aiy%2FrvPHogwp6U3t0kraOLH9ORFRf%2BXHQM8XQTbRuNFfB9VtrNz78RWB3kpCMdYv%2BDwqntnOaH9TN2Qav6kj5HDfOljSKVeX64FnoZDx2SFWX7sDgL7RL1k5IsFp6Cw1RWxOL01NosSVMAnnowK2hizBxrME8k%2BBMbin88jIBGIFXot8hvr5k8y9qL8%2B%2BlrxvSWIYAoJxg%3D%3D)
# 
# 

# **Helper Functions**

# In[ ]:


'''
Plot the graph of top k countries with top k values of Confirmed cases, deaths and recoveries

'''
def plot_growth_rate(df_data,k,text,plot):
    conf_sorted=df_data.loc[:,~df_data.columns.isin(['State','Lat','Long'])].groupby(
    by=['Country'],
    axis=0).sum().sort_values(
                              df_data.columns[-1],
                              ascending=False)

    x_dates=pd.date_range(start=df_data.columns[4:].values[0],periods=df_data.columns[4:].values.size)
    #create a dictionary where keys are first k highest affected countries 
    conf_dict = {conf_sorted.index.values[i]:conf_sorted.iloc[i,:].values for i in range(k)}
    conf_df_plt=pd.DataFrame(data=conf_dict,index=x_dates)

    
    if plot:
        conf_df_plt.plot(subplots=True,layout=(10, 2),sharey=False,figsize=(25,10),title=text);
    return conf_dict.keys(),conf_df_plt

'''
Plot the peaks in the percentage change value 
'''

def plot_phases(country,df_conf_all):
    p = figure(plot_width=800,
               plot_height=250,
               x_axis_type="datetime",
               tools='',
               title='Daily percentage Change - '+country,
              )

    p.line(df_conf_all.index, df_conf_all[country].pct_change(),color="green",legend_label=country)

    #peaks
    peaks_x = [df_conf_all.index[idx] for idx in find_peaks(x=df_conf_all[country].pct_change(),height=.3)[0]]
    peaks_y = [df_conf_all[country].pct_change()[idx] for idx in find_peaks(x=df_conf_all[country].pct_change(),height=.3)[0]]
    p.circle(peaks_x, peaks_y, size=20, color="red", alpha=0.5,legend_label='Peaks')


    output_notebook()
    show(p)
    return len(peaks_x)


# # Number of waves so far
# 
# __In this section I have found the number of peaks aka waves till date. This is a strong indicator of the frequency of small outbreaks which can be used to forecast if plateau can be expected in new future or not. I have picked up the k countries based on the most number of confirmed cases__

# In[ ]:


df_waves=pd.DataFrame(columns=['Country','NumWaves'])
i=-1
#build the dataframe of confirmed cases of all countries
_,df_conf_all = plot_growth_rate(confirmed_df,155,'Growth Chart - Confirmed cases',plot=False)


# # **Let's analyze the daily percentage increase/decrease in number of confirmed cases to understand the number of days it takes to get the growth percentage to converge to zero**

# In[ ]:


countries = ['China','Italy','Iran','Canada','US','Australia','United Kingdom','Singapore','India','Spain']

for country in countries:
    peaks=plot_phases(country,df_conf_all)
    i=i+1
    df_waves.loc[i]=[country,peaks]


# **Frequency of COVID-19 Waves till date**

# In[ ]:


counts=df_waves['NumWaves'].values

sorted_countries=sorted(countries,key=lambda x:counts[countries.index(x)])
p = figure(y_range=sorted_countries, 
           plot_width=650, 
           plot_height=300, 
           title="Frequency of COVID-19 Waves till date",
           toolbar_location=None, 
           tools=""
          )
p.hbar(y=countries, right=counts, height=0.7,fill_color='orange')

p.x_range.start = 0

show(p)


# # Growth Chart  - Confirmed cases

# In[ ]:


top_k_conf,top_k_conf_df=plot_growth_rate(confirmed_df,20,'Growth Chart - Confirmed cases',plot=True)


# # Growth Chart - Deaths

# In[ ]:


top_k_death,top_k_death_df=plot_growth_rate(death_df,20,'Growth Chart - Deaths',plot=True)


# # Growth Chart - Recovered Cases

# In[ ]:


top_k_recover,top_k_recover_df=plot_growth_rate(recovered_df,20,'Growth Chart - Recovered Cases',plot=True)


# **Since it originally started with China, we can use the China's initial historic data to fetch the pattern and can be used to forecast data in other countries**

# In[ ]:


date_rng = pd.date_range(start=df_conf_all.index[-1],periods=len(df_conf_all.index) , freq='D')
predicted_values=pd.DataFrame(index=date_rng,columns=['Count'])
idx=0
predicted_values.iloc[idx]=(1+df_conf_all['China'].pct_change()[1])*df_conf_all[country][-1]
country='US'

for pct_ch in df_conf_all['China'].pct_change()[2:]:
    idx=idx+1
    predicted_values.iloc[idx]=(1+pct_ch)*predicted_values.iloc[idx-1]

p = figure(plot_width=800,
               plot_height=250,
               x_axis_type="datetime",
               tools='',
               title='Forecast [Confirmed Cases] - '+country,
              )

p.line(predicted_values.index,predicted_values['Count'] ,color="red",legend=country)
output_notebook()
show(p)


# **Forecast of other countries, Please note: list of countries is not based on any order statistic - just countries of my interest :)**

# In[ ]:


countries = ['Italy','Iran','Canada','US','Australia','United Kingdom','Singapore','India','Spain']
predicted_count = {}

for country in countries:
    date_rng = pd.date_range(start=df_conf_all.index[-1],periods=len(df_conf_all.index) , freq='D')
    predicted_values=pd.DataFrame(index=date_rng,columns=['Count'])
    idx=0
    predicted_values.iloc[idx]=(1+df_conf_all['China'].pct_change()[1])*df_conf_all[country][-1]
    
    for pct_ch in df_conf_all['China'].pct_change()[2:]:
        idx=idx+1
        predicted_values.iloc[idx]=(1+pct_ch)*predicted_values.iloc[idx-1]
    
    predicted_count[country]=predicted_values['Count'][-2]


for k,v in predicted_count.items():
    print(k,' : ',int(v))


# In[ ]:


countries=list(predicted_count.keys())
counts=list(predicted_count.values())

sorted_countries=sorted(list(predicted_count.keys()),key=lambda x:counts[countries.index(x)])
p = figure(y_range=sorted_countries, 
           plot_width=650, 
           plot_height=300, 
           title="Forecasted COVID-19 Cases - May 2020",
           toolbar_location=None, 
           tools=""
          )
p.hbar(y=countries, right=counts, height=0.7,fill_color='orange')

p.x_range.start = 0

show(p)

