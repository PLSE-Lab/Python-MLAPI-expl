#!/usr/bin/env python
# coding: utf-8

# ## The below is only basic data analysis. I would request folks to do more analysis and visualization (trends,maps etc) to create meaningful insights as it will be useful in current situation.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install chart_studio')
get_ipython().system('pip install plotly-geo')


# In[ ]:


#Import plotly libraries
import chart_studio.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
plotly.offline.init_notebook_mode(connected=True)


# In[ ]:


majordir='/kaggle/input/covid19-us-county-trend/'
datadir=majordir+'csse_covid_19_daily_reports/'
date_today=27


# In[ ]:


covid_data_world_daily_0322=pd.read_csv(datadir+'03-22-2020.csv')
covid_data_world_daily_0322.rename(columns={'Deaths':'Deaths_0322'},inplace=True)
covid_data_world_daily_0322.head()
covid_data_us_daily_0322=covid_data_world_daily_0322[covid_data_world_daily_0322['Country_Region']=='US'].copy()
covid_data_us_daily_0322.shape


# In[ ]:



for i in range(23,date_today):
    dataset=datadir+'03-'+str(i)+'-2020.csv'
    col='Deaths_03'+str(i)
    covid_data_world_daily=pd.read_csv(dataset)
    covid_data_world_daily.rename(columns={'Deaths':col},inplace=True)
    covid_data_us_daily=covid_data_world_daily[covid_data_world_daily['Country_Region']=='US'].copy()

    if i==23:        
        covid_data_us_dailytrend=covid_data_us_daily_0322[['FIPS','Deaths_0322']].merge(covid_data_us_daily[['FIPS',col]],on='FIPS').dropna()
    else:
        print(i)
        covid_data_us_dailytrend=covid_data_us_dailytrend.merge(covid_data_us_daily[['FIPS',col]],on='FIPS').dropna()
covid_data_us_dailytrend.shape


# In[ ]:


covid_data_us_daily_0322.head()


# In[ ]:


covid_data_us_dailytrend.dropna().shape
covid_data_us_dailytrend.head()


# In[ ]:


census_df_fips = pd.read_excel(majordir+'PopulationEstimates_us_county_level_2018.xlsx',skiprows=1)
census_df_fips.head()


# In[ ]:


census_df_fips.FIPS=census_df_fips.FIPS.astype(float)


# In[ ]:


census_df_fips_covid=census_df_fips.merge(covid_data_us_dailytrend,on='FIPS') 


# In[ ]:


census_df_fips_covid.head()


# ## Death relative to the size of the population
# ### Death in each county normalized to its population count and presented as per 10000 people unit

# In[ ]:


for i in range(22,date_today):
    col='Deaths_03'+str(i)
    col_10000='Deaths_per10000_03'+str(i)
    census_df_fips_covid[col_10000]=10000*(census_df_fips_covid[col]/census_df_fips_covid['POP_ESTIMATE_2018'])


# In[ ]:


census_df_fips_covid.head()


# ##### Lets plot and observe counties that has atleast 50000 population and over 5 Deaths cases.

# In[ ]:


census_df_fips_covid_50000=census_df_fips_covid[(census_df_fips_covid['POP_ESTIMATE_2018']>50000) & (census_df_fips_covid['Deaths_0324']>5)]
plt.plot(census_df_fips_covid_50000['POP_ESTIMATE_2018'],census_df_fips_covid_50000[['Deaths_per10000_0322']],'*')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('POP_ESTIMATE_2018')
plt.ylabel('Deaths_per10000_0322')


# In[ ]:


col='Deaths_per10000_03'+str(date_today-1)
col0='Deaths_03'+str(date_today-1)
census_df_fips_covid_50000=census_df_fips_covid[(census_df_fips_covid['POP_ESTIMATE_2018']>50000) & (census_df_fips_covid[col0]>5)]
plt.plot(census_df_fips_covid_50000['POP_ESTIMATE_2018'],census_df_fips_covid_50000[[col]],'*')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('POP_ESTIMATE_2018')
plt.ylabel(col)


# ### I am guessing the negative trend in general with population indicates more spreading as of now stems from an individual source (who caught virus through travel and might be attending a large gathering like party,funeral,church event in his/her locality) rather community spread.

# In[ ]:


# census_df_fips_covid_100000.sort_values(by='Deaths_per10000_0324',ascending=False).reset_index(drop=True).head(50)


# In[ ]:


col='Deaths_per10000_03'+str(date_today-1)
census_df_fips_covid_50000.sort_values(by=col,ascending=False).reset_index(drop=True)


# In[ ]:


covid_data_us_dailytrend_nonan=covid_data_us_dailytrend[covid_data_us_dailytrend['FIPS'].notna()].copy()
covid_data_us_dailytrend_nonan.FIPS=covid_data_us_dailytrend_nonan.FIPS.astype(str).str.split('.',expand=True)[0]


# In[ ]:


census_df_fips_covid.head()


# In[ ]:


fips = census_df_fips_covid.FIPS.tolist()
values =census_df_fips_covid.Deaths_per10000_0322.tolist()

fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()


# In[ ]:


col='Deaths_per10000_03'+str(date_today-1)
fips = census_df_fips_covid.FIPS.tolist()
values =census_df_fips_covid[col].tolist()

fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()

