#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import os
import gc
import json
from scipy.optimize import curve_fit
import datetime
from pathlib import Path


# In[ ]:


pd.options.display.float_format = '{:,.3f}'.format
pd.set_option('display.max_columns', 50)


# In[ ]:


df = pd.read_csv('https://query.data.world/s/keax53lpqwffhayvcjmowjiydtevwo', parse_dates=['REPORT_DATE']).copy()


# In[ ]:


df.head()


# In[ ]:


print("date range: {0} to {1}".format(df['REPORT_DATE'].min(), df['REPORT_DATE'].max()))


# # US Trends

# In[ ]:


df_us = df[df['COUNTRY_ALPHA_2_CODE'] == 'US']


# In[ ]:


df_us['PROVINCE_STATE_NAME'].unique()


# In[ ]:


df_usp = df_us.groupby(['REPORT_DATE','PROVINCE_STATE_NAME']).sum()[
    [
        'PEOPLE_POSITIVE_CASES_COUNT', 
        'PEOPLE_POSITIVE_NEW_CASES_COUNT', 
        'PEOPLE_DEATH_COUNT', 
        'PEOPLE_DEATH_NEW_COUNT'
    ]
].unstack().copy().drop('District of Columbia', level=1, axis=1)


# In[ ]:


top_10 = df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[-1].sort_values(ascending=False)[0:10].index.values
top_25 = df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[-1].sort_values(ascending=False)[0:25].index.values


# In[ ]:


print("Total deaths to date:\n{0}".format(df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[-1][top_25]))


# In[ ]:


df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[30::][top_10].plot.line(
    figsize=(12,9),
    title="Top 10 US States with the most commulative COVID-19 fatalities"
);


# The graphs below show the growth trends by state.  Note that the left axis is not standardized between states so it's important to look at the magnitude.

# In[ ]:


df_usp.xs('PEOPLE_POSITIVE_CASES_COUNT', axis=1, level=0).iloc[30::].rolling(window=5).mean().diff().rolling(3).mean().plot(
    subplots=True, 
#     ylim=(-10,25), 
    layout=(10,5), 
    figsize=(18,24),
    grid=True, 
    title='New confirmed COVID-19 cases (US / daily rolling average)',
);


# In[ ]:


df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[30::].rolling(window=5).mean().diff().rolling(3).mean().plot(
    subplots=True, 
#     ylim=(-10,25), 
    layout=(10,5), 
    figsize=(18,24),
    grid=True, 
    title='New COVID-19 fatalities (US / daily rolling average)',
);


# # Global Trends

# In[ ]:


df_cp = df.groupby(['REPORT_DATE','COUNTRY_SHORT_NAME']).sum()[
    [
        'PEOPLE_POSITIVE_CASES_COUNT', 
        'PEOPLE_POSITIVE_NEW_CASES_COUNT', 
        'PEOPLE_DEATH_COUNT', 
        'PEOPLE_DEATH_NEW_COUNT'
    ]
].unstack().copy()


# In[ ]:


top_10c = df_cp.xs('PEOPLE_POSITIVE_CASES_COUNT', axis=1, level=0).iloc[-5:-1].max().sort_values(ascending=False)[0:10].index.values
top_25c = df_cp.xs('PEOPLE_POSITIVE_CASES_COUNT', axis=1, level=0).iloc[-5:-1].max().sort_values(ascending=False)[0:25].index.values


# In[ ]:


df_cp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0)[top_10c].plot.line(
    figsize=(12,9),
    title="Top 10 Countries with the most commulative COVID-19 fatalities"
);


# The graphs below show the growth trends by country. Note that the left axis is not standardized between states so it's important to look at the magnitude.  Also note that this depends on accurate reporting by the countried themselves, which is questionable in some cases.

# In[ ]:


df_cp.xs('PEOPLE_POSITIVE_CASES_COUNT', axis=1, level=0).iloc[30::][top_25c].rolling(window=5).mean().diff().rolling(3).mean().plot(
    subplots=True, 
#     ylim=(-25,100), 
    grid=True, 
    layout=(5,5), 
    figsize=(18,12), 
#     cmap='tab20',
    title='New confirmed COVID-19 cases (global / daily rolling average)'
);


# In[ ]:


df_cp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[30::][top_25c].rolling(window=5).mean().diff().rolling(3).mean().plot(
    subplots=True, 
#     ylim=(-25,100), 
    grid=True, 
    layout=(5,5), 
    figsize=(18,12), 
    cmap='tab20',
    title='New COVID-19 fatalities (global / daily rolling average)'
);


# # Population Normed Comparisons

# In[ ]:


df_statepop = pd.read_csv('../input/nst-est2019-alldata.csv').iloc[5::]
df_countrypop = pd.read_csv('../input/world_pop_2020.csv')
df_usppop = df_usp.iloc[-1].swaplevel(0,1).unstack().merge(df_statepop[['NAME','POPESTIMATE2019']], left_index=True, right_on='NAME').set_index('NAME')


# In[ ]:


df_cppop = df_cp.iloc[-5:-1].max().swaplevel(0,1).unstack().merge(df_countrypop[['country_code','population', 'country']], left_index=True, right_on='country').set_index('country')
df_cppop_lg = df_cppop[df_cppop['population'] > 10000000]


# In[ ]:


ax = df_cppop_lg[df_cppop_lg.columns[0:4]].div(df_cppop_lg['population'], axis=0)[['PEOPLE_POSITIVE_CASES_COUNT','PEOPLE_DEATH_COUNT']].sort_values(ascending=False, by='PEOPLE_POSITIVE_CASES_COUNT')[0:50].plot.bar(
    figsize=(20,8), 
    title="% of population infected with or killed by COVID-19 (by country >10M pop)",
#     stacked=True,
#     logy=True
#     icons='child', 
#     icon_size=18, 
#     icon_legend=True,
);
vals = ax.get_yticks();
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals]);


# In[ ]:


ax = df_usppop[df_usppop.columns[0:4]].div(df_usppop['POPESTIMATE2019'], axis=0)[['PEOPLE_POSITIVE_CASES_COUNT','PEOPLE_DEATH_COUNT']].sort_values(ascending=False, by='PEOPLE_POSITIVE_CASES_COUNT')[0:50].plot.bar(
    figsize=(20,8), 
    title="% of population infected with or killed by COVID-19 (by US state)",
#     stacked=True,
#     logy=True
#     icons='child', 
#     icon_size=18, 
#     icon_legend=True,
);
vals = ax.get_yticks();
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals]);


# The chart below shows the "mortality" of COVID-19 by region.  The very wide range of mortality could be due to very different potency of the virus in different regions (unlikely), differences in testing and reporting accuracy (more likely) or lag (since fatalities happen days or weeks after infection).  Germany seems to have the most robust testing and reporting system at the moment and their mortality rate hovers between 4-5%. 

# In[ ]:


ax = (df_cppop_lg['PEOPLE_DEATH_COUNT']/df_cppop_lg['PEOPLE_POSITIVE_CASES_COUNT']).sort_values(ascending=False)[0:50].plot.bar(
    figsize=(15,8), 
    title="Mortality (fatalities per infections / by country >10M pop)"
);
vals = ax.get_yticks();
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals]);


# In[ ]:


ax = (df_usppop['PEOPLE_DEATH_COUNT']/df_usppop['PEOPLE_POSITIVE_CASES_COUNT']).sort_values(ascending=False).plot.bar(
    figsize=(15,8), 
    title="Mortality (fatalities per infections / by US state)"
);
vals = ax.get_yticks();
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals]);


# # infected population

# In[ ]:


df_per = df_usp.iloc[-14::].sum().swaplevel(0,1).unstack().merge(df_statepop[['NAME','POPESTIMATE2019']], left_index=True, right_on='NAME').set_index('NAME')


# In[ ]:


(df_per['PEOPLE_POSITIVE_NEW_CASES_COUNT']/df_per['POPESTIMATE2019']*10000).sort_values(ascending=False).plot.bar(
    figsize=(15,8), 
    title="Active infections per 10,000 people (based on 14 day infection period)"
);


# In[ ]:


df_rinf = df_usp.xs('PEOPLE_POSITIVE_NEW_CASES_COUNT', axis=1, level=0).rolling(window=14).sum().iloc[-90::].T.merge(
    df_statepop[['NAME','POPESTIMATE2019']], left_index=True, right_on='NAME').set_index('NAME')
(df_rinf[df_rinf.columns[0:-2]].div(df_rinf[df_rinf.columns[-1]], axis=0)*10000).T.plot(
    subplots=True, 
#     ylim=(0.01,50),
#     logy=True,
    grid=True, 
    layout=(12,5), 
    figsize=(18,24), 
#     cmap='tab20',
    title='Active infections per 10,000 people (based on 14 infection period)'
);


# In[ ]:


# df_trc = df.groupby(['Date','Country_Region','Case_Type']).agg({'Cases':sum,'Population_Count':sum})

df_rinfc = df_cp.xs('PEOPLE_POSITIVE_NEW_CASES_COUNT', axis=1, level=0)[top_25c].rolling(window=14).sum().iloc[-90::].T.merge(
    df_countrypop[['country_code','population', 'country']], left_index=True, right_on='country').set_index('country')

(df_rinfc[df_rinfc.columns[0:-2]].div(df_rinfc[df_rinfc.columns[-1]], axis=0)*10000).T.plot(
    subplots=True, 
#     ylim=(0.01,50),
#     logy=True,
    grid=True, 
    layout=(5,5), 
    figsize=(18,12), 
    cmap='tab20',
    title='Active infections per 10,000 people (based on 14 infection period)'
);


# In[ ]:




