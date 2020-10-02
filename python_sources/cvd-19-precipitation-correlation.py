#!/usr/bin/env python
# coding: utf-8

# This notebook attempts to model the California outbreak from Day 0.
# 
# * Missing Infection Data
# * Taylor Series
# * NOAA Weather Data (San Francisco, CA) 
# * Precipitation Correlation Heatmap

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# CONFIG
mpl.rcParams['figure.figsize'] = 8, 6
pd.options.mode.chained_assignment = None  # default='warn'
sns.set(style='whitegrid', palette='muted', font_scale=1.5)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# LOAD DATA\ntrain = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')\n# SCRUB DATA\njunk =['Id','Country/Region','Lat','Long','Province/State']\ntrain.drop(junk, axis=1, inplace=True)\n# LOAD MISSING CASE DATA\norg = pd.read_csv('/kaggle/input/gthubdata-new/time_series_19-covid-Confirmed.csv')\nus = org[org['Country/Region']=='US']\ndays = us.columns[4:]\nca_state = us[us['Province/State']=='California']\nca_counties = us[us['Province/State'].str.find('CA')>0]\nmissing=(ca_counties[days].sum() + ca_state[days])[days[4:48]]\n# LOAD MISSING DEATH DATA\nxx = pd.read_csv('/kaggle/input/gthubdata-new/time_series_19-covid-Deaths.csv')\nus_death = xx[xx['Country/Region']=='US']\ndays = us_death.columns[4:]\nca_state_death = us_death[us_death['Province/State']=='California']\nca_counties_death = us_death[us_death['Province/State'].str.find('CA')>0]\nmissing_death=(ca_counties_death[days].sum() + ca_state_death[days])[days[40:48]]\n# LOAD WEATHER DATA\nweather = pd.read_csv('/kaggle/input/noaa-california-weather/2087824.csv')\n# SCRUB WEATHER DATA\nempty=['STATION','SNOW','SNWD']\nweather.drop(empty, axis=1, inplace=True)")


# In[ ]:


# ADD MISSING CASE DATA
train.ConfirmedCases.loc[4]=int(missing['1/26/20'])
train.ConfirmedCases.loc[5]=int(missing['1/27/20'])
train.ConfirmedCases.loc[6]=int(missing['1/28/20'])
train.ConfirmedCases.loc[7]=int(missing['1/29/20'])
train.ConfirmedCases.loc[8]=int(missing['1/30/20'])
train.ConfirmedCases.loc[9]=int(missing['1/31/20'])
train.ConfirmedCases.loc[10]=int(missing['2/1/20'])
train.ConfirmedCases.loc[11]=int(missing['2/2/20'])
train.ConfirmedCases.loc[12]=int(missing['2/3/20'])
train.ConfirmedCases.loc[13]=int(missing['2/4/20'])
train.ConfirmedCases.loc[14]=int(missing['2/5/20'])
train.ConfirmedCases.loc[15]=int(missing['2/6/20'])
train.ConfirmedCases.loc[16]=int(missing['2/7/20'])
train.ConfirmedCases.loc[17]=int(missing['2/8/20'])
train.ConfirmedCases.loc[18]=int(missing['2/9/20'])
train.ConfirmedCases.loc[19]=int(missing['2/10/20'])
train.ConfirmedCases.loc[20]=int(missing['2/11/20'])
train.ConfirmedCases.loc[21]=int(missing['2/12/20'])
train.ConfirmedCases.loc[22]=int(missing['2/13/20'])
train.ConfirmedCases.loc[23]=int(missing['2/14/20'])
train.ConfirmedCases.loc[24]=int(missing['2/15/20'])
train.ConfirmedCases.loc[25]=int(missing['2/16/20'])
train.ConfirmedCases.loc[26]=int(missing['2/17/20'])
train.ConfirmedCases.loc[27]=int(missing['2/18/20'])
train.ConfirmedCases.loc[28]=int(missing['2/19/20'])
train.ConfirmedCases.loc[29]=int(missing['2/20/20'])
train.ConfirmedCases.loc[30]=int(missing['2/21/20'])
train.ConfirmedCases.loc[31]=int(missing['2/22/20'])
train.ConfirmedCases.loc[32]=int(missing['2/23/20'])
train.ConfirmedCases.loc[33]=int(missing['2/24/20'])
train.ConfirmedCases.loc[34]=int(missing['2/25/20'])
train.ConfirmedCases.loc[35]=int(missing['2/26/20'])
train.ConfirmedCases.loc[36]=int(missing['2/27/20'])
train.ConfirmedCases.loc[37]=int(missing['2/28/20'])
train.ConfirmedCases.loc[38]=int(missing['2/29/20'])
train.ConfirmedCases.loc[39]=int(missing['3/1/20'])
train.ConfirmedCases.loc[40]=int(missing['3/2/20'])
train.ConfirmedCases.loc[41]=int(missing['3/3/20'])
train.ConfirmedCases.loc[42]=int(missing['3/4/20'])
train.ConfirmedCases.loc[43]=int(missing['3/5/20'])
train.ConfirmedCases.loc[44]=int(missing['3/6/20'])
train.ConfirmedCases.loc[45]=int(missing['3/7/20'])
train.ConfirmedCases.loc[46]=int(missing['3/8/20'])
train.ConfirmedCases.loc[47]=int(missing['3/9/20'])
# ADD MISSING DEATH DATA
train.Fatalities.loc[42]=int(missing_death['3/4/20'])
train.Fatalities.loc[43]=int(missing_death['3/5/20'])
train.Fatalities.loc[44]=int(missing_death['3/6/20'])
train.Fatalities.loc[45]=int(missing_death['3/7/20'])
train.Fatalities.loc[46]=int(missing_death['3/8/20'])
train.Fatalities.loc[47]=int(missing_death['3/9/20'])


# In[ ]:


train.plot(subplots=False)


# In[ ]:


# CALCULATE EXPANSION TABLE
diff_conf, conf_old = [], 0 
diff_fat, fat_old = [], 0
dd_conf, dc_old = [], 0
dd_fat, df_old = [], 0

for row in train.values:
    diff_conf.append(row[1]-conf_old)
    conf_old=row[1]
    diff_fat.append(row[2]-fat_old)
    fat_old=row[2]
    dd_conf.append(diff_conf[-1]-dc_old)
    dc_old=diff_conf[-1]
    dd_fat.append(diff_fat[-1]-df_old)
    df_old=diff_fat[-1]
    
print(len(diff_conf),train.shape)


# In[ ]:


# POPULATE DATAFRAME FEATURES
train['CasePerDay'] = diff_conf
train['DeathPerDay'] = diff_fat
train['CaseDifferential'] = dd_conf
train['DeathDifferential'] = dd_fat
# PREP TRAIN DATA 
#train = train[:-1] 


# In[ ]:


train.plot(subplots=True)


# In[ ]:


weather.plot(subplots=True)


# In[ ]:


# ADD WEATHER FEATURES
train['Rain'] = weather['PRCP']
train['TempMin'] = weather['TMIN']
train['TempMax'] = weather['TMAX']

# MISSING WEATHER DATA (3/20-24)
train.TempMax[58] = 64
train.TempMin[58] = 48
train.Rain[58] = 0.0
train.TempMax[59] = 63
train.TempMin[59] = 49
train.Rain[59] = 0.0
train.TempMax[60] = 67
train.TempMin[60] = 51
train.Rain[60] = 0.01
train.TempMax[61] = 55
train.TempMin[61] = 50
train.Rain[61] = 0.0
train.TempMax[62] = 56
train.TempMin[62] = 48
train.Rain[62] = 0.0
train[60:]


# In[ ]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corrmat, vmax=.8, square=True);

