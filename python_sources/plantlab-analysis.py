#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import new packages
get_ipython().system('pip install psypy')

#load the github repository into workspace
get_ipython().system('cp -r ../input/plantlab/* ./')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import psypy.psySI as si
import os, sys

import data_io


# In[ ]:


#import the sensor readings table
df = pd.read_sql_table('sensor_readings',data_io.SQL_DBNAME)
df.set_index('datetime',inplace=True)
df = df.loc['2019'].copy()
df[df.sensorid=='TA01'].plot()


# Since startup in May, it has been difficult to get the sensors to read consistently, so consequently the data series is very choppy.  Only recently has the data been reliable and consistent.  Lets zoom into recent data collected in a stable period from 18th and 19th.  This time series is the temperature experienced by the plants with the HVAC system ON continuously at full power and the lights on 12h ON/12h OFF cycles.  Not enough data was collected to clearly see the effect of the lights on the temperature because the sensors failed in shorter than the 12h cycle.

# In[ ]:


window = df.loc['2019-07-18':'2019-07-20'].copy()
window[window.sensorid=='TA01'].plot()


# There are a number of ways of analyzing the data.  Total are 4x sensors, two sensors (TA04,TA05) are outside the chamber and two are inside the chamber (TA01-return, TA02-supply).  The sensor that represents the conditions experienced by the plants is TA01.  Each sensor has a humidity DHX and a temperature TAX.  Each sensor reading is taken at slightly different timestamps, so for convenience it would be easier to group the readings by 5m or 10m intervals so that their timestamps match up as close as possible.  

# In[ ]:


##convert to nearest 10 minute dataset
int_min = 10
window.reset_index(inplace=True)
window['dt_short'] = window.apply(lambda x:dt.datetime(
    year=x.datetime.year,month=x.datetime.month,day=x.datetime.day,
    hour=x.datetime.hour,minute=int(x.datetime.minute/int_min)*int_min),axis=1)
del window['datetime']
window.rename(columns={'dt_short':'datetime'},inplace=True)


# In[ ]:


#create pivot table
# datetime, value 
tsdata = pd.pivot_table(window,values='value',columns='sensorid',index='datetime',aggfunc='mean')
tsdata.head()


# Two parameters of interest are the specific enthalpy and the absolute humidity. To add these we will use psypy package si.state() function

# In[ ]:


tsdata['ENTH01'] = tsdata.apply(lambda x: si.state("DBT", x.TA01+273.15, "RH", x.HA01/100, 101325)[1],axis=1)
tsdata['MOIST01'] = tsdata.apply(lambda x: si.state("DBT", x.TA01+273.15, "RH", x.HA01/100, 101325)[4],axis=1)
tsdata[['TA01','HA01','ENTH01','MOIST01']].head()


# In[ ]:


tsdata['ENTH01'].plot()


# In[ ]:


tsdata['MOIST01'].plot()


# The trend of enthalpy and moisture track in a similar pattern to the temperature trend. Finally, repeat the analysis by zooming into second period from 8:00-20:00 on 19-Jul.  The spikes are suspected to be periods when the HVAC temporarily shutoff, which is a problem that has been observed due to water dripping onto the blower motor and has since been resolved.

# In[ ]:


tsdata.loc['2019-07-19 08':'2019-07-19 20']['ENTH01'].plot()


# In[ ]:


tsdata.loc['2019-07-19 08':'2019-07-19 20']['MOIST01'].plot()


# In[ ]:


tsdata.loc['2019-07-19 08':'2019-07-19 20']['TA01'].plot()


# In[ ]:


tsdata.loc['2019-07-19 08':'2019-07-19 20']['HA01'].plot()

