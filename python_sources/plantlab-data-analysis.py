#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install psypy')

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import psypy.psySI as si


# In[ ]:


CSV_TBL = "../input/plantlab-data/sensor_readings.csv"
df = pd.read_csv(CSV_TBL,parse_dates=['datetime'],dayfirst=True)


# In[ ]:


df.tail()


# In[ ]:


df.set_index('datetime',inplace=True)
df = df.loc['2019'].copy()

window = df.loc['2019-08-01':].copy()

int_min = 10
window.reset_index(inplace=True)
window['dt_short'] = window.apply(lambda x:dt.datetime(
    year=x.datetime.year,month=x.datetime.month,day=x.datetime.day,
    hour=x.datetime.hour,minute=int(x.datetime.minute/int_min)*int_min),axis=1)
del window['datetime']
window.rename(columns={'dt_short':'datetime'},inplace=True)

#create pivot table
# datetime, value 
tsdata = pd.pivot_table(window,values='value',columns='sensorid',index='datetime',aggfunc='mean')
tsdata['ENTH01'] = tsdata.apply(lambda x: si.state("DBT", x.TA01+273.15, "RH", x.HA01/100, 101325)[1],axis=1)
tsdata['ENTH04'] = tsdata.apply(lambda x: si.state("DBT", x.TA04+273.15, "RH", x.HA04/100, 101325)[1],axis=1)
tsdata['MOIST01'] = tsdata.apply(lambda x: si.state("DBT", x.TA01+273.15, "RH", x.HA01/100, 101325)[4],axis=1)
tsdata['deltaE'] = tsdata['ENTH04'] - tsdata['ENTH01']


# In[ ]:


import seaborn as sb
import matplotlib.pyplot as plt
sb.set(rc={'figure.figsize':(16, 8)})


# During the period 09-16 Aug 2019 the photoperiod was using 50%, 50%, 50% of Blue, White and Red (12hr on, 12hr off) and not taking any feedback from the CO2 sensors or the LED %.  The reason for the small humps is because the system performs a manual reset every 30 minutes as a workaround to a NAN problem with the DHT sensors.

# In[ ]:


window = tsdata.loc['2019-08-09':'2019-08-16']


# In[ ]:


def plot_parameters(window,parameters,x_axis=False):
    N = len(parameters)
    f, axes = plt.subplots(N,1,figsize=(16,6*N))
    i = 0
    for p in parameters:
        window[p].plot(linewidth=0.5,ax=axes.flat[i]).set_title(parameters[p])
        axes.flat[i].get_xaxis().set_visible(x_axis)
        i = i + 1
    plt.tight_layout()


# In[ ]:


parameters = {'TA01':'Temp C',
              'HA01':'rel hum %',
              'ENTH01':'Enthalpy kJ/kg',
              'deltaE':'HVAC work kJ/kg',
             }
plot_parameters(window,parameters)


# On the 17th, code was uploaded to Arduino for the lights to be on timer setting to cycle 24hr on-off with a peak mid-day all 3 channels 75% and a simulated "evening" setting with RED only for 2hrs.  When analyzing the data however on 22nd, it appears that although the settings were updated, the LED lights were not switched on as expected.  The system was reset and again the problem was identified. On investigation an error was found and the Arduino code was updated.  After 24hrs the system was verified to be working again as expected.

# In[ ]:


window = tsdata.loc['2019-08-17':'2019-08-25']


# In[ ]:


parameters = {'LED01':'LED Blue %',
              'LED02':'LED White %',
              'LED03':'LED Red %',
              'TA01':'Temp C',
              'HA01':'rel hum %',
              'ENTH01':'Enthalpy kJ/kg',
              'deltaE':'HVAC work kJ/kg',
             }
plot_parameters(window,parameters)


# The new carbon dioxide sensor readings also show strange behavior that is difficult to interpret. They now are reading closer to the expected value, but have a wide range of faulty readings that must be filtered out to be able to interpret meaningfully.

# In[ ]:


window['CD01'].plot(linewidth=0.5);
window['CD02'].plot(linewidth=0.5);


# data from the past 7 days

# In[ ]:


lastweek = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(days=9),'%Y-%m-%d')
window = tsdata.loc[lastweek:]


# In[ ]:


parameters = {'LED01':'LED Blue %',
              'LED02':'LED White %',
              'LED03':'LED Red %',
              'TA01':'Temp C',
              'HA01':'rel hum %',
              'deltaE':'HVAC work kJ/kg',
             }
plot_parameters(window,parameters)


# In[ ]:


last48hrs = dt.datetime.strftime(dt.datetime.today()-dt.timedelta(hours=48),'%Y-%m-%d')
window = tsdata.loc[last48hrs:]


# In[ ]:


parameters = {'LED01':'LED Blue %',
              'LED02':'LED White %',
              'LED03':'LED Red %',
              'TA01':'Temp C',
              'HA01':'rel hum %',
              'deltaE':'HVAC work kJ/kg',
             }
plot_parameters(window,parameters,x_axis=True)


# The data shows that the controls are working.  The additional increase to 75% of light intensity shows as a temperature increase of +0.4-0.6 C and drops the HVAC work to 10 kJ/kg.  The 50% Red light for 2hrs before evening is not easily observed in the temperature or the enthalpy series.  At night the HVAC is able to pull 16 kJ/kg of work with the lights off.  The temperature is not constant and drifts with the conditions in the workshop, so a further improvement is to have feedback control instead of 100% ON HVAC.  The range of temperatures is small - 27-29C day and 26-27C night.
