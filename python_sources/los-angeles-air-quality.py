#!/usr/bin/env python
# coding: utf-8

# This kernal contains one experiment using CO2 and TVOC data collected from an [Adafruit CCS811 gas sensor.](https://www.adafruit.com/product/3566?gclid=CjwKCAjw4NrpBRBsEiwAUcLcDC7rfEBlaclDQMmAmDsiB-NlT1wL61pWEKSJDLwR02b2QgCL3pEg2RoCNKAQAvD_BwE)
# New CO2 and TVOC readings are published every 20 minutes.  You can subscribe by choosing the LA Air Quality sensor from the I3 Data Marketplace.
# 
# CO2 Data
# CO2 data is measured in PPM or parts per million.  OSHA set the permissible exposure limits at 5000 PPM.  Read more about this in the [USA Food Safety and Inspection Service Carbon Dioxide Health Hazard Information Sheet](https://www.fsis.usda.gov/wps/wcm/connect/bf97edac-77be-4442-aea4-9d2615f376e0/Carbon-Dioxide.pdf?MOD=AJPERES)
# 
# TVOC Data
# TVOC stands for Total Volatile Organic Compounds.  This device measures TVOCs in PPB or parts per billion.  Other devices may measure TVOCs in ug/m3.  To learn more, search Google for "TVOC" and you will find many good resources.

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.display.max_columns=50


# In[ ]:


a = pd.read_csv("../input/airquality-July19.csv")

# clean data
a['MAX_CO2_PPM'] = pd.Series()
a['MAX_CO2_PPM'] = a['MAX_CO2_PPM'].fillna(5000.0)

# remove readings that are beyond the sensor's range of 400 to 8192 ppm for CO2 and 0 to 1187 for TVOCs
a = a.loc[a['CO2_PPM'] > 0]
a = a.loc[a['CO2_PPM'] < 8192]


# Take a look at the following chart.  Do you see anything strange, or perhaps a bit scary?

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(15,5))
lines = plt.plot(a['TIMESTAMP'], a['CO2_PPM'], a['MAX_CO2_PPM']) 
plt.setp(lines,color='r',linewidth=1.0, marker='')
plt.xlabel('Time')
plt.ylabel('Parts Per Million (PPM)')
plt.title("CO2_PPM - Indoor levels compared to Auto Exhaust")
plt.xticks(rotation='vertical', fontsize='7')
plt.show()


# Most of the time, CO2 levels are about 400.  On July 19, from 10:10 until 10:24, CO2 levels are very high!  At this time, I took this device outside, turned on my car, and put the sensor near the exhaust.  On July 23, from 10:54 until 11:21, CO2 levels spike again.  What do you think happened?

# In[ ]:


fig = plt.figure(figsize=(15,5))
lines = plt.plot(a['TIMESTAMP'], a['TVOC_PPB']) 
plt.setp(lines,color='k',linewidth=1.0, marker='')
plt.xlabel('Time')
plt.ylabel('Parts Per Billion (PPB)')
plt.title("TVOC_PPB (Total Volatile Organic Compound) - Indoor levels compared to Auto Exhaust")
plt.xticks(rotation='vertical', fontsize='7')
plt.show()


# Notice how the TVOCs spike during the car exhaust experiment.  Otherwise, TVOC levels are typically around zero.

# Special thanks to Venkatesh Prabhu, Mehdi, and WenChen Mei for technical support

# 
