#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# import numpy as np # linear algebra
import pandas as pd # data processing, XLSX file I/O for pd.read_excel()
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
# The following line allows us to display plots without typing the plt.show() command for each plot.
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime, time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_excel('../input/anom5j.xls')
df.sample(5)
# this spreadsheet is sorted by satellite


# We begin by removing the EDATE (date the data was entered) column since these dates are not relevant to data analysis.

# In[ ]:


df.drop(columns='EDATE', inplace=True)


# We are interested in the anomaly time in terms of both UTC and Local (STIMEU and STIMEL). The UTC time is important for possible correlation with space weather events, and the Local time tends to indicate where the satellite is located with respect to the earth-sun line. This is relevant since the space environment varies with respect to local solar time and satellite altitude above the earth's surface. (Notice the inclusion of SVE, or Sun-Vehicle-Earth angle in degrees.)
# 
# For analysis, we are going to combine the ADATE and STIMEU and assess STIMEL separately. We may add error bars in plots for the STIMEQ, LATQ, and LONGQ since those are uncertainty data.
# 
# The STIMEU and STIMEL have unusable values (9999) . We will address these invalid entries when we clean the data.

# In[ ]:


print("There are {} uinque satellites listed in this spreadsheet.".format(df['BIRD'].nunique()))
# df['BIRD'].unique()


# **Anomaly Type,  what was the anomalous behavior?**
# * PC   : Phantom Command - uncommanded status change. 
# * PF   : Part Failure.
# * TE   : Telemetry Error.
# * SE   : Soft Error, recoverable bit flip type error.
# * HE   : Hard Error, permanent chip damage or Latch-Up.
# * SS   : System Shutdown.
# * ESDM : ElectroStatic Discharge Measured (SCATHA specific).
# * ATT  : Attitude Control Problem.
# * UNK  : Unknown.

# In[ ]:


atype_count = df.groupby('ATYPE').VER.count().reset_index() # number of each type of anomaly
# print(atype_count)
adiag_count = df.groupby('ADIAG').VER.count().reset_index() # number of each type of diagnosis
# print(adiag_count)
atype_adiag = df.groupby(['ATYPE','ADIAG']).VER.count().reset_index() # number of each type of diagnosed cause for each type of anomaly
# atype_adiag 


# In[ ]:


plt.figure(figsize=(12,8))
plt.bar(atype_count['ATYPE'], atype_count['VER'])
plt.title('Spacecraft Anomalies by Type')
plt.xlabel('Type of Anomaly')
plt.ylabel('Count'); # the semicolon prevents the code from displaying "Text(0,0.5,'Count')", an artifact of not including the plt.show()


# **Anomaly diagnosis, what type of event caused the anomaly?**
# * ECEMP : Electron Caused ElectroMagnetic Pulse...Internal, Deep dielectric charging.
# * ESD : ElectroStatic Discharge...Surface charging.
# * SEU : Single Event Upset, Solar or Galactic Cosmic Rays.
# * MCP : Mission Control Problem, human induced or software error.
# * RFI : Radio Frequency Interference.
# * UNK : Unknown diagnosis.

# In[ ]:


plt.figure(figsize=(12,8))
plt.bar(adiag_count['ADIAG'], adiag_count['VER'])
plt.title('Spacecraft Anomalies by Cause')
plt.xlabel('Diagnosed Cause')
plt.ylabel('Count');


# **Correlate type of anomaly with the diagnosed cause.**
# 
# Notice the large numbers of unknowns for both.

# In[ ]:


anom_td_pivot = atype_adiag.pivot(index="ADIAG", columns="ATYPE", values="VER").reset_index()
# anom_td_pivot


# In[ ]:


ax = anom_td_pivot.set_index('ADIAG').reindex(anom_td_pivot.set_index('ADIAG').sum().sort_values().index, axis=1).T.plot(kind='bar', stacked=True,
        colormap=ListedColormap(sns.color_palette("Set3")),
        figsize=(12,8),
        title='Correlation between Type of Anomaly and Diagnosed Cause',
       )
ax.set_xlabel('Type of Anomaly');


# In[ ]:


a_td = atype_adiag.pivot(index="ADIAG", columns="ATYPE", values="VER") # td = type_diagnosis
plt.figure(figsize=(12,8))
ax = sns.heatmap(a_td, annot=True, fmt="g", cmap='Set3')
ax.set_xlabel('Type of Anomaly')
ax.set_ylabel('Diagnosed Cause')
plt.title('Heatmap of Anomaly Type versus Diagnosed Cause');


# In[ ]:


df.ADATE.sample()


# In[ ]:


plt.figure(figsize=(12,8))
plt.hist(df.ADATE, bins=32)
plt.title('Spacecraft Anomalies by Year')
plt.xlabel('Year')
plt.ylabel('Count');


# In[ ]:


anom_w_dtg = df[df.STIMEU != 9999].reset_index(drop=True) # drop these rows since we can
anom_w_dtg.tail() # looking at the tail since the original file had a lot of 9999 values at the end
anoms_loc_tm = anom_w_dtg[~anom_w_dtg.STIMEL.isna()] # eliminate entries with NaN values and those that are empty


# Now we use a partially cleaned dataset to plot anomalies by local time. Notice how the majority of those reported are from GEO satellites, so they dominate.

# In[ ]:


plt.figure(figsize=(12,8))
plt.hist(anoms_loc_tm.STIMEL, bins=24)
plt.xticks([hour*100 for hour in range(24)],
           ['%2d' %hr for hr in range(24)])
plt.title('Spacecraft Anomalies by Local Time')
plt.xlabel('Hour')
plt.ylabel('Count');


# In[ ]:


geo_anoms = anoms_loc_tm[anoms_loc_tm.ORBIT == 'G'].reset_index(drop=True)
inclined_anoms = anoms_loc_tm[anoms_loc_tm.ORBIT == 'I'].reset_index(drop=True)
polar_anoms = anoms_loc_tm[anoms_loc_tm.ORBIT == 'P'].reset_index(drop=True)
elliptical_anoms = anoms_loc_tm[anoms_loc_tm.ORBIT == 'E'].reset_index(drop=True)


# Observe that the majority of GEO anomalies occur in the midnight to dawn sector.

# In[ ]:


plt.figure(figsize=(12,8))
plt.hist(geo_anoms.STIMEL, bins=24)
plt.xticks([hour*100 for hour in range(24)],
           ['%2d' %hr for hr in range(24)])
plt.title('Geostationary Spacecraft Anomalies by Local Time')
plt.xlabel('Hour')
plt.ylabel('Count');


# Satellites in inclined and polar orbits tend to experience 12 hours apart (local time) since they repeat these orbits. These are commonly sun-synchronous.
# https://earthobservatory.nasa.gov/features/OrbitsCatalog/page2.php

# In[ ]:


plt.figure(figsize=(12,8))
plt.hist(inclined_anoms.STIMEL, bins=24)
plt.xticks([hour*100 for hour in range(24)],
           ['%2d' %hr for hr in range(24)])
plt.title('Inclined Orbit Spacecraft Anomalies by Local Time')
plt.xlabel('Hour')
plt.ylabel('Count');


# In[ ]:


plt.figure(figsize=(12,8))
plt.hist(polar_anoms.STIMEL, bins=24)
plt.xticks([hour*100 for hour in range(24)],
           ['%2d' %hr for hr in range(24)])
plt.title('Polar-orbiting Spacecraft Anomalies by Local Time')
plt.xlabel('Hour')
plt.ylabel('Count');


# Elliptical orbits tend to vary more often with respect to local time. While there are 326 anomalies for these satellites listed in the database, only 9 local times are recorded.

# 

# In[ ]:


plt.figure(figsize=(12,8))
plt.hist(elliptical_anoms.STIMEL, bins=24)
plt.xticks([hour*100 for hour in range(24)],
           ['%2d' %hr for hr in range(24)])
plt.title('Elliptical Orbit Spacecraft Anomalies by Local Time')
plt.xlabel('Hour')
plt.ylabel('Count');

