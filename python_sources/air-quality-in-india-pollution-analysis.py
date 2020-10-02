#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


air_quality = pd.read_excel("/kaggle/input/airquality/AirQuality.xlsx")


# In[ ]:


type(air_quality)


# In[ ]:


air_quality.head()


# In[ ]:


air_quality.describe()


# **# Grouping by States**

# In[ ]:


groups_state = air_quality.groupby('State')


# In[ ]:


groups_state.head()


# ## MEAN OF STATES
# 

# In[ ]:


groups_state.mean()


# In[ ]:


type(groups_state)


# ## MAKE DATA FRAME OF GROUPS_STATES

# In[ ]:


group_state_df = pd.DataFrame(groups_state)


# In[ ]:


type(group_state_df)


# In[ ]:


group_state_df.head()


# ## MAKE DATA FRAME OF AIR_QUALITY 

# In[ ]:


air_quality_df = pd.DataFrame(air_quality)


# In[ ]:


air_quality_df.head()


# ## Plot of Histograme of (air_quality_df.Avg)

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


plt.hist(air_quality_df.Avg, histtype ='bar', rwidth=0.8)
plt.xlabel("AVERAGE")
plt.ylabel("COUNT")
plt.title("AVERAGE OF AIR QUALITY OF INDIA")
plt.show()


# # Histograme of (air_quality_df.State)

# In[ ]:


plt.figure(figsize=(17,7), dpi = 100)
sns.countplot(x='State',data=air_quality)
plt.xlabel('State')
plt.tight_layout()


# # Histograme of (air_quality_df.MAX)

# In[ ]:


plt.hist(air_quality_df.Max, histtype ='bar', rwidth=0.8)
plt.xlabel("MAX")
plt.ylabel("COUNT")
plt.title("MAX OF AIR QUALITY OF INDIA")
plt.show()


# # Histograme of (air_quality_df.MIN)

# In[ ]:


plt.hist(air_quality_df.Min, histtype ='bar', rwidth=0.8)
plt.xlabel("Min")
plt.ylabel("COUNT")
plt.title("Min OF AIR QUALITY OF INDIA")
plt.show()


# # plot of Pollutants

# In[ ]:


air_quality['Pollutants'].value_counts().plot()
plt.xlabel("Pollutants")
plt.ylabel("COUNT")
plt.title("Pollutants OF AIR QUALITY OF INDIA")
plt.show()


# # Type Of Pollutants Of AIR QUALITY IN INDIA

# In[ ]:


air_quality['Pollutants'].value_counts().plot('bar')
plt.xlabel("Pollutants")
plt.ylabel("COUNT")
plt.title("Pollutants OF AIR QUALITY OF INDIA")
plt.show()


# # EXTRACT YEAR FROM DATA AND CHANGE DATA TYPE AND UPDATE DATA

# In[ ]:


air_quality['lastupdate'].head()


# In[ ]:


air_quality.lastupdate.str.slice(-5, -3).astype(int).head()


# In[ ]:


air_quality['lastupdate'] = pd.to_datetime(air_quality.lastupdate)
air_quality.head()


# # See the changes in lastupdate data type

# In[ ]:


air_quality.dtypes


# In[ ]:


## see the date in day of year 
air_quality.lastupdate.dt.dayofyear.head()


# In[ ]:


ts = pd.to_datetime('12-12-2018')


# In[ ]:


air_quality.loc[air_quality.lastupdate >= ts, :].head()


# In[ ]:


air_quality.head()


# PLOTING OF STATE

# In[ ]:


from matplotlib import pyplot
pyplot.plot(air_quality.State)
plt.xlabel("COUNT")
plt.ylabel("STATES")
plt.title("COUNTS OF STATES")
pyplot.show()


# # GROUPING BY STATE

# In[ ]:


group_state = air_quality.groupby('State')


# # Mean Pollution

# In[ ]:


group_state.mean().head()


# In[ ]:


group_state.max().head()


# In[ ]:


list(air_quality['Pollutants'].unique())


# # USE FUNCTION IN PLOT (PLOTING OF POLLUTANTS)

# In[ ]:


pollutant = list(air_quality['Pollutants'].unique())
for poll in pollutant:
    plt.figure(figsize=(18,8), dpi = 100)
    sns.countplot(air_quality[air_quality['Pollutants'] == poll]['State'], data = air_quality)
    plt.tight_layout()
    plt.title(poll)


# In[ ]:


list(air_quality['State'].unique())


# In[ ]:




