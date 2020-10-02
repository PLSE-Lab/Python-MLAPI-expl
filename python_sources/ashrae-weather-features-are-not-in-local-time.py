#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# **HYPOTHESIS**
# * **The weather data is not in local time**
# 
# **POTENTIAL CONSEQUENCES**
# * **We will count with very uninformative weather features for our models**
# * **Straightforward features such as DAY, HOUROFDAY, etc. will have little meaning if they do not match the weather conditions and the measured data localtime timestamp**

# In[ ]:


weather_data = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")
weather_data.set_index('timestamp', inplace=True)
weather_data.index = pd.to_datetime(weather_data.index)
weather_data['hourofday'] = weather_data.index.hour
weather_data['dayofyear'] = weather_data.index.dayofyear
weather_data.tail()


# **Lets plot some days in site 0 during winter**

# In[ ]:


weather_data[weather_data['site_id']== 0]['air_temperature'][:24].plot()


# **Lets plot the some days in site 0 during Summer**

# In[ ]:


weather_data[weather_data['site_id']== 0]['air_temperature'][5496:5520].plot()


# In[ ]:


weather_data[weather_data['site_id']== 0]['air_temperature'][24:48].plot()


# **Now Let's look at a random site, let's say site 10 for winter**

# In[ ]:


weather_data[weather_data['site_id']== 10]['air_temperature'][:24].plot()


# **Now for Summer**

# In[ ]:


weather_data[weather_data['site_id']== 10]['air_temperature'][5496:5520].plot()


# **PRELIMINAR CONCLUSIONS**
# 
# * **There seems to be something off. High temperatures during the day at 8:00pm or even at mid-night?**
# * **We need to do some extra checks: E.g., Estimate the the frequency of peak temperatures per hour of the day for each site**

# In[ ]:


site_ids = list(set(weather_data['site_id']))
result = pd.DataFrame()
for site in site_ids:
    weather_data_site = weather_data[weather_data['site_id']== site]
    t_air_max_every_day = weather_data_site.groupby('dayofyear')['air_temperature'].max()
    weather_data_site['hit_t_air_max'] = weather_data_site.apply(lambda x: 1 if x['air_temperature'] == t_air_max_every_day[x['dayofyear']] else 0,axis=1)
    result['site'+str(site)] = weather_data_site.groupby('hourofday')['hit_t_air_max'].sum().values


# In[ ]:


plt.figure(figsize=(20, 10))
sns.heatmap(result, annot=True)


# ** FINAL CONCLUSIONS**
# * **The peak temperature of outdoor air takes place most of the time between 7:00pm and 9:00pm (sites 0, 3,6,7,8,9, 11, 14, 15)**
# * **Some locations even register the peak daily temperature close to 1:00am (sites 2, 4, 10, and 13) **
# * **This does not seem to be a realistic behavior**
# * **I believe the data is not in a local time zone which makes hard to extract correct representations of features such as DAY, HOUROFDAY etc.**
# * **Should this behavior be different than that of the measurements, we might be having a mismatch of energy consumption and weather conditions, that would lead us to very uninformative weather features**
# * **Some locations could be ok, from what we could expect a typical peak of temperature to occur (between 12:00pm and 5:00pm).
# 
# ** NEXT STEPS**
# 
# * **Perhaps the organizers have some ideas if this is the case?**
# * **Perhaps the 'error' is systematic? i.e., all data's timestamp is shifted equally?**
