#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
from __future__ import division
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

print(os.listdir("../input"))


# In[ ]:


xl = pd.ExcelFile('../input/AirQuality.xlsx')


# In[ ]:


xl.sheet_names


# In[ ]:


df = xl.parse('Sheet1')


# In[ ]:


df.head()


# In[ ]:


df_pm25 = df[df['Pollutants']=='PM2.5']
df_no2 = df[df['Pollutants']=='NO2']
df_so2 = df[df['Pollutants']=='SO2']
df_co = df[df['Pollutants']=='CO']
df_ozone = df[df['Pollutants']=='OZONE']


# In[ ]:


df.shape


# In[ ]:


plt.figure(figsize=(10,6))
print (df['State'].value_counts())
ax = sns.countplot(x='State', data=df);
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");


# In[ ]:


fig, axes = plt.subplots(nrows=5,ncols=4,figsize=(17,20))
space = 0.25
lspace = 0.24
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1.25);
i = 0
for state in df['State'].unique():
    ax1 = sns.countplot(data=df[df['State']==state],x='city',ax=axes.flatten()[i]);
    ax1.set_title(state)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right");
    i = i + 1


# ## All india statistics analysis

# In[ ]:


df_pm25.head()
all_df_list = [df_pm25,df_no2,df_so2,df_co,df_ozone]
all_df_poll_list = ['PM2.5','NO2','SO2','CO','OZONE']

i=0
for df_poll,df_name in zip(all_df_list,all_df_poll_list):
    print (df_name)
    print (df_poll[['Avg','Max','Min']].describe())
    print('\n')


# ## Inferences:
# 
# ### PM2.5
# Refrer to [this](http://aqicn.org/faq/2013-09-09/revised-pm25-aqi-breakpoints/) link to learn about the acceptable values. It seems that the air in India is indeed unhealthly. The avergae PM2.5 value is 251 which puts it in the 'Very Unhealthy' category. The maximum value was 51 at a particluar place which puts it at the 'hazardous' level.
# 
# ### NO2
# Refer to [this](https://www3.epa.gov/airnow/no2.pdf) link to learn about the acceptable values. The average value is 62. This puts it in the 'moderate' category. The verdict is: "Individuals who are unusually sensitive to nitrogen dioxide should consider limiting prolonged outdoor exertion".
# 
# ### SO2
# According to [this](http://www.air-quality.org.uk/04.php) link, in most major cities in UK, the So2 mean values range from 15-20. The mean value from this dataset is 19. The most important sources of SO2 are fossil fuel combustion.
# 
# ### CO
# [This](https://www.kidde.com/home-safety/en/us/support/help-center/browse-articles/articles/what_are_the_carbon_monoxide_levels_that_will_sound_the_alarm_.aspx) link states that a value of around 50, dosent have any serious effects on healthy adults even with a continuous exposire to upto 8 hours. The maximum at a particular place reached close to 200. This might casue Slight headache, fatigue, dizziness, and nausea after two to three hours.
# 
# ### OZONE
# [This](https://www.airnow.gov/index.cfm?action=pubs.aqiguideozone) link has detaits regarding ozone levels. The with the mean value of 54, it is classified as 'moderate'. Unusually sensitive people should consider reducing prolonged or heavy outdoor exertion.
# **

# City specific analysis coming soon. If you found this analysis helpful, please consider upvoting it!
