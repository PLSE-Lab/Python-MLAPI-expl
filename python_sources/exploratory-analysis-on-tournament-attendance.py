#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # What is SmashGG
# 
# [SmashGG](https://smash.gg/) is used to create videogame tournaments, add events to that tournament (for individual games), and run the bracket. 
# 
# # About the Data
# 
# The data discussed in this notebook is from all SmashGG tournaments between December 1st, 2018 to June 1st, 2020. It is mostly concerned with tournament attendants, which can be any individual involved in the tournament who registered with SmashGG.
# 
# # Objective
# 
# To determine how COVID has affected tournament attendance. SmashGG is very popular with the United States fighting game community (FGC), of which I am a part. The FGC is mostly organized around offline (in person) tournaments, but the community has been forced to adopt more and more online events for safety during the ongoing pandemic.
# 
# Tournament Attendance is extremely important to growing the community, and supporting long-standing members. An understanding of how attendance has changed could inform decisions moving forward about rescheduling or modifying events.

# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = pd.read_csv("/kaggle/input/smashgg-tournament-attendees-2019-through-mid2020/ggdf.csv")

df['startDate'] = pd.to_datetime(df['startAt'], unit='s')
df['endDate'] = pd.to_datetime(df['endAt'], unit='s')

#Filtered out June data because month is not over, grouped by Month/ Year, and isOnline, summed numAttendees over
#each month
by_month = df.loc[pd.to_datetime(df['startAt'], unit='s') <= pd.to_datetime(1590953558, unit='s')].set_index('startDate').groupby([pd.Grouper(freq="M"),'isOnline'])['numAttendees'].sum().reset_index()

#Sum of Attendees each month for offline tournaments
offendees = by_month.query('isOnline == 0')['numAttendees'].rename("offlineAttendees")
#Sum of Attendees each month for online tournaments
onendees = by_month.query('isOnline == 1')['numAttendees'].rename("onlineAttendees")
#Join Offline and Online Attendees with Monthly Attendees
joined_df = by_month.join(offendees, how='left').join(onendees, how='left')
#Sum values by month to get data to plot
plot_df = joined_df.set_index('startDate').groupby([pd.Grouper(freq="M")])[['numAttendees','offlineAttendees','onlineAttendees']].sum()


fig, ax = plt.subplots(figsize = (12,6)) 
fig = sns.lineplot(data = plot_df, ax=ax)
plt.xlabel("Month")
plt.ylabel("Sum of Attendees")
plt.title("Attendees at All SmashGG Tournaments Over Time")
x_dates = pd.date_range(start ='12-1-2018', end ='5-31-2020', freq ='2M').strftime('%b-%Y')

ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
_ = ""


# In[ ]:


#Group data by countryCode then sum Attendees. Sort Descending.
country = df.groupby('countryCode')['numAttendees'].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.barplot(x='countryCode',y='numAttendees', data = country.reset_index().head(10),
            palette = 'hls',  
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'   
            )

plt.xlabel("Country Code")
plt.ylabel("Total Number of Attendees")
plt.title("Highest SmashGG Attendants by Country")
_ = ""


# Looks like the United States has most SmashGG attendants by a large margin. One caveat of this whole enterprise is that, although SmashGG is very popular with the FGC in the United States, it is possible that this data is skewed by a lack of activity in other countries. 
# 
# I will eventually normalize attendants using each Country's population so I can analyze percentage-change in attendance year-by-year. It will be telling if 2020 is an outlier due to COVID; however, SmashGG recent dominance over its competition (like Challonge) might also be a factor.
# 
# 

# In[ ]:


#Made a function that prepares the data as before, filtering by Country Code
def plot_by_country(df, countryCode=None):
    if countryCode is None:
        c_df = df
    else:
        c_df = df.loc[df['countryCode'] == countryCode]

    c_month = c_df.loc[pd.to_datetime(c_df['startAt'], unit='s') <= pd.to_datetime(1590953558, unit='s')]    .set_index('startDate').groupby([pd.Grouper(freq="M"),'isOnline'])['numAttendees'].sum().reset_index()

    c_offendees = c_month.query('isOnline == 0')['numAttendees'].rename("offlineAttendees")
    c_onendees = c_month.query('isOnline == 1')['numAttendees'].rename("onlineAttendees")

    c_joined = c_month.join(c_offendees, how='left').join(c_onendees, how='left')

    c_plot_df = c_joined.set_index('startDate').groupby([pd.Grouper(freq="M")])[['numAttendees','offlineAttendees','onlineAttendees']].sum()

    return c_plot_df  


# In[ ]:


import datetime as dt
#Filter down to US data to see if Attendancy change is reaction to lockdowns there
data = plot_by_country(df, 'US')
                  
fig, ax = plt.subplots(figsize = (12,6)) 
fig = sns.lineplot(data = data, ax=ax)
plt.xlabel("Month")
plt.ylabel("Sum of Attendees")
plt.title("United States Smash GG Attendees")
x_dates = pd.date_range(start ='12-1-2018', end ='5-31-2020', freq ='2M').strftime('%b-%Y')

ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')

plt.axvline(dt.datetime(2020, 3, 15),0,1,linestyle='--',linewidth=1,color = 'red') 
#First Mainland COVID Lockdown in SF was on March 15th, according to wikipedia
#https://en.wikipedia.org/wiki/COVID-19_pandemic_lockdowns#United_States

plt.text(dt.datetime(2020, 3, 15),45000,'Lockdowns Start',rotation=90, horizontalalignment='right',verticalalignment='top')
_ = ""


# Evo Online is happening this summer, and lockdowns in the US are starting to relax. When the summer is over I will add to the dataset and rerun this notebook.
