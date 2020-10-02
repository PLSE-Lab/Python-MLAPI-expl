#!/usr/bin/env python
# coding: utf-8

# I always have this question in mind --- how to find my potential users. In other words, in case where I wish to promote my app to new users, where and when can I expect to meet them. 
# 
# Before a new app launch, a check point in market research is to study similar products in the market. So here I am going to use data from a category of app, and try to infer on the users demographics in the aspect of activity. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In the giving datasets, the most trivially revealing attribute of an app is its category. Apps belonging to the same category are of similar traits to some extent. We first join app_labels.csv and label_categories.csv to get a mapping between app and category

# In[ ]:


label = pd.read_csv('../input/app_labels.csv')
cat = pd.read_csv('../input/label_categories.csv')
app_cat = pd.merge(label, cat, how='left', on='label_id')
print(app_cat.head())
del label, cat


# In[ ]:


app_cat['category'].value_counts()


# Too much categories are defined. The category trait may not be as indicative as it was expected. 
# In my case, I wish to understand user demographics for one specific category of app. So I will take out one category (i.e. Fiance) for exploration. 
# 
# The category summary script is taken from  categorizing labels by nikdudaev 
# https://www.kaggle.com/nickdudaev/talkingdata-mobile-user-demographics/categorizing-labels/run/302941

# In[ ]:


def to_Finance(x):
    if re.search('([iI]ncome)|([pP]rofitabil)|([lL]iquid)|([rR]isk)|([bB]ank)|([fF]uture)|([fF]und)|([sS]tock)|([sS]hare)',
                 x) is not None:
        return('Finance')
    if re.search('([fF]inanc)|([pP]ay)|(P2P)|([iI]nsura)|([lL]oan)|([cC]ard)|([mM]etal)|'
                 '([cC]ost)|([wW]ealth)|([bB]roker)|([bB]usiness)|([eE]xchange)', x) is not None:
        return('Finance')
    if x in ['High Flow', 'Housekeeping', 'Accounting', 'Debit and credit', 'Recipes', 'Heritage Foundation', 'IMF',]:
        return('Finance')
    else:
        return(x)

app_cat['general_cat'] = app_cat['category'].apply(to_Finance)


# In[ ]:


app_finance = app_cat[app_cat['general_cat']=='Finance']
print(app_finance.head())
del app_cat


# The next step is to append user demographic fields (i.e. event time, event location) to Finance app. 

# In[ ]:


app_ev = pd.read_csv('../input/app_events.csv')
ev = pd.read_csv('../input/events.csv')
events = pd.merge(app_ev, ev, how='inner', on='event_id')
del app_ev, ev

print(events.head())


# In[ ]:


finance_events = events[events['app_id'].isin(app_finance['app_id'])]
del events, app_finance
print(finance_events.head())


# In[ ]:


print(finance_events.shape)


# Not bad. We got enough Finance app events for study. We further sample it down to events happening in China

# In[ ]:


# Sample it down to only the China region
lon_min, lon_max = 75, 135
lat_min, lat_max = 15, 55

idx_china = (finance_events["longitude"] > lon_min) &            (finance_events["longitude"] < lon_max) &            (finance_events["latitude"] > lat_min) &            (finance_events["latitude"] < lat_max)

china_finance = finance_events[idx_china]

print (china_finance.shape)


# In[ ]:


plt.figure(1, figsize=(12,6))

m_1 = Basemap(projection='merc',
             llcrnrlat=lat_min,
             urcrnrlat=lat_max,
             llcrnrlon=lon_min,
             urcrnrlon=lon_max,
             lat_ts=35,
             resolution='i')

m_1.drawmapboundary(fill_color='#000000')                # black background
m_1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

# Plot the data
mxy = m_1(china_finance["longitude"].tolist(), china_finance["latitude"].tolist())
m_1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=0.05, zorder=5)

plt.title("China view of finance events")
plt.show()


# Plotting of events in China is consistent with population density in China. 
# 
# Now we zoom in to Beijing.

# In[ ]:


# Sample it down to only the Beijing region
lon_min, lon_max = 116, 117
lat_min, lat_max = 39.75, 40.25

idx_beijing = (finance_events["longitude"]>lon_min) &              (finance_events["longitude"]<lon_max) &              (finance_events["latitude"]>lat_min) &              (finance_events["latitude"]<lat_max)

beijing_finance = finance_events[idx_beijing]

# Mercator of Beijing
plt.figure(2, figsize=(12,6))

m_2 = Basemap(projection='merc',
             llcrnrlat=lat_min,
             urcrnrlat=lat_max,
             llcrnrlon=lon_min,
             urcrnrlon=lon_max,
             lat_ts=35,
             resolution='c')

m_2.drawmapboundary(fill_color='#000000')                # black background
m_2.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

# Plot the data
mxy = m_2(beijing_finance["longitude"].tolist(), beijing_finance["latitude"].tolist())
m_2.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.1, zorder=5)

plt.title("Beijing view of events")
plt.show()


# Plotting of events in Beijing is still consistent with population density. Events amassed in central area. 
# 
# Based on the plotting in Beijing, we now plot out events for different segmentation in interests. The fields we take into consideration are:
# 1.  male vs. female
# 2 . active events vs. inactive events. My inference of these two events are:
# 1) active events (is_active=1): events where users are using the app
# 2) inactive events (is_active=0): events where users download the app
# 
# In the following section, I subset the events data and plot them accordingly. 

# In[ ]:


# Load the train data and join on the events
df_train = pd.read_csv("../input/gender_age_train.csv")

bj_finance_demo = pd.merge(df_train, beijing_finance, how="inner", on="device_id")

df_m = bj_finance_demo[bj_finance_demo["gender"]=="M"]
df_f = bj_finance_demo[bj_finance_demo["gender"]=="F"]

print(df_m.shape, df_f.shape)


# In[ ]:


df_m['is_active'].value_counts()[1] / len(df_m['is_active'])


# In[ ]:


df_f['is_active'].value_counts()[1] / len(df_f['is_active'])


# In[ ]:


def bj_map():
    bj_map= Basemap(projection='merc',
                 llcrnrlat=lat_min,
                 urcrnrlat=lat_max,
                 llcrnrlon=lon_min,
                 urcrnrlon=lon_max,
                 lat_ts=35,
                 resolution='c')
    bj_map.drawmapboundary(fill_color='#000000')              
    bj_map.drawcountries(linewidth=0.1, color="w")        
    return bj_map


# In[ ]:


plt.figure(3, figsize=(12,6))

# Male/female plot
# df_m and df_f 
plt.subplot(321)
m3a = bj_map()
mxy = m3a(df_m["longitude"].tolist(), df_m["latitude"].tolist())
m3a.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.1, zorder=5)
plt.title("Male events in Beijing")

plt.subplot(322)
m3b = bj_map()
mxy = m3b(df_f["longitude"].tolist(), df_f["latitude"].tolist())
m3b.scatter(mxy[0], mxy[1], s=5, c="#fd3096", lw=0, alpha=0.1, zorder=5)
plt.title("Female events in Beijing")


# Active Male/female plot
df_m_active = df_m[df_m['is_active']==1]
df_f_active = df_f[df_f['is_active']==1]

plt.subplot(323)
m4a = bj_map()
mxy = m4a(df_m_active["longitude"].tolist(), df_m_active["latitude"].tolist())
m4a.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.1, zorder=5)
plt.title("Male active events in Beijing")

plt.subplot(324)
m4b = bj_map()
mxy = m4b(df_f_active["longitude"].tolist(), df_f_active["latitude"].tolist())
m4b.scatter(mxy[0], mxy[1], s=5, c="#fd3096", lw=0, alpha=0.1, zorder=5)
plt.title("Female active events in Beijing")


# Inactive Male/female plot
df_m_inactive = df_m[df_m['is_active']==0]
df_f_inactive = df_f[df_f['is_active']==0]

plt.subplot(325)
m5a = bj_map() 
mxy = m5a(df_m_inactive["longitude"].tolist(), df_m_inactive["latitude"].tolist())
m5a.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.1, zorder=5)
plt.title("Male inactive events in Beijing")

plt.subplot(326)
m5b = bj_map()
mxy = m5b(df_f_inactive["longitude"].tolist(), df_f_inactive["latitude"].tolist())
m5b.scatter(mxy[0], mxy[1], s=5, c="#fd3096", lw=0, alpha=0.1, zorder=5)
plt.title("Female inactive events in Beijing")

plt.show()


# Plotting by segments have revealed different patterns. 
# 
# Male and female recipients have slightly different zones, so as active and inactive users. If you have a portrait of potential users to seek after, you roughly know where to find them. 
# 
# Next we explore on time of events. 

# In[ ]:


# plot time of events

plt.figure(4, figsize=(12,18))
plt.subplot(611)
plt.hist(df_m['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("male")

plt.subplot(612)
plt.hist(df_f['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("female")

plt.subplot(613)
plt.hist(df_m_active['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("male active")

plt.subplot(614)
plt.hist(df_f_active['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("female active")

plt.subplot(615)
plt.hist(df_m_inactive['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("male inactive")

plt.subplot(616)
plt.hist(df_f_inactive['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("female inactive")

plt.subplots_adjust(hspace=.8)
plt.show()


# Plotting on time by male and female revealed interesting patters. 
# 
# 1) male vs. female: generally female users showed a declining activity trend in the afternoon. Specifically, around 16-19 o'clock in a day. It may be due to the fact that females need to take care of housework (i.e. take children from school, buy stuff for dinner, prepare for dinner) around that time. 
# 
# 2) Way to work (7-9 o'clock) and after-work (20-22 o'clock) are common high activity period for all groups. Golden hours are universal :)
# 
# Still much to dig out. For now, it is only a exploration view on two user demographics dimensions -- location and time. This is useful in cases where you have produced your user portrait. You know who you want, and the next step is where and when to find them.  
