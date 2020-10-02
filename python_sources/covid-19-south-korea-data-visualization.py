#!/usr/bin/env python
# coding: utf-8

# # Load packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import plotly.offline
import cufflinks as cf 
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

cf.set_config_file(theme='white')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load the data
# Let's load the dataset file first

# In[ ]:


patient=pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')
time=pd.read_csv('/kaggle/input/coronavirusdataset/time.csv')
route=pd.read_csv('/kaggle/input/coronavirusdataset/route.csv')
trend=pd.read_csv('/kaggle/input/coronavirusdataset/trend.csv')


# In[ ]:


patient.head()


# In[ ]:


time.head()


# In[ ]:


route.head()


# In[ ]:


trend.head()


# # convert date to datetime
# 
# for extract month, day_name, day! 

# In[ ]:


def date_time_col(df, col, form='%Y/%m/%d'):
    df[col] = pd.to_datetime(df[col], format=form)
    return df[col]

patient['confirmed_date'] = date_time_col(patient,'confirmed_date')
patient['released_date'] = date_time_col(patient,'released_date')
time['date'] = date_time_col(time,'date')


# In[ ]:


patient['month_confirmed'] = patient['confirmed_date'].dt.month
patient['dow_confirmed'] = patient['confirmed_date'].dt.day_name()
patient['day_confirmed'] = patient['confirmed_date'].dt.day


# # Visualization of data
# * Epidemiological data of COVID-19 patients in South Korea
# * Route data of COVID-19 patients in South Korea (where they had visited)
# * Time series data of COVID-19 status in South Korea
# * Trend data of the keywords searched in NAVER

# In[ ]:


total = len(patient)
plt.figure(figsize=(15,19))

plt.subplot(311)
g = sns.countplot(x="day_confirmed", data=patient)
g.set_title("Confirmed Patient by DAY OF MONTH", fontsize=20)
g.set_ylabel("Count",fontsize= 17)
g.set_xlabel("Day of Confirmations", fontsize=17)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
g.set_ylim(0, max(sizes) * 1.15)

plt.subplot(312)
g2 = sns.countplot(x="dow_confirmed", data=patient)
g2.set_title("Confirmed Patient by DAY OF WEEK", fontsize=20)
g2.set_ylabel("Count",fontsize= 17)
g2.set_xlabel("Day of the Week of Confirmations", fontsize=17)
sizes=[]
for p in g2.patches:
    height = p.get_height()
    sizes.append(height)
    g2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g2.set_ylim(0, max(sizes) * 1.15)

plt.subplot(313)
g1 = sns.countplot(x="month_confirmed", data=patient)
g1.set_title("Confirmed Patient by MONTHS", fontsize=20)
g1.set_ylabel("Count",fontsize= 17)
g1.set_xlabel("", fontsize=17)
sizes=[]
for p in g1.patches:
    height = p.get_height()
    sizes.append(height)
    g1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g1.set_ylim(0, max(sizes) * 1.15)

plt.subplots_adjust(hspace = 0.3)

plt.show()


# In[ ]:


sex_series = patient['sex'].value_counts()
sex_series.iplot(kind='bar', yTitle='Count', title='the sex of the patient')


# In[ ]:


country_series = patient['country'].value_counts()
country_series.iplot(kind='bar', yTitle='Count', title='the country of the patient')


# In[ ]:


region_series = patient['region'].value_counts()
region_series.iplot(kind='bar', yTitle='Count', title='the region of the patient')


# In[ ]:


group_series = patient['group'].value_counts()
group_series.iplot(kind='bar', yTitle='Count', title='the collective infection')


# In[ ]:


infection_reason_series = patient['infection_reason'].value_counts()
infection_reason_series.iplot(kind='bar', yTitle='Count', title='the reason of infection')


# In[ ]:


state_series = patient['state'].value_counts()
state_series.iplot(kind='bar', yTitle='Count', title='isolated / released / deceased')


# # latitude & longitude
# Let's visualize the locations of the route

# In[ ]:


center = [37.541, 126.986] # seoul 
m = folium.Map(location=center, zoom_start=6.5)

df = route[['id', 'latitude', 'longitude']]


for i in df.index[:]:
    folium.CircleMarker(location =df.loc[i, ['latitude', 'longitude']],
                        tooltip = df.loc[i, 'id'], radius = 6,
                        fill=True,
                        fill_opacity=0.7
                        ).add_to(m)

m


# In[ ]:


province_series = route['province'].value_counts()
province_series.iplot(kind='bar', yTitle='Count', title='Special City / Metropolitan City / Province(-do)')


# In[ ]:


visit_series = route['visit'].value_counts()
visit_series.iplot(kind='bar', yTitle='Count', title='the type of place visited')


# In[ ]:


acc_df = time[['date','acc_test','acc_negative','acc_confirmed','acc_released','acc_deceased']].set_index("date")


# In[ ]:


py.iplot([{
    'x': acc_df.index,
    'y': acc_df[col],
    'name': col
}  for col in acc_df.columns], filename='cufflinks/simple-line')


# In[ ]:


new_df = time[['date','new_test', 'new_negative', 'new_confirmed', 'new_released', 'new_deceased']].set_index("date")


# In[ ]:


py.iplot([{
    'x': new_df.index,
    'y': new_df[col],
    'name': col
}  for col in new_df.columns], filename='cufflinks/simple-line')


# In[ ]:


trend_df = trend[['date','cold', 'flu', 'pneumonia', 'coronavirus']].set_index("date")


# In[ ]:


py.iplot([{
    'x': trend_df.index,
    'y': trend_df[col],
    'name': col
}  for col in trend_df.columns], filename='cufflinks/simple-line')


# In[ ]:




