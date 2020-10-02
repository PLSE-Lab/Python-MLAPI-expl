#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df=pd.read_excel('../input/flight_schedule.xlsx')
df.head()


# In[ ]:


df.Date=pd.to_datetime(df.Date)
df.Time=df.Time.astype(str)
df.Time=pd.to_datetime(df.Time)


# In[ ]:


df['Weekday'] = df.Date.dt.weekday
df['Weekday'] = df['Weekday'].apply(lambda x: calendar.day_name[x])
df['Time'] = df.Time.dt.hour
df.info()


# In[ ]:


cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)
df['Weekday'] =df['Weekday'].astype(cat_dtype)


# ## Flights per day of week

# In[ ]:


weekdays_total=df.groupby('Weekday').size().plot.bar(title='Flights per weekday', figsize=(8,5))
totals = []

# find the values and append to list
for i in weekdays_total.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in weekdays_total.patches:
    # get_x pulls left or right; get_height pushes up or down
     weekdays_total.text(i.get_x()+0.05, i.get_height()+1,             i.get_height(), fontsize=12,
                color='black')
plt.savefig('books_read.png')


# ## The TOP-15 destinations weekly

# In[ ]:


grouped=df.groupby('Route').size().sort_values(ascending=False)
graph_grouped=grouped[:15].plot.bar(title='The most popular destinations weekly', figsize=(12,8))

for i in graph_grouped.patches:
    totals.append(i.get_height())

total = sum(totals)

for i in graph_grouped.patches:
     graph_grouped.text(i.get_x()+.05, i.get_height()+1,             i.get_height(), fontsize=12,
                color='black')


# ## Air traffic on workdays

# In[ ]:


tf=df.loc[df.Weekday.isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])]
workdays=tf.groupby(['Time','Weekday']).size().unstack().plot(kind='line',linewidth=3.0, title='Departure time distribution on workdays',                                                                        figsize=(18,10))
workdays
plt.xticks(np.arange(0,24,step=1))
plt.ylabel('Flights amount')
plt.legend()


# ## Air traffic on weekends

# In[ ]:


kf=df.loc[df.Weekday.isin(['Saturday','Sunday'])]


weekends=kf.groupby(['Time','Weekday']).size().unstack().plot(kind='line',linewidth=3.0, title='Departure time distribution on weekends', figsize=(16,6))
weekends
plt.xticks(np.arange(0, 24, step=1))
plt.ylabel('Flights amount')
plt.show()


# ## Most popular aircraft types

# In[ ]:


df.Type.unique()


# In[ ]:


df.Type.value_counts()[:10]


# In[ ]:


df['Week'] = df.Date.dt.week


# In[ ]:


plt.figure(figsize=(6,6))
labels = 'DH4','CS300','B733','Other','B738','A320'
list = ['320N','CNJ750','E35L', 'C17', 'LRJ-60', 'CCJ', 'CL30', 'CRJ2', 'CNJ-560X', 'BET-200', 734, 752, 744, 'C250', 332, 321, 'M82','GRJ5', 'SU95','CRJ9','AT7',735]
df['Type'] = df['Type'].replace(list,'Other')
types=df[df['Week']==26]['Type'].value_counts()
plt.pie(types,labels=labels,autopct='%1.1f%%', startangle=60)
plt.axis('equal')
plt.title('Aircraft types per week', fontsize=20)
plt.tight_layout()

