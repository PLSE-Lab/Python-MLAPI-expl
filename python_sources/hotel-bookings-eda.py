#!/usr/bin/env python
# coding: utf-8

# ## Question-Driven Exploratory Data Analysis(EDA) is an essential skill to master with as those questions you come up with become the initial motivations for yourself to explore the dataset and also train your mind to be familiar with this process.
# 
# ## In my opinion, a good data scientist not just having good technical skills but also stay curious to the data and good at storytelling

# ## * For simplicity, I used the term 'Confirmed' refering to 'Not Canceled' for bookings

# # Environment Setup

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Load Dataset

# In[ ]:


df_original = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
df_original.head()


# In[ ]:


df_original.tail()


# - Observing the dataset, it started in July,2015 and ended in August,2017. July and August both had three entries from 2015,2016,2017 while others all had two from whether 2015,2016 or 2016,2017. In this case, I dropped data from July 2017 to August 2017 to make sure the result did not bias to those two months.

# ## Drop cases from July,2017 to August,2017

# In[ ]:


df_original = pd.concat([df_original,df_original[(df_original.arrival_date_year == 2017) & ((df_original.arrival_date_month == 'August') | (df_original.arrival_date_month == 'July'))]]).drop_duplicates(keep=False)


# In[ ]:


df_original.tail()


# # EDA Phase

# ## Brief description of the dataset

# ### Number of datapoints in the dataset

# In[ ]:


display(f'Dataset shape is: {df_original.shape}',df_original.describe())


# ## Feature Examination

# ## Feature -> `is_canceled`

# ### Number of canceled VS number of confirmed

# In[ ]:


ax = sns.distplot(df_original.is_canceled,kde=False,color='r')
plt.title('Not Canceled VS. Canceled')
for p in ax.patches:
    if p.get_height() > 0: ax.text(p.get_x()+p.get_width()/2,p.get_height(),f"{int(p.get_height())}",fontsize=16) 


# - There are 30942 more confirmed bookings than canceled

# ## Let's first look at bookings that were not canceled

# ## Which year had the most bookings?

# In[ ]:


bookings_by_year_not_canceled = df_original[df_original.is_canceled == 0].groupby('arrival_date_year').arrival_date_year.count()
ax = sns.barplot(bookings_by_year_not_canceled.index,bookings_by_year_not_canceled.values)
plt.title('Number of Confirmed Bookings')
for p in ax.patches:
    ax.text(p.get_x()+p.get_width()/4,p.get_height(),f"{int(p.get_height())}",fontsize=16)


# * It seems like most bookings appears in 2016, and least in 2015 with a gap number of 9496 bookings

# ## Which month had the most bookings?

# In[ ]:


booking_by_year_month_not_canceled = df_original[df_original.is_canceled == 0].groupby(['arrival_date_year','arrival_date_month']).arrival_date_month.count().sort_values(ascending=False)
plt.figure(figsize=(10,30))
plt.title('Number of Confirmed Bookings by Year-Month')
ax = sns.barplot(booking_by_year_month_not_canceled.values,booking_by_year_month_not_canceled.index)
for p in ax.patches:
    ax.text(p.get_width(),p.get_y()+p.get_height()/2,f"{int(p.get_width())}",fontsize=16)


# - May,2017 had the most not canceled bookings of 2832, next followed by August,2016 with a tiny gap of 4

# In[ ]:


booking_by_month_not_canceled = df_original[df_original.is_canceled == 0].groupby('arrival_date_month').arrival_date_month.count().sort_values(ascending=False)
plt.figure(figsize=(10,15))
plt.title('Number of Confirmed Bookings by Month')
ax = sns.barplot(booking_by_month_not_canceled.values,booking_by_month_not_canceled.index)
for p in ax.patches:
    ax.text(p.get_width(),p.get_y() + p.get_height()/2,f"{int(p.get_width())}",fontsize=20)


# - In total, May had the most bookings which were 5408. This was expected because May,2017 was the time had the most booking and May,2016 was ranked at 7th place from last figure

# ## Now let's examine those canceled samples

# ## Which year had most canceled bookings?

# In[ ]:


bookings_by_year_canceled = df_original[df_original.is_canceled == 1].groupby('arrival_date_year').arrival_date_year.count()
ax = sns.barplot(bookings_by_year_canceled.index,bookings_by_year_canceled.values)
plt.title('Number of Canceled Bookings')
for p in ax.patches:
    ax.text(p.get_x()+p.get_width()/4,p.get_height(),f"{int(p.get_height())}",fontsize=16)


# - Still most canceled bookings were in 2016

# ## Which month had most canceled bookings?

# In[ ]:


booking_by_year_month_canceled = df_original[df_original.is_canceled == 1].groupby(['arrival_date_year','arrival_date_month']).arrival_date_month.count().sort_values(ascending=False)
plt.figure(figsize=(10,30))
plt.title('Number of Canceled Bookings by Year-Month')
sns.set(font_scale=1.1)
ax = sns.barplot(booking_by_year_month_canceled.values,booking_by_year_month_canceled.index)
for p in ax.patches:
    ax.text(p.get_width(),p.get_y()+p.get_height()/2,f"{int(p.get_width())}",fontsize=16)


# - August,2016 and May,2017 seem to be two most busy periods as they were the top 2 which got most cancels and confirms

# In[ ]:


booking_by_month_canceled = df_original[df_original.is_canceled == 1].groupby('arrival_date_month').arrival_date_month.count().sort_values(ascending=False)
plt.figure(figsize=(10,15))
plt.title('Number of Canceled Bookings by Month')
ax = sns.barplot(booking_by_month_canceled.values,booking_by_month_canceled.index)
for p in ax.patches:
    ax.text(p.get_width(),p.get_y() + p.get_height()/2,f"{int(p.get_width())}",fontsize=20)


# In[ ]:


cancel_rate_by_month = ((df_original[df_original.is_canceled == 1].groupby('arrival_date_month').arrival_date_month.count() / df_original[df_original.is_canceled == 0].groupby('arrival_date_month').arrival_date_month.count()) * 100).sort_values(ascending=False)
plt.figure(figsize=(10,15))
ax = sns.barplot(cancel_rate_by_month.values,cancel_rate_by_month.index)
plt.title('Cancel Rates by Month')
for p in ax.patches:
    ax.text(p.get_width(),p.get_y() + p.get_height()/2,f"{round(p.get_width(),2)}%",fontsize=20)


# - Max cancel rate was 40.97% in June and least was 25.61% in November

# ## Feature -> `hotel`

# ## How is the distribution of feature `hotel`?

# In[ ]:


hotel_dist = pd.concat([df_original.hotel.value_counts(),df_original[df_original.is_canceled == 0].hotel.value_counts(),df_original[df_original.is_canceled==1].hotel.value_counts()],axis=1)
hotel_dist.columns = ['Total','Confirmed','Canceled']
ax = hotel_dist.plot.bar(rot=0,figsize=(8,6))
for p in ax.patches:
    ax.text(p.get_x(),p.get_height(),f"{int(p.get_height())}",fontsize=16)


# - A difference of 14084 samples between City and Resort Hotels
# - 72% confirm rate for city hotel, 79% confirm rate for resort hotel
# - 28% cancel rate for city hotel, 21% cancel rate for resort hotel

# ## Feature -> `lead_time`

# ## How many days ahead people usually book?

# In[ ]:


df_original.lead_time.agg(['min','mean','max'])


# - People usually made a booking 2 months ahead
# - Extreme cases appeared to be someone booked on the same day they checked in or someone made a booking 2 years ahead
# - I wonder if there's any diff between canceled and confirmed cases?

# ## Any difference in lead_time between canceled and confirmed cases?

# In[ ]:


lead_time_diff = pd.concat([df_original[df_original.is_canceled == 0].lead_time.agg(['min','mean','max']),df_original[df_original.is_canceled == 1].lead_time.agg(['min','mean','max'])],axis=1)
lead_time_diff.columns=['Confirmed','Canceled']
ax = lead_time_diff.plot.bar(rot=0,figsize=(8,6))
for p in ax.patches:
    ax.text(p.get_x(),p.get_height(),f"{int(p.get_height())}",fontsize=16)


# - In average, canceled bookings were made 3 months ahead while confirmed were made 2 months ahead. A gap of 1 month between
# - For the extreme case, the longest canceled booking was made 1 year and 3 months ahead while comparing to 2 years ahead for the confirmed case. There was a gap of 8 months between

# ## Feature -> `stays-in-weekend-nights` & `stays-in-week-nights`

# ## When I look at the dataset, I had noticed that some records had 0 night stay in both weekend and week, I would like to investigate it further

# In[ ]:


zero_night_stays = df_original[(df_original.stays_in_weekend_nights == 0) & (df_original.stays_in_week_nights == 0)]
zero_night_stays


# - There were 597 cases that customer did not stay for a single night which meant they either canceled the booking or checked-out on the same day they checked-in
# - The case at index 1 was the extreme case that the customer booked 2 years ahead but did not stay for any night when they arrived, hmm.. interesting

# ## How many were canceled and how many were checked out on the same day?

# In[ ]:


zero_night_stays.is_canceled.value_counts()


# - 571 cases were checked out on the same day they checked in
# - 26 cases were canceled bookings
