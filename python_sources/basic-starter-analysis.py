#!/usr/bin/env python
# coding: utf-8

# # Hotel Booking Demand - Basic Starter Analysis

# Hi Guys! I thought I should pop my Kaggle notebook cherry by doing a simple analysis on this hotel booking data. Please fill free to leave any feedback in the comments, I will be adding to this notebook as time goes by. 
# 
# Also please do leave suggestions as to how I can make this better and where I should expand on.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:



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


data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')


# In[ ]:


def explore(data):
    summaryDf = pd.DataFrame(data.dtypes, columns=['dtypes'])
    summaryDf = summaryDf.reset_index()
    summaryDf['Name'] = summaryDf['index']
    summaryDf['Missing'] = data.isnull().sum().values
    summaryDf['Total'] = data.count().values
    summaryDf['MissPerc'] = (summaryDf['Missing']/data.shape[0])*100
    summaryDf['NumUnique'] = data.nunique().values
    summaryDf['UniqueVals'] = [data[col].unique() for col in data.columns]
    print(summaryDf.head(30))


# In[ ]:


explore(data)


# In[ ]:


data.dtypes


# Looking at correlations

# In[ ]:


sns.heatmap(data.corr())


# Checking split of data based on year

# In[ ]:


sns.countplot(data.arrival_date_year)


# Extending this to see the common portions. Only month that is common for all 3 years is July.

# In[ ]:


data_july = data.loc[data.arrival_date_month=='July']
plt.figure(figsize=(15,6))
sns.countplot(data_july.arrival_date_day_of_month, hue=data_july.arrival_date_year)
plt.xticks(rotation=90)


# Looks like an overall increase in the number of bookings year on year. Let's group by the year and make sure.

# Taking the 'class' column, here being hotel, we can see how there's been a steady year on year increase of:

# In[ ]:


pct_change = pd.DataFrame(data_july.groupby(['arrival_date_year'])['hotel'].count())
pct_change['pct_change'] = data_july.groupby(['arrival_date_year'])['hotel'].count().pct_change() * 100
pct_change


# A percentage increase of 64.7% between 2015 and 2016, with 16.2% seen between 2016 and 2017. Why such a large change? Let's see if we can find out why by looking at the places that people were visiting from.

# A better view of the split by country

# In[ ]:


fig = plt.figure(figsize=(20,10))
plt.pie(data['country'].value_counts(), labels=data['country'].value_counts().index)
fig.set_facecolor('lightgrey')
plt.show()


# ## Grouping by country and hotel type

# In[ ]:


df_bycountry = data_july.groupby(['country', 'arrival_date_year']).size().reset_index(name='counts')
plt.figure(figsize=(20,5))
sns.barplot(data=df_bycountry, x='country', y='counts', hue='arrival_date_year')
plt.ylabel('Count')
plt.xlabel('Country Code')
plt.xticks(rotation=90)


# Interestingly, there is an increase in visitors per country across the board. That is, apart from Portugal (PRT), which has seen quite worrying declining numbers. Why?

# Let's try doing the same thing, but this time taking into consideration the different hotel types (City Hotel and Resort Hotel).

# In[ ]:


int_df = data_july.loc[data_july.country.isin(['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE'])]
df_by_country_hotel = int_df.groupby(['country', 'arrival_date_year', 'hotel']).size().reset_index(name='counts')


# In[ ]:


plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
ax1 = sns.barplot(data=df_by_country_hotel.loc[df_by_country_hotel.hotel=='Resort Hotel'], x='country', y='counts', hue='arrival_date_year')
ax1.set_title('Resort Hotel')
plt.subplot(2,1,2)
ax2 = sns.barplot(data=df_by_country_hotel.loc[df_by_country_hotel.hotel=='City Hotel'], x='country', y='counts', hue='arrival_date_year')
ax2.set_title('City Hotel')


# No surprises there, Portugal having the largest portion of visitors to the hotel. Interestingly, the decrease in Portugal's visitors can mostly be seen in the visitors that stayed at the city hotel, with the resort hotel's visitors remaining somewhat constant for PRT while increasing steadily elsewhere.

# ## Plotting on a week by week basis

# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(data.arrival_date_week_number, hue=data.arrival_date_year)
plt.xticks(rotation=90)


# ## Looking at lead time

# In[ ]:


data['lead_time'].describe()


# on average, 104 days pass between the booking and the client checking into the hotel

# In[ ]:


sns.distplot(data['lead_time'])


# But the majority (~50%) are booked within 69 days from the travel date. Is there a relationship between the country of origin and the amount of lead time. 

# In[ ]:


data_time = data[['country', 'lead_time']]


# Take 2 countries, 1 being far away from the assumed location of the hotel (USA) with the other being somewhere closeby. In this case NLD was chosen since it has a comparable number of occurrances in the dataset w.r.t. USA.

# In[ ]:


data_nld = data_time.loc[data_time.country=='NLD']
data_usa = data_time.loc[data_time.country=='USA']
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
ax1=sns.distplot(data_nld['lead_time'])
ax1.set_title('lead time distribution - NLD')
plt.subplot(1,2,2)
ax2=sns.distplot(data_usa['lead_time'])
ax2.set_title('lead time distribution - USA')
plt.show()


# They have comparable distributions. Maybe it's because a direct flight can be booked (and it's easier)(assumption) from the US to the hotel location. Trying the same thing now comparing a country farther away, such as China and another european country, austria (picked due to nearly same number of occurrances).

# In[ ]:


data_cn = data_time.loc[data_time.country=='CN']
data_aut = data_time.loc[data_time.country=='AUT']
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
ax1=sns.distplot(data_cn['lead_time'])
ax1.set_title('lead time distribution - CN')
plt.subplot(1,2,2)
ax2=sns.distplot(data_usa['lead_time'])
ax2.set_title('lead time distribution - AUT')
plt.show()


# It seems like no matter the distance, people usually book around 3 months prior to a holiday. That goes to show how easy travel has become nowadays.

# In[ ]:


labels = ['Resort Hotel Lead Time', 'City Hotel Lead Time']
plt.figure()
sns.kdeplot(data.loc[data.hotel=='Resort Hotel', 'lead_time'], shade=True)
sns.kdeplot(data.loc[data.hotel=='City Hotel', 'lead_time'], shade=True)
plt.legend(labels)


# From the plot and stats, a customer is more likely to book the resort hotel closer to the holiday date than the city hotel (20 day difference on average). This could be for a number of reasons. The first thing that comes to mind is that if a customer wants to go to a resort, usually (assumption), they will stay within the resort and enjoy the holiday there. On the other hand, a city hotel stay usually involves a greater level of planning due to the customer wanting to see the surrounding area (city).

# Now lets look at the same type of plot but this time for the total number of nights spent on holiday

# In[ ]:


data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']


# In[ ]:


data.groupby(['total_nights', 'hotel']).size().reset_index(name='counts')


# In[ ]:


data_tn = data.groupby(['total_nights', 'hotel']).size().reset_index(name='counts')
plt.figure(figsize=(15,5))
sns.barplot(x='total_nights', y='counts', hue='hotel', data=data_tn)
plt.xticks(rotation=90)
plt.show()


# Majority of bookings for the city hotel are short stays, 4-5 nights. The resort hotel also sees a majority of short stays, but more people opt to go for a whole weeks worth of nights. 

# ## Taking a full year

# In[ ]:


data_yr = data.loc[data.arrival_date_year==2016]


# In[ ]:


months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
data_yr_grp = data_yr.groupby(['arrival_date_month', 'hotel']).size().reset_index(name='counts')
plt.figure(figsize=(15,5))
sns.barplot(x='arrival_date_month', y='counts', hue='hotel', data=data_yr_grp, order=months)


# ## The effect of Families on the hotel choice

# In[ ]:


data['kids'] = data.children + data.babies


# In[ ]:


data_kids = data.loc[data.kids>0]
ct1 = pd.crosstab(data_kids.kids, data_kids.hotel).apply(lambda x: x/x.sum(), axis=0)
data_babies = data.loc[data.babies>0]
ct2 = pd.crosstab(data_babies.kids, data_babies.hotel).apply(lambda x: x/x.sum(), axis=0)
ct1.plot.bar()


# In[ ]:


ct2.plot.bar()


# It seems like having kids with you doesn't lead you to pick one hotel over another.

# # Meal insight

# Meal value definitions:
# * BB - bed & breakfast
# * HB - half board - breakfast and one other meal, usually dinner
# * FB - beakfast, lunch and dinner
# * Undefined/SC - no meal package

# In[ ]:


data.loc[data.meal=='Undefined', 'meal'] = 'SC'


# In[ ]:


# taking top 10 seen countries
data_meal_tc = data.loc[data.country.isin(['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE'])].groupby(['country', 'meal']).size().reset_index(name='counts')
plt.figure(figsize=(20,5))
sns.barplot(data=data_meal_tc, x='country', y='counts', hue='meal')


# In[ ]:


perc = data.loc[data.country.isin(['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE'])].groupby(['country', 'meal']).size()
percbycountry = perc.groupby(level=0).apply(lambda x: 100 * x/float(x.sum())).reset_index(name='percgp')


# Plotting previous as percentage of group totals

# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(data=percbycountry, x='country', y='percgp', hue='meal')


# It seems like there is no large difference between the top 10 countries in terms of the type of meal chosen. Not splitting according to the chosen hotel to see if that has an affect.

# In[ ]:


perc = data.loc[data.country.isin(['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE'])].groupby(['country', 'hotel']).size()
percbycountry = perc.groupby(level=0).apply(lambda x: 100 * x/float(x.sum())).reset_index(name='percgp')


# In[ ]:


# looking at percentage from top 10 countries going to which hotel
plt.figure(figsize=(20,5))
sns.barplot(data=percbycountry, x='country', y='percgp', hue='hotel')


# In[ ]:


countries = ['PRT', 'GBR', 'ESP', 'DEU', 'FRA', 'BEL', 'IRL', 'ITA', 'USA', 'CHE']
x = 1
plt.figure(figsize=(20,10))
for country in countries:
    temp_df = data.loc[data.country==country].groupby(['hotel', 'meal']).size()
    perc = temp_df.groupby(level=0).apply(lambda x: x/float(x.sum()) * 100).reset_index(name='percgrp')
    plt.subplot(2, 5, x)
    ax = sns.barplot(data=perc, x='hotel', y='percgrp', hue='meal')
    ax.set_title(country)
    x+=1
plt.show()


# Straight away we can see how the majority of clients pick the bed and breakfast option. Interestingly, people originating from Germany staying at the Resort hotel go for the half board option almost as much as bed and breakfast.

# ## Market Segment & Distribution Channel

# TBC
