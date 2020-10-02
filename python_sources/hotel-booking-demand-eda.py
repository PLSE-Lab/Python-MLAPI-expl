#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df


# In[ ]:


df.info()


# In[ ]:


# Converting reservation_status_date to datetime object
df['reservation_status_date'] = df['reservation_status_date'].astype('datetime64[ns]')


# In[ ]:


df.isnull().sum()


# In[ ]:


df.deposit_type.value_counts()


# In[ ]:


df.is_canceled.value_counts()


# In[ ]:


df.reservation_status.value_counts()


# Separate the booking that were canceled and not canceled

# In[ ]:


a = df[df['is_canceled']==0]


# In[ ]:


b = df[df['is_canceled']==1]


# droping the duplicates

# In[ ]:


a = a.drop_duplicates()


# In[ ]:


b = b.drop_duplicates()


# In[ ]:


a.shape


# Checking for the stay at weekday nights at city hotel and resort hotel

# In[ ]:


pd.crosstab(a['stays_in_week_nights'],a['hotel'])


# Checking for the stay at weekend nights at city hotel and resort hotel

# In[ ]:


pd.crosstab(a['stays_in_weekend_nights'],a['hotel'])


# count of members checked out from the two hotels

# In[ ]:


pd.crosstab(a['hotel'],a['reservation_status'])


# count of members canceled and no-show in those two hotels

# In[ ]:


pd.crosstab(b['hotel'],b['reservation_status'])


# In[ ]:


pd.crosstab(a['hotel'],a['deposit_type'])


# In[ ]:


pd.crosstab(b['hotel'],b['deposit_type'])


# In[ ]:


a


# In[ ]:


sns.catplot(x = 'hotel',data=a,kind='count')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.catplot(x = 'arrival_date_month',data=a,kind='count',hue='hotel')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.catplot(x = 'customer_type',data=a,kind='count',hue='hotel')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.distplot(a['lead_time'])
plt.show()


# In[ ]:


sns.catplot(x = 'market_segment',data=a,kind='count',hue='hotel')
plt.xticks(rotation=90)
plt.show()


# Checking whether the guest repeated or not

# In[ ]:


sns.catplot(x = 'is_repeated_guest',data=a,kind='count',hue='hotel')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.catplot(x = 'distribution_channel',data=a,kind='count',hue='hotel')
plt.xticks(rotation=90)
plt.show()


# installing plotly to display maps

# In[ ]:


get_ipython().system('pip install plotly')


# booking count at different countries

# In[ ]:


country_count = df['country'].value_counts()
country_count_df = pd.DataFrame(country_count)
country_count_df = country_count_df.reset_index()
country_count_df.columns = ['country','booking_count']
country_count_df = country_count_df[country_count_df['booking_count'] > 0]

import plotly.express as px

fig = px.choropleth(country_count_df, locations="country",
                    color="booking_count",
                    hover_name="country")
fig.show()


# In[ ]:




