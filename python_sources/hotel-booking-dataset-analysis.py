#!/usr/bin/env python
# coding: utf-8

# **Introduction:** 
# This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.
# 
# **Goal:** The goal of this notebook is to practice EDA and figure out the standard patterns of booking.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


#visualize missing data
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df.isnull(),cmap='Blues', yticklabels=False, cbar=False)
plt.show()


# We are working with a lot of data in this data set. In order to clean the data, we will want to remove NaN columns. Typically we may remove rows containing NaN values if we have a large enough dataset. For this dataset, company and agent may not be important columns of data for analysis on booking trends. For that reason, we can omit these columns from our dataframe.

# In[ ]:


df = df.drop(columns = ['agent', 'company'])


# With the agent and company columns drop, we will need to clean up the remaining NaN country rows. There are only 488 rows out of the 119,390 rows in this data set containing NaN country values. Removing these rows will not have a significant effect on our analysis.

# In[ ]:


#drops rows containing missing data in 'country' column
df = df.dropna(axis=0)


# In[ ]:


df.isnull().sum()


# In[ ]:


#Overview of type of hotel

#Enlarging pie chart
plt.rcParams['figure.figsize'] = 9,9

#Indexing labels
labels = df['hotel'].value_counts().index.tolist()

#Convert value counts to list
sizes = df['hotel'].value_counts().tolist()

#Explode to determine how much each section is separated from each other
explode = (0,0.075)

#Coloring pie chart
colors = ['0.75', 'maroon']

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})


# From the pie chart above we can see that two thirds of bookers chose the city hotel option.

# In[ ]:


df.head()


# In[ ]:


#Grouping by adults to get summary statistics on hotel type
df['adults'].groupby(df['hotel']).describe()


# In[ ]:


#Grouping by children to get summary statistics on hotel type
df['children'].groupby(df['hotel']).describe()


# The data above shows us that on average, more children are booked into resort hotels.

# In[ ]:


#analyzing canceled bookings data
df['is_canceled'] = df.is_canceled.replace([1,0], ['canceled', 'not_canceled'])
canceled_data = df['is_canceled']
sns.countplot(canceled_data)


# Using a countplot, we were able to graph the total amout of canceled vs non-canceled data. It appears the majority of bookings were not_canceled.

# In[ ]:


#Analyzing cancellation rate amongst hotel types
lst1 = ['is_canceled', 'hotel']
type_of_hotel_canceled = df[lst1]
canceled_hotel = type_of_hotel_canceled[type_of_hotel_canceled['is_canceled'] == 'canceled'].groupby(['hotel']).size().reset_index(name='count')
sns.barplot(data = canceled_hotel, x = 'hotel', y = 'count').set_title('Graph depicting cancellation rates in city and resort hotel')


# We see a large number of cancelations from city hotels. Keep in mind most bookings were at city hotels.

# In[ ]:


#Graph arrival year
lst3 = ['hotel', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
period_arrival = df[lst3]

sns.countplot(data = period_arrival, x = 'arrival_date_year', hue = 'hotel').set_title('Graph showing number of arrivals per year')


# In the plot above we se that city hotels had the most bookings consistently each year with the largest amount of bookings in 2016.

# In[ ]:


#Graph arrival month
plt.figure(figsize=(20,5)) # adjust the size of the plot

sns.countplot(data = period_arrival, x = 'arrival_date_month', hue = 'hotel', order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']).set_title('Graph showing number of arrivals per month',fontsize=20)
plt.xlabel('Month') # Creating label for xaxis
plt.ylabel('Count') # Creating label for yaxis


# In[ ]:


#Graph arrival dates
plt.figure(figsize=(15,5))

sns.countplot(data = period_arrival, x = 'arrival_date_day_of_month', hue = 'hotel').set_title('Graph showing number of arrivals per day', fontsize = 20)


# Booking rates were highest during the year of 2016. Additionally, the trend shows that bookings occurs at the highest rate around the middle of year, with August being the highest. Data shows that summer is a peak season for hotel booking.
# There is a wave like structure to arrivals by day. My speculation is that these peaks depict hotel bookings on the weekends at a higher rate.

# In[ ]:


#Graphing weekend vs. weekday data
sns.countplot(data = df, x = 'stays_in_weekend_nights').set_title('Number of stays on weekend nights', fontsize = 20)


# In[ ]:


sns.countplot(data = df, x = 'stays_in_week_nights' ).set_title('Number of stays on weekday night' , fontsize = 20)


# Our hypothesis was proven false as the majority of stays were on weekday nights.

# In[ ]:


#Graphing data by types of visitors
sns.countplot(data = df, x = 'adults', hue = 'hotel').set_title("Number of adults", fontsize = 20)


# In[ ]:


sns.countplot(data = df, x = 'children', hue = 'hotel').set_title("Number of children", fontsize = 20)


# In[ ]:


sns.countplot(data = df, x = 'babies', hue = 'hotel').set_title("Number of babies", fontsize = 20)


# The data shows us that travelers tend to book hotels in pairs and that those traveling with a baby prefer to book a resort hotel.

# In[ ]:


#Graphing booking data by country of origin
country_visitors = df[df['is_canceled'] == 'not_canceled'].groupby(['country']).size().reset_index(name = 'count')

# We will be using Plotly.express to plot a choropleth map. Big fan of Plotly here!
import plotly.express as px

px.choropleth(country_visitors,
                    locations = "country",
                    color= "count", 
                    hover_name= "country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Home country of visitors")


# From the plot we can see a large number of hotel bookers are from the UK, with the highest numbers of bookings originating from France and Portugal.

# In[ ]:


#graphing deposit types
sns.countplot(data = df, x = 'deposit_type').set_title('Graph showing types of deposits', fontsize = 20)


# Majority of bookings did not require a deposit, this could explain the high cancelation rate.

# In[ ]:


#graph repeated guests
sns.countplot(data = df, x = 'is_repeated_guest').set_title('Graph showing whether guest is repeated guest', fontsize = 20)


# Data shows a low number of repeated guests. In business, it is far more expensive to gain a new customer than to retain an exisiting one. Further business or marketing efforts could be drafted to improve rate of return.

# In[ ]:


#graph types of guests
sns.countplot(data = df, x = 'customer_type').set_title('Graph showing type of guest', fontsize = 20)


# The majority of bookings are transient. This is defined as a booking that is not a part of a group or contract. Booking online independtly is becoming increasingly consumer friendly which could explain this data.

# In[ ]:


#graphing prices per month per hotel
#average daily rate = (sumOfAllLodgingTransaction/TotalNumberOfStayingNight)
#average daily rate per person = (ADR/Adults+Children)

# Resizing plot 
plt.figure(figsize=(12,5))

# Calculating average daily rate per person
df['adr_pp'] = df['adr'] / (df['adults'] + df['children']) 
actual_guests = df.loc[df["is_canceled"] == 'not_canceled']
actual_guests['price'] = actual_guests['adr'] * (actual_guests['stays_in_weekend_nights'] + actual_guests['stays_in_week_nights'])
sns.lineplot(data = actual_guests, x = 'arrival_date_month', y = 'price', hue = 'hotel')


# Prices of the resort hotel are typically higher than the city hotel, with highest rates during busy months of travel in the summer, ie. August, June, and July.

# # Summary:
# 1. City hotels account for 2/3s of all bookings, Resort hotels account for 1/3.
# 2. About 50% of all bookings are cancelled.
# 3. 2016 showed the highest rate of hotel bookings. (data from 2015-2017)
# 4. Highest daily rates occurred in the summer (June, July, August)
# 5. More bookings occured on weekdays vs weekends.
# 6. Bookers with children or babies booked at resorts at a higher rate.
# 7. UK, France, and Portugal booked the most hotel stays worldwide.
# 8. Majority of hotel bookings did not require a deposit.
# 9. Most bookings came from independent, transient customers.
# 10. >90% of bookings were not return guests.
# 
# # Discussion:
# Analysis of this dataset highlights a few key take-aways. First, there is a disproportianate amount of cancellations on hotel bookings (~50%). Bookers are not required to send in a deposit in most bookings which could explain the high rate of cancellations. There are more bookings on weekdays, this data suggests that hotel business could be improved by targeting working travelers or improving daily rates for weekdays. Although resorts only account for 1/3 of all bookings, they out-compete city hotels when there is a baby in the booking group. Resort hotels could target young families with this knowledge by appealing to customers looking for a safe, family atmosphere. Europe has a high amount of travelers that book both resort and city hotels. European business typically has more holiday vacation time for workers and aggressive marketing campaigns could be made in this region to corner this market. Resort hotels have a steep rise in price during busy summer seasons, knowing this information city hotels would potentially raise their daily rate (So long as they are not competing against each other). Lastly, most bookings came from independent travelers and there is a significantly low rate of return (<10%). Hotels should take efforts to improve relationships with customers or target past guests for return stays.
# 
