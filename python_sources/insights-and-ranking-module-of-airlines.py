#!/usr/bin/env python
# coding: utf-8

# ## **Data Exploration**

# In[ ]:


import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


airlines = pd.read_csv('../input/airlines.csv')
airports = pd.read_csv('../input/airports.csv')
flights = pd.read_csv('../input/flights.csv')


# In[ ]:


#Airlines
airlines.shape[0] #no of rows=14
airlines.shape[1] #no of columns=2
airlines.head(3)


# In[ ]:


#Airports
airports.shape[0] #no of rows=322
airports.shape[1] #no of columns=7
airports.head(3)


# In[ ]:


flights.shape[0] #no of rows = 5819079
flights.shape[1] #no of columns = 31
flights.head()


# ## Some important data definitions and relationships are mentioned below.
# ### Data Definition
# *  AIR_TIME - The time duration between wheels_off and wheels_on time.
# *  WHEELS_OFF Time - The time point that the aircraft's wheels leave the ground.
# *  WHEELS_ON Time - The time point that the aircraft's wheels touch on the ground.
# *  TAXI_OUT Time - The time duration elapsed between departure from the origin airport gate and wheels off.
# *  TAXI_IN Time - The time duration elapsed between wheels-on and gate arrival at the destination airport.
# 
# ### Data Relationship
# *  departure_time = wheels_off - taxi_out
# *  departure_delay = departure_time - scheduled_departure
# *  arrival_time = wheels_on + taxi_in
# *  arrival_delay = arrival_time - scheduled_arrival
# *  elapsed_time =air_time + taxi_in + taxi_out
# *  air_time = wheels_on - wheels_off
# 
# ### We can check the relationships from the following table.

# In[ ]:


flights[['ELAPSED_TIME','TAXI_IN','AIR_TIME','TAXI_OUT','ARRIVAL_TIME','WHEELS_ON','WHEELS_OFF','ARRIVAL_DELAY','DEPARTURE_TIME','DEPARTURE_DELAY','SCHEDULED_TIME','SCHEDULED_ARRIVAL','SCHEDULED_DEPARTURE']][0:5]


# ### We can also note the following from the above table.
# ##### The following times are in the xx:yy - hour:minute format (e.g. 2354 means 11:54pm, 5 means 00:05 am)
# scheduled_departure,   departure_time,    scheduled_arrival,    arrival_time,    wheels_off,    wheels_on
# ##### And the following times are in minutes format (negatives mean actual_time is ahead of scheduled_time for the absolute value of that negative number)
# arrival_delay,    departure_delay,    taxi_in,    taxi_out,    scheduled_time,    elapsed_time,    air_time
# ### Now,let's add some features of our own.

# In[ ]:


flights.YEAR.unique() #2015
day_of_week_desc={
    7:'Monday',
    1:'Tuesday',
    2:'Wednesday',
    3:'Thursday',
    4:'Friday',
    5:'Saturday',
    6:'Sunday'
}
flights['DESC_DOW']=flights['DAY_OF_WEEK'].apply(lambda a:day_of_week_desc[a])
flights.head()['DESC_DOW']


# In[ ]:


airlines_dict = dict(zip(airlines['IATA_CODE'],airlines['AIRLINE']))
airport_dict = dict(zip(airports['IATA_CODE'],airports['AIRPORT']))
flights['DESC_AIRLINE'] = flights['AIRLINE'].apply(lambda x: airlines_dict[x])
flights.head()['DESC_AIRLINE']


# A lot of airlines activity is given and the first question that pops up in our head is 
# 'Which airlines is the best?'
# So,let's create a ranking module for airlines.We will store the relevant information throughout the process.
# Following are the factors that decide the rank of an airline.
# 
# *  Highest Ratio of (Operated flights)/(Scheduled flights)
# *  Flight speed
# * Average arrival delay
# * Flight volume
# * Taxi In and Out Time
# 
# I have not included avg. departure delay because usually it depends on the departure airport.
# ###### Let's calculate the ratio of operated flights/scheduled flights for each airlines.

# In[ ]:


# flights.CANCELLED.unique() #0,1
#Each airline is either cancelled or operated.
rank_airlines = pd.DataFrame(flights.groupby('DESC_AIRLINE').count()['SCHEDULED_DEPARTURE'])
rank_airlines['CANCELLED']=flights.groupby('DESC_AIRLINE').sum()['CANCELLED']
rank_airlines['OPERATED']=rank_airlines['SCHEDULED_DEPARTURE']-rank_airlines['CANCELLED']
rank_airlines['RATIO_OP_SCH']=rank_airlines['OPERATED']/rank_airlines['SCHEDULED_DEPARTURE']
rank_airlines.drop(rank_airlines.columns[[0,1,2]],axis=1,inplace=True)
rank_airlines.head()


# In[ ]:


rank_airlines.sort(['RATIO_OP_SCH'],ascending = 1,inplace=True)
# rank_airlines.head()
rank_airlines['RATIO_OP_SCH'].plot(kind='bar',figsize=(12,6),rot=45)
plt.title('Ratio of operated and scheduled flights for each airlines in increasing order')


# In[ ]:


flights['FLIGHT_SPEED'] = 60*flights['DISTANCE']/flights['AIR_TIME']
rank_airlines['FLIGHT_SPEED'] = flights.groupby('DESC_AIRLINE')['FLIGHT_SPEED'].mean()
flights[['DESC_AIRLINE','FLIGHT_SPEED']].boxplot(column = 'FLIGHT_SPEED',by='DESC_AIRLINE',figsize=(12,7),rot=45)


# The plot clearly shows that almost all the flights of any airline run at the speed of 350-450 miles/hour.
# We have also added this information in our ranking module dataframe for future reference.

# In[ ]:


rank_airlines.head()


# Now, let's find the average delay for a particular 
# 
# Both arrival delay and departure delay are in minutes.

# In[ ]:


flights.groupby('DESC_AIRLINE')[['ARRIVAL_DELAY','DEPARTURE_DELAY']].mean()
#Let's add arrival delay to our ranking module as well.
rank_airlines['ARRIVAL_DELAY']= flights.groupby('DESC_AIRLINE')['ARRIVAL_DELAY'].mean()
#As our flight speed is in miles/hour,it's probably best to keep ARRIVAL DELAY in hours.
rank_airlines['ARRIVAL_DELAY']=rank_airlines['ARRIVAL_DELAY'].apply(lambda x:x/60)
rank_airlines.head()


# #### Meanwhile, let's take a look at the plot of arrival and departure delay for better understanding about airlines and airports.

# In[ ]:


df_delay = pd.DataFrame(flights.groupby('DESC_AIRLINE')[['ARRIVAL_DELAY','DEPARTURE_DELAY']].mean())
df_delay.sort(['ARRIVAL_DELAY','DEPARTURE_DELAY'],ascending = [1,1],inplace=True)
plt.figure(figsize=(10,7))
sns.set_color_codes("deep")
sns.set_context(font_scale=2.5)
plot = sns.barplot(x='DEPARTURE_DELAY',y=df_delay.index,data = df_delay,color = 'y')
plot = sns.barplot(x='ARRIVAL_DELAY',y=df_delay.index,data = df_delay,color = 'g')
plot.set(xlabel='Mean flight delays (Arrival : Green,Departure : Yellow)')


# As we can see, almost all the airlines have arrival delays greater than departure delays, which is logical as the departure delays are mostly due to late arrival, security reasons etc. Departure delays mostly depend upon the airport. We can keep this in mind while creating the module for ranking of airports.
# 
# One **important** thing to note is that Alaska Airlines has a negative arrival delay which means it arrives before scheduled time on an average.
# 
# Now,let's look at the flight volume of each airline.

# In[ ]:


rank_airlines['FLIGHTS_VOLUME'] = flights.groupby('DESC_AIRLINE')['FLIGHT_NUMBER'].count()
#Let's change it into ratio of flight_vol/total flight_vol
total = rank_airlines['FLIGHTS_VOLUME'].sum()
rank_airlines['FLIGHTS_VOLUME'] = rank_airlines['FLIGHTS_VOLUME'].apply(lambda x:(x/float(total)))
rank_airlines['FLIGHTS_VOLUME'].plot.pie(figsize=(10,10),rot=45)


# Now,let us consider taxi in and out time as well.
# 
# **TAXI_OUT time** - The time duration elapsed between departure from the origin airport gate and wheels off.
# 
# **TAXI_IN time** - The time duration elapsed between wheels-on and gate arrival at the destination airport.

# In[ ]:


rank_airlines[['TAXI_IN','TAXI_OUT']] = flights.groupby('DESC_AIRLINE')[['TAXI_IN','TAXI_OUT']].mean()
#Taxi in and out time are in minutes.Let's change them to hours.
# rank_airlines[['TAXI_IN','TAXI_OUT']] = rank_airlines[['TAXI_IN','TAXI_OUT']].apply(lambda x, y : (x/float(60),y/float(60)))
rank_airlines['TAXI_IN'] = rank_airlines['TAXI_IN'].apply(lambda x:(x/float(60)))
rank_airlines['TAXI_OUT'] = rank_airlines['TAXI_OUT'].apply(lambda x:(x/float(60)))
plt.figure(figsize=(11, 8))
sns.set_color_codes("deep")
sns.set_context(font_scale=2.5)
plot = sns.barplot(x='TAXI_OUT',y=rank_airlines.index,data = rank_airlines,color = 'crimson')
plot = sns.barplot(x='TAXI_IN',y=rank_airlines.index,data = rank_airlines,color = 'cyan')
plot.set(xlabel='Mean taxi out and in time (Taxi Out : Green,Taxi In : Yellow)')


# Now,let's find the rank of a particular airline.
# 
# We have 5 variables which decide a score.
# The score is proportional to a subset (a) of the variables whereas being inversely proportional to a different subset (b) of the variables.
# 
# The most naive way to capture this information is through the following formula.<br>
# 
# **Score_airline = a/(1+b)**, where
# 
#         a = (RATIO_OP_SCH) \* (FLIGHT_SPEED) \* (FLIGHTS_VOLUME)
# 
#         and 
# 
#         b = (ARRIVAL_DELAY) \* (TAXI_IN) \* (TAXI_OUT)
#         
# A higher score indicates a better rank.

# In[ ]:


# I have scaled the data to 1-2
for i in rank_airlines.columns:
    rank_airlines[i] = ((rank_airlines[i]-rank_airlines[i].min())/(rank_airlines[i].max()-rank_airlines[i].min()))+1
a = rank_airlines.RATIO_OP_SCH*rank_airlines.FLIGHT_SPEED*rank_airlines.FLIGHTS_VOLUME
b = rank_airlines.ARRIVAL_DELAY*rank_airlines.TAXI_IN*rank_airlines.TAXI_OUT
rank_airlines['SCORE'] = a/(1+b)
rank_airlines.sort(['SCORE'],ascending=False,inplace=True)
rank_airlines.head()


# In[ ]:


rank_airlines['SCORE'].plot.bar(figsize = (11,8),rot=55)


# In our earlier plots, we noticed the highest flight volume for Southwest Airlines and negative arrival delay for Alaska Airlines. Hence, it's no surprise that these two airlines grab the top spots. 

# # Let's look at some insights from the data.
# ## Most busy day.

# In[ ]:


df_busyday = pd.DataFrame(flights.groupby('DESC_DOW').count()['SCHEDULED_DEPARTURE'])
df_busyday = df_busyday.sort(['SCHEDULED_DEPARTURE'],ascending = 1)
df_busyday.head()
df_busyday.plot(kind='line',subplots=True,c='r',figsize=(12,6),legend=True)
plt.title('Number of Scheduled flights per day')


# ### Daily flight volume of each airline

# In[ ]:


flight_volume_airline_day = flights.pivot_table(index="DESC_DOW",columns="DESC_AIRLINE",values="SCHEDULED_DEPARTURE",aggfunc=lambda x:x.count())
#flight_volume_airline_day.head()
fig = plt.figure(figsize=(12,10))
sns.heatmap(flight_volume_airline_day, linecolor="w", linewidths=1)
plt.xticks(rotation=45)


# ### Trend of flight Cancellations

# In[ ]:


df_cancellations = pd.DataFrame(flights.groupby('DESC_DOW').sum()['CANCELLED'])
df_cancellations=df_cancellations.sort(['CANCELLED'],ascending =1)
df_cancellations.head()
df_cancellations.plot(kind='line',figsize=(12,6),subplots=True,legend=True)
plt.title('Number of cancellations each day')


# In[ ]:


flights.CANCELLATION_REASON.unique()


# 'CANCELLATION_REASON' indicates with a letter the reason for the cancellation of the flight.
# 
#  A - Carrier; B - Weather; C - National Air System; D - Security

# In[ ]:


cancellation_reason = pd.DataFrame(flights.groupby(['DESC_AIRLINE'])['AIR_SYSTEM_DELAY', 'AIRLINE_DELAY',
                                               'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'].mean())
# flight_volume_airline_day = flights.pivot_table(index="DESC_DOW",columns="DESC_AIRLINE",values="SCHEDULED_DEPARTURE",aggfunc=lambda x:x.count())
cancellation_reason.head()
cancellation_reason.plot.bar(legend = True,figsize = (12,11),rot=55)
plt.legend(loc=2,prop={'size':13})
plt.tick_params(labelsize = 13)


# That's all for now. I will keep adding more insights.
# 
# I appreciate your suggestions. Please leave them in the comment box.

# In[ ]:




