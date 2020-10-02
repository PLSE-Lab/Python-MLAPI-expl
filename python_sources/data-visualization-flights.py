#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime, warnings, scipy 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# ## Reading the data: 

# In[ ]:


df_flights = pd.read_csv('../input/flight-delays/flights.csv', low_memory=False)
df_airports=pd.read_csv('../input/flight-delays/airports.csv', low_memory=False)
df_airlines=pd.read_csv('../input/flight-delays/airlines.csv', low_memory=False)
print('flights data dimensions:', df_flights.shape)
print('airports data dimensions:', df_airports.shape)
print('airlines data dimensions:', df_airlines.shape)

#some information about


# We have 5 819 079 flights recoded during the year 2015. 
# 

# In[ ]:


df_flights.head()


# Each flights described by some features : 
# * **YEAR, MONTH, DAY, DAY_OF_WEEK:** dates of the flight 
# * **AIRLINE:** An identification number assigned by US DOT to identify a unique airline 
# * **ORIGIN_AIRPORT and DESTINATION_AIRPORT:** code attributed by IATA to identify the airports 
# * **SCHEDULED_DEPARTURE and SCHEDULED_ARRIVAL :** scheduled times of take-off and landing
# * **DEPARTURE_TIME and ARRIVAL_TIME:** real times at which take-off and landing took place 
# * **DEPARTURE_DELAY and ARRIVAL_DELAY:** difference (in minutes) between planned and real times 
# * **DISTANCE:** distance (in miles) **

# In[ ]:


# display the percentage of the null value for each column (feature)
df_flights.isnull().sum()/len(df_flights)


# In[ ]:


df_airports.info()


# The airports data is a detailed description about the airports. As we can see from the above information that just we miss some Latitude and longitude values.

# In[ ]:


df_airlines.head()


# ## Data exploration:
# 

# In data exploration part, i want to find the reasons of delays by answering the following questions. 
# * Basic queries:
#     * How many unique origin airports?
#     * How many unique destination airports?
#     * How many flights that have a scheduled departure time later than 18h00?
# * Statistics on flight volume: this kind of statistics are helpful to reason about delays. Indeed, it is plausible to assume that "the more flights in an airport, the higher the probability of delay".
#     * How many flights in each month of the year?
#     * Is there any relationship between the number of flights and the days of week?
#     * How many flights in different days of months and in different hours of days?
#     * Which are the top 20 busiest airports (this depends on inbound and outbound traffic)?
#     * Which are the top 20 busiest carriers?
# * Statistics on the fraction of delayed flights
#     * What is the percentage of delayed flights (over total flights) for different hours of the day?
#     * Which hours of the day are characterized by the longest flight delay?
#     * What are the fluctuation of the percentage of delayed flights over different time granularities?
#     * What is the percentage of delayed flights which depart from one of the top 20 busiest airports?
# 

# ### Basic queries:

# In[ ]:


#How many unique origin airports?
n_orig_arp=len(df_flights.ORIGIN_AIRPORT.unique())
#How many unique destination airports?
n_dest_arp=len(df_flights.DESTINATION_AIRPORT.unique())
print("Origin Airports: ", n_orig_arp)
print("Destination Airports: ", n_dest_arp)


# There is a discrepancy between the number of origin airports and the number of destination airports. We would expect those two values to be the same but they are not. This may be due to missing data.

# In[ ]:


#How many flights that have a scheduled departure time later than 18h00?
n_night_flight=len(df_flights.SCHEDULED_DEPARTURE[df_flights.SCHEDULED_DEPARTURE>1800])
print("Night Flights: ", n_night_flight)
print("Night Flights over Total: ", (n_night_flight/len(df_flights))*100, "%")


# ### Statistics on flight volume:

# In[ ]:


# How many flights in each month of the year?
import datetime as dt

months = []
for month in range(1, 13):
    months.append(dt.datetime(year=1994, month=month, day=1).strftime("%B"))

fl_per_month = list(df_flights.groupby('MONTH').count().YEAR
)
plt.xlabel('Month')
plt.ylabel('Night Flights')
plt.xticks(range(1,13), months, rotation='vertical')
plt.plot(range(1,13), np.array(fl_per_month), '.-')
plt.show()


# From the plot we can see that in February we have the lowest flight traffic amount, while in July we have the hightest one. In fact, another relevant information shown by the above plot is that in holiday periods (Summer holidays in the months of July and August and Christmas holidays in December) the flight amount is much bigger with respect to other months, especially those following a holiday period.

# In[ ]:


flights_dayofweek = (
    df_flights.groupby(df_flights.DAY_OF_WEEK)
    .count()
    ).YEAR


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#Global aggregates
days=['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
n_flight_week=list(df_flights.groupby('DAY_OF_WEEK').count().YEAR)
frequencies=[n_flight_week[i] for i in range(len(n_flight_week)) ]
plt.bar(range(0,7),frequencies)
plt.xlabel('Day of week ')
plt.ylabel('number of flights')
plt.xticks(range(0,7),days,rotation='vertical')
plt.show()


# In[ ]:



figure(num=None, figsize=(17, 6), dpi=80, facecolor='w', edgecolor='k');


n_flight_day_month=list(df_flights.groupby(['MONTH', 'DAY_OF_WEEK']).count().YEAR)
frequencies=[]
frequency=[]
for i in range(1,len(n_flight_day_month)+1):
    
    frequency.append(n_flight_day_month[i-1])
    if (i%7==0):
        frequencies.append(frequency)
        frequency=[]


# data to plot
n_groups = 12


colors = ['b',  'r', 'c',  'y', 'm','g','k']
# create plot
index = np.arange(0, n_groups * 5, 5)
bar_width = 0.55
opacity = 0.8

for i in range(7):
    plt.bar(index+bar_width*i,tuple([row[i] for row in frequencies]),align='edge',width=0.4,
    alpha=opacity,
    color=colors[i],
    label=days[i])

plt.xlabel('Day of week per month')
plt.ylabel('number of flights')
plt.xticks(index + bar_width, ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'),rotation='vertical')
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure(figsize=(17,5));
plt.show();


# In[ ]:


# use a heatmap to better visualize the data
pddf=pd.DataFrame({'count' : df_flights.groupby( ['MONTH', 'DAY_OF_WEEK'] ).size()}).reset_index()
pddf = pddf.pivot("MONTH", "DAY_OF_WEEK", "count")
fig, ax=plt.subplots(figsize=(10,8))
ax = sns.heatmap(pddf, 
                  linewidths=.5,
                  annot=True,
                  vmin=50000,
                  vmax=75000,
                  fmt='d',
                  cmap='Blues', ax=ax)
# set plot's labels
ax.set_xticklabels(days)
ax.set_yticklabels(list(reversed(months)), rotation=0)
ax.set(xlabel='Day of the week', ylabel='Month')
ax.set_title("Number of flights per day of the week for each month", fontweight="bold", size=15)

plt.show()


# The first plot shows that, over the year, the number of flights during the week days is generally stable while it decreases on Saturdays and Sundays.
# <br>
# <br>
# The second bar plot delves into the details of each month. The pattern seen in the first plot doesn't actually apply to the monthly aggregates. In fact it can be noticed that each month has its own peaks, for example, in January there are more flights during Fridays and over the weekends while in March they are in Mondays.
# <br>
# <br>
# Moreover, the heat map can offer a better data visualization to look for relevant information coming from the data. The map better shows the difference between week days and weekends, the highest number of flights are always during the week while the minimums are in the weekends, even if some weekends present increasing values in the months containing holidays like Christmas, Summer or the spring break.
# 

# In[ ]:


# How many flights in different days of months and in different hours of days?
# number of flights per day of the month
# create the pandas dataframe
pddf=pd.DataFrame({'count' : df_flights.groupby(df_flights.DAY).size()}).reset_index()

# plot the number of flights per day of the month
f, ax = plt.subplots(figsize=(18, 6))
sns.barplot(x="DAY",
            y="count",
            data=pddf,
            palette=sns.color_palette("GnBu_d", 31),
            ax=ax)

# set plot's labels
ax.set(xlabel='Day of the month', ylabel='Number of flights')
ax.set_title("Number of flights per day of the month during the year", fontweight="bold", size=15)

plt.plot()


# In[ ]:



pddf=pd.DataFrame({'count' : df_flights.groupby(['MONTH','DAY']).size()}).reset_index()

# use a heatmap to better visualize the data
pddf = pddf.pivot("MONTH", "DAY", "count")
f, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pddf,
            square=True,
            vmin=11400,
            vmax=15000,
            cmap='Blues')

# set plot's labels
ax.set_yticklabels(list(reversed(months)), rotation=0)
ax.set(xlabel='Day of the month', ylabel='Month')
ax.set_title("Number of flights per day of the month for each month", fontweight="bold", size=15)

plt.show()


# In[ ]:


# create the pandas dataframe
# number of flights per hour

pddf=pd.DataFrame({'count' : df_flights.groupby(((df_flights.SCHEDULED_DEPARTURE/100).astype(int))).size()}).reset_index()

# plot the number of flights per hour
f, ax = plt.subplots(figsize=(18, 6))
sns.barplot(x="SCHEDULED_DEPARTURE",
            y="count",
            data=pddf,
            palette=sns.color_palette("GnBu_d", 24),
            ax=ax)

# set plot's labels
ax.set(xlabel='Hour of the day', ylabel='Number of flights')
ax.set_title("Number of flights per hour during the year", fontweight="bold", size=15)

plt.plot()


# In[ ]:


df_flights['HOUR']=(df_flights.SCHEDULED_DEPARTURE/100).astype(int)


# In[ ]:


# create the pandas dataframe
# number of flights per hour per month

pddf=pd.DataFrame({'count' : df_flights.groupby(['MONTH','HOUR']).size()}).reset_index()

# use a heatmap to better visualize the data
pddf = pddf.pivot("MONTH", "HOUR", "count")
f, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pddf,
            square=True,
            cmap='Blues')

# set plot's labels
ax.set_yticklabels(list(reversed(months)), rotation=0)
ax.set(xlabel='Hour of the day', ylabel='Month')
ax.set_title("Number of flights per hour for each month", fontweight="bold", size=15)

plt.show() 


# The first plot shows a stable number of flights during the days of the month. This result is expected when we look at the previous plot because by plotting the number of flights per day during the year we loose information about the day of the week and, as the plots show, this information is more relevant than the actual day of the month. Furthermore, the number of flights decreases in the last days of the months, this is due to the fact that not all the months have the same number of days (as can be noticed from the white boxes in the next plot). 
# 
# The heat map shows the number of flights per day of the month for each month.
# 
# Finally, the second bar plot and heat map, clearly show how, regardless of the month, the number of flights drops down at night and it reaches is maximim values between 6 and 7 AM, with other two small peaks between 15 and 16, and 17 and 18. 
# 
# The last plots show also that February has a lower number of flights with respect to the other months, this information was already suggested by the previous plots showing the values per day of the week.

# In[ ]:


# Which are the **top 20** busiest airports? 
# number of outbound flights per airport
df_out_airport = pd.DataFrame({'count' : df_flights.groupby(df_flights.ORIGIN_AIRPORT).size()}).reset_index()
df_out_airport=df_out_airport.rename(columns={"ORIGIN_AIRPORT": "airport"})

# number of inbound flights per airport
df_in_airport = pd.DataFrame({'count' : df_flights.groupby(df_flights.DESTINATION_AIRPORT).size()}).reset_index()
df_in_airport=df_out_airport.rename(columns={"DESTINATION_AIRPORT": "airport"})
# number of flights per airport
df_airport=pd.DataFrame( pd.concat([df_out_airport,df_in_airport],ignore_index=True).groupby('airport').sum()).reset_index()
print("Top 20 busiest airports (outbound)")
print(df_out_airport.sort_values('count',ascending=False).head(20))

print("Top 20 busiest airports (inbound)")
print(df_in_airport.sort_values('count',ascending=False).head(20))

print("Top 20 busiest airports")
print(df_airport.sort_values('count',ascending=False).head(20))


# ### Statistics on the fraction of delayed flights:

# In this series of questions we focus on the computation of statistics about the percentage of delayed flights. 
# 
# A flight is considered as delayed if it's actual arrival time is more than 15 minutes later than the scheduled arrival time.

# In[ ]:


# What is the percentage of delayed flights for different hours of the day?
# number of delayed flights per hour
df_flights_hour_delay = pd.DataFrame({'count' : df_flights[df_flights.ARRIVAL_DELAY > 15].groupby('HOUR').size()}).reset_index()

# number of flights per hour
df_flights_hour = pd.DataFrame({'count' : df_flights.groupby('HOUR').size()}).reset_index()


# percentage of flight in delay per hour
df=df_flights_hour.join(df_flights_hour_delay,on='HOUR',rsuffix='_d', how='inner')
df['percentage']=df['count_d']*100/df['count']
percentage_hour_delay = df[['HOUR','percentage']]


# In[ ]:


# create the pandas dataframe
# plot the percentage of flights in delay per hour
f, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="HOUR",
            y="percentage",
            data=percentage_hour_delay,
            palette=sns.color_palette("GnBu_d", 24),
            ax=ax)

# set plot's labels
ax.set(xlabel='Hour of the day', ylabel='Percentage of delayed flights')
ax.set_title("Percentage of delayed flights for different hours of the day", fontweight="bold", size=15)

plt.plot()


# This plot shows that, even if the previous plot showed a very low number of flights at night, there a not-indifferent percentage of delayed flights in the late hours (up to 19%). In particular, it can easily be noticed that the percentage starts to increase at 5 AM and reaches its maximum values between 19 and 20 (more than one flight out of four (25%)), then it decreases again. To get a better understanding of the delays it would be useful to have statistics about the actual value of the delay (next plots).

# In[ ]:


df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
                    'B': [np.nan, 2, 3, 4, 5],
                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])


# In[ ]:


# create the pandas dataframe
# average delay per hour
hour_avg_delay = pd.DataFrame(data=df_flights.groupby('HOUR')['ARRIVAL_DELAY'].mean()).reset_index()

# plot the avg number of flights in delay per hour
f, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="HOUR",
            y="ARRIVAL_DELAY",
            data=hour_avg_delay,
            palette=sns.color_palette("GnBu_d", 24),
            ax=ax)

# set plot's labels
ax.set(xlabel='Hour of the day', ylabel='Average Delay')
ax.set_title("Average delay of the flights during the day", fontweight="bold", size=15)

plt.show()


# From the average number of delayed flights per hour it can be noticed that night hours have the smallest average delays and this value increases over time until it reaches it maximum at 19. This information is coherent with what we've already seen, infact we know that night hours have less flights and the shape of this curve to the one of the previous one so we can say that there could be a correlation between the percentage of delayed flights and the average value of the delay.
# 
# With data of year 2015, the flight from 5AM to 8AM often depart earlier than in their schedule. The flights in the morning have less delay then in the afternoon and evening.

# In[ ]:


import matplotlib.patches as mpatches
pdf_delay_ratio_per_hour = percentage_hour_delay
pdf_mean_delay_per_hour = hour_avg_delay
plt.xlabel("Hours")
plt.ylabel("Ratio of delay")
plt.title('The radio of delay over hours in day')
plt.grid(True,which="both",ls="-")
bars = plt.bar(pdf_delay_ratio_per_hour['HOUR'], pdf_delay_ratio_per_hour['percentage'], align='center', edgecolor = "black")
for i in range(0, len(bars)):
    color = 'red'
    if pdf_mean_delay_per_hour['ARRIVAL_DELAY'][i] < 0:
        color = 'lightgreen'
    elif pdf_mean_delay_per_hour['ARRIVAL_DELAY'][i] < 2:
        color = 'green'
    elif pdf_mean_delay_per_hour['ARRIVAL_DELAY'][i] < 4:
        color = 'yellow'
    elif pdf_mean_delay_per_hour['ARRIVAL_DELAY'][i] < 8:
        color = 'orange'

    bars[i].set_color(color)
        
patch1 = mpatches.Patch(color='lightgreen', label='Depart earlier')
patch2 = mpatches.Patch(color='green', label='delay < 2 minutes')
patch3 = mpatches.Patch(color='yellow', label='delay < 4 minutes')
patch4 = mpatches.Patch(color='orange', label='delay < 8 minutes')
patch5 = mpatches.Patch(color='red', label='delay >= 8 minutes')

plt.legend(handles=[patch1, patch2, patch3, patch4, patch5], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.margins(0.05, 0)
plt.show()


# In the new figure, we have more information in a single plot. The flights in 5AM to 8AM have very low probability of being delayed, and actually depart earlier than their schedule. In contrast, the flights in the 4PM to 9PM range have higher chances of being delayed: in more than 50% of the cases, the delay is 8 minutes or more.

# In[ ]:


df_flights.columns


# In[ ]:


# create the pandas dataframe
df_daymonth_delayed =  pd.DataFrame({'count' : df_flights[df_flights.ARRIVAL_DELAY > 15].groupby('DAY').size()}).reset_index()
df_daymonth= pd.DataFrame({'count' : df_flights.groupby('DAY').size()}).reset_index()

df_daymonth_delayed['percentage']=df_daymonth_delayed['count']*100/df_daymonth['count']


# plot the number of delayed flights per day of the month
f, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="DAY",
            y="percentage",
            data=df_daymonth_delayed,
            palette=sns.color_palette("GnBu_d", 31),
            ax=ax)

# set plot's labels
ax.set(xlabel='Day of the month', ylabel='Percentage of delayed flight')
ax.set_title("Percentage of delayed flights over the days of the month", fontweight="bold", size=15)

plt.plot()


# The percentage of delayed flights oscillates around 15% over the whole month. It can be noticed that the value has small periodical drops, probably related to the day of the week, and it generally increases, reaching its peak, at the end of the month. As already discussed, the values from the 29th to the 31st have to be taken in consideration carefully because not all the months have records for these days.

# In[ ]:


# create the pandas dataframe
df_dayweek_delayed =  pd.DataFrame({'count' : df_flights[df_flights.ARRIVAL_DELAY > 15].groupby('DAY_OF_WEEK').size()}).reset_index()
df_dayweek= pd.DataFrame({'count' : df_flights.groupby('DAY_OF_WEEK').size()}).reset_index()

df_dayweek_delayed['percentage']=df_dayweek_delayed['count']*100/df_dayweek['count']


# plot the number of delayed flights per day of the month
f, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="DAY_OF_WEEK",
            y="percentage",
            data=df_dayweek_delayed,
            palette=sns.color_palette("GnBu_d", 7),
            ax=ax)

# set plot's labels
ax.set(xlabel='Day of the week', ylabel='Percentage of delayed flight')
ax.set_xticklabels(days)
ax.set_title("Percentage of delayed flights over the days of the week", fontweight="bold", size=15)

plt.plot()


# As expected from the previous plot, the percentage of delayed flights is higher in the middle of the week (Thursday), while it decreases on saturday. This is probably what creates the bigger oscillations in the days-of-the-month plot. Furthermore it can be noted that the percentage of delayed lights drops to its minimum value on staurday, where we know from the previous plots that there are less flights.

# In[ ]:


# create the pandas dataframe
df_month_delayed =  pd.DataFrame({'count' : df_flights[df_flights.ARRIVAL_DELAY > 15].groupby('MONTH').size()}).reset_index()
df_month= pd.DataFrame({'count' : df_flights.groupby('MONTH').size()}).reset_index()

df_month_delayed['percentage']=df_month_delayed['count']*100/df_month['count']



# plot the number of delayed flights per month
f, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="MONTH",
            y="percentage",
            data=df_month_delayed,
            palette=sns.color_palette("GnBu_d", 12),
            ax=ax)

# set plot's labels
ax.set(xlabel='Month', ylabel='Percentage of delayed flight')
ax.set_xticklabels(months)
ax.set_title("Percentage of delayed flights over the months of the year", fontweight="bold", size=15)

plt.plot()


# We are ready now to draw some observations from our data, even if we have only looked at data coming from a year worth of flights:
# 
# * The probability for a flight to be delayed is low at the beginning or at the very end of a given months
# * Flights on two first weekdays and on the weekend, are less likely to be delayed
# * October and September are very good months for travelling, as the probability of delay is low 

# In[ ]:


# What is the delay probability for the top 20 busiest airports?
# top 20 busiest airports
df_busiest_airports = df_airport.sort_values('count',ascending=False).head(20).reset_index()
df_busiest_airports=df_busiest_airports.drop('index',axis=1)

# number of delayed flights in the top 20 busiest airport
df_delays_busiest_src_airports=pd.DataFrame({'count' : df_flights[(df_flights.ORIGIN_AIRPORT.isin([df_busiest_airports.iloc[i][0] for i in range(len(df_busiest_airports))])) & (df_flights.ARRIVAL_DELAY > 15)].groupby('ORIGIN_AIRPORT').size()}).reset_index()
df_delays_busiest_dest_airports =pd.DataFrame({'count' : df_flights[(df_flights.DESTINATION_AIRPORT.isin([df_busiest_airports.iloc[i][0] for i in range(len(df_busiest_airports))])) & (df_flights.ARRIVAL_DELAY > 15)].groupby('DESTINATION_AIRPORT').size()}).reset_index()
df_delays_busiest_src_airports=df_delays_busiest_src_airports.rename(columns={"ORIGIN_AIRPORT":"airport"})

# delay probability per source
df_prob_delay_busiest_src=df_busiest_airports.merge(df_delays_busiest_src_airports, on='airport',how='inner')
df_prob_delay_busiest_src['probability']=df_prob_delay_busiest_src['count_y']/df_prob_delay_busiest_src['count_x']
# delay probability per destination
df_delays_busiest_dest_airports=df_delays_busiest_dest_airports.rename(columns={"DESTINATION_AIRPORT":"airport"})

df_prob_delay_busiest_dest=df_busiest_airports.merge(df_delays_busiest_dest_airports, on='airport',how='inner')
df_prob_delay_busiest_dest['probability']=df_prob_delay_busiest_dest['count_y']/df_prob_delay_busiest_dest['count_x']

# delay propability per source and destination
prob_delay_busiest_any=df_prob_delay_busiest_src.merge(df_prob_delay_busiest_dest,on='airport',how='inner')
prob_delay_busiest_any=prob_delay_busiest_any.drop(['count_x_x','count_y_x','count_x_y','count_y_y'],axis=1)
prob_delay_busiest_any['probability']=prob_delay_busiest_any['probability_x']+prob_delay_busiest_any['probability_y']
prob_delay_busiest_any=prob_delay_busiest_any.drop(['probability_x','probability_y'],axis=1)


# In[ ]:


# plot 
f, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="airport",
            y="probability",
            data=prob_delay_busiest_any,
            palette=sns.color_palette("GnBu_d", 20),
            ax=ax)
ax.set(ylabel="All", xlabel="Airport")
ax.set(ylabel="Delay probability", xlabel="Airport")
ax.set_title("Delay probability of the top 20 busiest airports", fontweight="bold", size=15)

plt.plot()


# From the above figure we can said that for the top 20 busiest airport, it is more likely to have a delays. the maximum probability to get a delays is on the airport "LGA" it equals to 0.28. 
