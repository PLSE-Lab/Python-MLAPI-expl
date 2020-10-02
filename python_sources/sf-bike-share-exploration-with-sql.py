#!/usr/bin/env python
# coding: utf-8

# Today I will be performing an exploratory analysis on the SF Bay Area Bike Share database using SQLite supplemented by the pandas library.  The goal of this project is to demonstrate proficiency in extracting data from database files using SQL and the ability to analyze these values in context. This database contains four data tables: station, status, trip, and weather.
# 
# Here are some questions I will try to answer over the course of this exploration:
# * What was the trip with the longest duration?
# * Do unregistered users take longer or shorter trips?
# *  Which stations are the most popular?
# * Which routes are the most popular?
# 
# 
# 

# **Importing Packages**

# In[ ]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


# First, the data must be imported into python. I will set up a function that takes an sql query as a parameter and returns a dataframe visualization of that query. This will save coding time in the long run.

# In[ ]:


db = sqlite3.connect('../input/database.sqlite')

def run_query(query):
    return pd.read_sql_query(query, db)


# Now that the data is ready to go, lets see what tables we have to work with.

# In[ ]:


query='SELECT name FROM sqlite_master;'

run_query(query)


# In[ ]:


query = 'SELECT * FROM trip LIMIT 3;'

run_query(query)


# **Q1: What was the trip with the longest duration? **

# In[ ]:


query = '''
SELECT *
FROM trip
ORDER BY duration DESC
LIMIT 1
'''

run_query(query)


# So it appears that the longest ride recorded is over six months long! It is very likely that this could have been a glitch. Pulling up the top 10 longest rides will hopefully provide some context as to whether this datapoint is a fluke.

# In[ ]:


query = '''
SELECT *
FROM trip
ORDER BY duration DESC
LIMIT 10
'''

run_query(query)


# Lets see if how common it is for a ride to go over 24 hours

# In[ ]:


query = '''
SELECT count(*)
AS \'Long Trips\'
FROM trip 
WHERE 
duration >= 60*60*24;
'''
#60 seconds in a minute, 60 minutes in an hour, 24 hours in a day

run_query(query)


# It appears that all but two of the 10 longest trips were made by unsubscribed customers. Let's see if unregistered customers are the main culprits.
# 
# 
# 
# **Q2: Do unregistered users take longer or shorter trips?**
# 
# First let's plot a pie chart to determine the proportion of unregistered users.

# In[ ]:


query = '''
SELECT subscription_type, count(*) AS count
FROM trip
GROUP BY subscription_type
'''

df = pd.read_sql_query(query, db)

labels = ['Casual', 'Subscriber']
sizes = df['count']
colors = ['lightblue', 'lightgreen']
explode = (0.1, 0)  # explode 1st slice
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, )
plt.title('Subscribed vs Unsubscribed Riders')
plt.axis('equal')
plt.show()


# In[ ]:


query = '''
SELECT subscription_type, AVG(duration)/60 AS 'Average Duration'
FROM trip
GROUP BY subscription_type
''' 
#since duration is in seconds, we will convert to minutes
run_query(query)


# So unsubscribed customers take much longer rides on average. This would make sense as unsubscribed customers are more unaware of the charges that are incurred if you go over your 30 minute time limit. Other potential influential factors include the fact that unsubscribed customers come from certain demographics, such as tourists who would be doing more sight-seeing.
# 
# 
# 
# **Q3: Which stations are the most popular?**
# 
# In order to extract the data needed, the "trip" and "station" tables will need to be inner-joined:
# 
# 

# In[ ]:


query = '''
SELECT station.name AS Station, count(*) AS Count
FROM station
INNER JOIN trip
ON station.id = trip.start_station_id
GROUP BY station.name
ORDER BY count DESC
LIMIT 5
''' 

run_query(query)


# For anyone who is familiar with San Fransisco, it would make sense why these are the most popular stations. Townsend at 4th is blocks away from AT&T park, the financial district, and is the location of both a Caltrain and BART station. Both the Caltrain and BART serve hundreds of thousands of commuters every day. The Ferry Building is also one of the iconic landmarks of San Fransisco and is right along the Embarcadero, a very popular area among tourists.
# 
# 
# Lets take this a step further. We can utilize the "status" table in the database and determine which stations are empty the most. It appears that the status table consists of status updates pulled from each station every two minutes. Each station contains 1,047,142 unique status readings.

# In[ ]:



#there are 1047142 total status readings for each station

query = '''
SELECT station.name AS Station, count(*) AS 'Total Empty Readings'
FROM station

INNER JOIN status
ON status.station_id=station.id
WHERE status.bikes_available=0
GROUP BY station.name

ORDER BY count(*) DESC
LIMIT 10

''' 

run_query(query)


# It turns out only two out of 5 of our most popular stations are in the top 10 most empty stations. The other three must be very popular drop-off stations as well, which allows the kiosks to maintain a steady number of bikes.
# 
# Lets create a quick plot to show the distribution of available bike readings for all stations.

# In[ ]:


query = '''
SELECT bikes_available AS 'Bikes Available'
FROM status

''' 

df = pd.read_sql_query(query, db)
df['Bikes Available'].plot.hist(bins=27, title='Bikes Available (All Stations)', 
                                ec='black', alpha=0.5)


# **Q4: Which routes are the most popular?**
# 
# 
# This can be done using a simple "group by" statement.

# In[ ]:


query='''
SELECT start_station_name, end_station_name, COUNT(*) AS Count
FROM trip
GROUP BY start_station_name, end_station_name
ORDER BY Count DESC
LIMIT 10;
  '''
run_query(query)


# 
