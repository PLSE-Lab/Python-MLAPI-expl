#!/usr/bin/env python
# coding: utf-8

# # Data Analysis with SQLite

# ## There is no better tool for analyzing relational databases than SQL. This tool has been around for quite a while and its not going anywhere anytime soon; because it is fast, reliable, elegant and easy to read even for non-programmers. Therefore, I am forcing myself out of my comfort (pandas) zone and tryng to better undestand the SQLite by fully engaging with it. In this short tutorial, I will ask and answer questions about the bicycle sharing dataset by writing queries and comments. If you are someone who is good with SQL, please comment on my queries with your advice(s) on improving it. Hope you will find it worthy of your time! 

# In[ ]:


# Importing all the essential libraries for our data analysis with SQLite
import sqlite3
import pandas as pd


# In[ ]:


conn = sqlite3.connect('../input/database.sqlite')


# In[ ]:


#There are four connected tables: 1)Station info, 2)Status with timestamps, 3)Trips & 4)Weather
#Let's see all the columns and first 10 rows of the Station table
pd.read_sql('''
    SELECT *
    FROM station
    LIMIT 10;
''', con=conn)


# ### For each given location/city, how many docks were installed, how many stations are there and what is the average capacity of each station?

# In[ ]:


pd.read_sql('''
    SELECT city, 
    SUM(dock_count) AS total_capacity, 
    COUNT(name) AS station_count, 
    ROUND(SUM(dock_count)/COUNT(name), 2) AS average_capacity_per_station
    FROM station
    GROUP BY city
    ORDER BY station_count DESC;
''', con=conn, index_col='city')


# ### Per city and date, what is the dock and station count?

# In[ ]:


pd.read_sql('''
    SELECT city,
           CASE
           -- m/d/yyyy
           WHEN (length(installation_date) = 8 AND substr(installation_date,2,1) = '/') 
           THEN substr(installation_date,5,4)||'-0'||substr(installation_date,1,1)||'-0'||substr(installation_date,3,1)
           -- m/dd/yyyy
           WHEN (length(installation_date) = 9 AND substr(installation_date,2,1) = '/') 
           THEN substr(installation_date,6,4)||'-0'||substr(installation_date,1,1)||'-'||substr(installation_date,3,2)
           -- mm/d/yyyy
           WHEN (length(installation_date) = 9 AND substr(installation_date,3,1) = '/') 
           THEN substr(installation_date,6,4)||'-0'||substr(installation_date,1,2)||'-'||substr(installation_date,4,1)
           -- mm/dd/yyyy
           WHEN (length(installation_date) = 10 AND substr(installation_date,3,1) = '/') 
           THEN substr(installation_date,7,4)||'-'||substr(installation_date,1,2)||'-'||substr(installation_date,4,2)
           ELSE installation_date
           END AS installed_date, 
           SUM(dock_count) AS total_dock_ct,
           COUNT(name) AS station_count
    FROM station
    GROUP BY 1,2
    ORDER BY 2,1;
''', con=conn)


# ### With many other RDBMS based on Structured Query Languages (such as MySQL, or PostgreSQL), we would have avoided such a lengthy "ifelse" statements b/c their data type support streches well beyond the five primitive data types (which SQLite offers)
# * **For example in PostgreSQL, we could easily change the 'string' object into 'date' by writing  <<CAST(date_column AS DATE)>> or check below for our specific case**

# In[ ]:


'''SELECT city, 
       CAST(SUBSTR(installation_date, LENGTH(installation_date)-3) || '-' ||  
       SUBSTR(installation_date, 0, INSTR(installation_date,'/')) || '-' ||
       REPLACE((SUBSTR(installation_date, INSTR(installation_date,'/') + 1, 2)), '/', '') AS DATE) AS installed_date
       SUM(dock_count) as station_count,
FROM station
GROUP BY 1,2
ORDER BY 2,1;'''


# ### From above query's result, it seems like most installations occurred on August (of 2013). Let's find exact number for AUG13 installments and the rest 

# In[ ]:


pd.read_sql('''
    WITH t1 AS (SELECT city,
                       CASE
                       -- m/d/yyyy
                       WHEN (length(installation_date) = 8 AND substr(installation_date,2,1) = '/') 
                       THEN substr(installation_date,5,4)||'-0'||substr(installation_date,1,1)||'-0'||substr(installation_date,3,1)
                       -- m/dd/yyyy
                       WHEN (length(installation_date) = 9 AND substr(installation_date,2,1) = '/') 
                       THEN substr(installation_date,6,4)||'-0'||substr(installation_date,1,1)||'-'||substr(installation_date,3,2)
                       -- mm/d/yyyy
                       WHEN (length(installation_date) = 9 AND substr(installation_date,3,1) = '/') 
                       THEN substr(installation_date,6,4)||'-0'||substr(installation_date,1,2)||'-'||substr(installation_date,4,1)
                       -- mm/dd/yyyy
                       WHEN (length(installation_date) = 10 AND substr(installation_date,3,1) = '/') 
                       THEN substr(installation_date,7,4)||'-'||substr(installation_date,1,2)||'-'||substr(installation_date,4,2)
                       ELSE installation_date
                       END AS installed_date, 
                       SUM(dock_count) AS total_dock_ct,
                       COUNT(name) AS station_count
                FROM station
                GROUP BY 1,2
                ORDER BY 2,1)
    SELECT CASE 
        WHEN month = '2013-08-01'
        THEN 'AUG13'
        WHEN month > '2013-08-01'
        THEN 'after_AUG13'
        ELSE 'before_AUG13'
        END AS installation_month,
        SUM(total_dock_ct) AS dock_ct,
        SUM(station_count) AS station_ct
    FROM (SELECT DATE(installed_date, 'start of month') as month,
                 total_dock_ct, 
                 station_count
         FROM t1
         ) AS innerquery
    GROUP BY 1;
''', con=conn, index_col='installation_month')


# ### Above 90 percent of all installments occurred on AUG13

# In[ ]:


#Let's see all the columns and first 10 rows of the Status table which updated its status every minute
pd.read_sql('''
     SELECT * 
     FROM status
     LIMIT 10;
''', con=conn)


# ### For each given station, on average how many bikes were available vs docks? And which station had the least bicycles and most docks available on average (or was the busiest station)?

# In[ ]:


pd.read_sql('''
     SELECT ROUND(AVG(status.bikes_available),2) AS avg_available_bikes, 
            ROUND(AVG(status.docks_available),2) AS free_dock_count,
            station.dock_count AS max_dock_capacity,
            station.name
     FROM status
     INNER JOIN station
     ON status.station_id = station.id
     GROUP BY name
     ORDER BY 1, 2 DESC
     LIMIT 10;
''', con=conn)


# ### Quite suprisingly, the 2nd and Folsom St. seems to be the busiest (Mabe b/c there is a bus station terminal nearby). Let's explore more and maybe this table is not revealing the full picture
# ### But if the 2nd and Folsom St. is the busiest; then what month (on average) had the least available bikes and does the resulted query seems bias for any particular month?  

# In[ ]:


pd.read_sql('''
    SELECT ROUND(AVG(status.bikes_available),2) AS avg_available_bikes, 
           ROUND(AVG(status.docks_available),2) AS free_dock_count,
           COUNT(*) AS num_occurrences,
           DATE(SUBSTR(time, 1,4) || '-' || SUBSTR(time, 6,2) || '-' || 
                SUBSTR(time, 9,2) ||  SUBSTR(time, 11,9), 'start of month') AS month 
    FROM status
    INNER JOIN station
    ON status.station_id = station.id
    WHERE station.name = '2nd at Folsom'
    GROUP BY station.name, month
    ORDER BY 2 DESC;
''', con=conn)


# ### AUG13 - it has the most but it only has 3,200 rows for that month compared to above 40,000 for any other month and year. Therefore, April of 14 is the busiest month.
# ### Let's check out the trips table and see whether our definition for being the 'busiest' based on available bikes and free docks hold up or falls apart 

# In[ ]:


pd.read_sql('''
    SELECT *
    FROM trip
    LIMIT 10
''', con=conn)


# ### Is there anything fishy about the duration column?  What are the maximum duration rides and are they common?

# In[ ]:


pd.read_sql('''
    SELECT duration/60/60 AS duration_hr, COUNT(*) AS frequency
    FROM trip
    GROUP BY duration_hr
    ORDER BY duration_hr DESC 
    LIMIT 15;
''', con=conn)


# ### The answer is no and probably, those outliers need to be dropped. If the day-long ride is 15 hours or less, then how often riders were within that timeframe?

# In[ ]:


pd.read_sql('''
    SELECT duration/60/60 AS duration_hr, COUNT(*) AS frequency
    FROM trip
    WHERE duration_hr <= 15
    GROUP BY duration_hr
    ORDER BY duration_hr DESC;
''', con=conn)


# ### Okay, overwhelming majority of the rides duration were less than an hour. Now back to our original question -  which stations are the busiest pickup and dropoff locations in SF? 

# In[ ]:


pd.read_sql('''
    SELECT COUNT(*) AS num_count,
           trip.start_station_name
    FROM trip
    INNER JOIN station
    ON station.id = start_station_id
    WHERE station.city = 'San Francisco'
    GROUP BY 2
    ORDER BY 1 DESC
    LIMIT 5;
''', con=conn)


# In[ ]:


pd.read_sql('''
    SELECT COUNT(*) AS num_count, 
           trip.end_station_name
    FROM trip
    INNER JOIN station
    ON station.id = start_station_id
    WHERE station.city = 'San Francisco'
    GROUP BY 2
    ORDER BY 1 DESC
    LIMIT 5;
''', con=conn)


# ### Unlike the 2nd&Folsom, Now those two results make sence
# ### For the remaining queries, we will not include the rows which has longer than 15 hour-duration
# ### What are the most popular bicycle routes in SF and what are the average duration time in minutes? 

# In[ ]:


pd.read_sql('''
    SELECT COUNT(*) AS num_count,
           ROUND(AVG(duration/60), 2) AS avg_duration_mins,
           trip.start_station_name,
           trip.end_station_name
    FROM trip
    INNER JOIN station
    ON station.id = start_station_id
    WHERE station.city = 'San Francisco' AND duration/60/60 <= 15
    GROUP BY 3, 4
    ORDER BY 1 DESC
    LIMIT 15;
''', con=conn)


# ### The duration ranges from 4 to 20 minutes for the most popular routes in SF
# ### For the trips which occurred above 2,000 times (in 2 years priod) in SF, what is the average duration in minutes?

# In[ ]:


pd.read_sql('''
    SELECT AVG(avg_duration_mins) AS avg_duration_for_most_pop
    FROM (SELECT COUNT(*) AS num_count,
                 ROUND(AVG(duration/60), 2) AS avg_duration_mins,
                 trip.start_station_name,
                 trip.end_station_name
          FROM trip
          INNER JOIN station
          ON station.id = start_station_id
          WHERE station.city = 'San Francisco' AND duration/60/60 <= 15
          GROUP BY 3, 4
          HAVING COUNT(*) > 2000
          ORDER BY 1 DESC) AS most_popular_trips
''', con=conn)


# ### What are the popular long-trip routes which occurred at least 50 times (in the last 2 years)?

# In[ ]:


pd.read_sql('''
    SELECT COUNT(*) AS num_count,
           ROUND(AVG(duration/60), 2) AS avg_duration_mins,
           trip.start_station_name,
           trip.end_station_name
    FROM trip
    INNER JOIN station
    ON station.id = start_station_id
    WHERE city='San Francisco' AND duration/60/60 <= 15
    GROUP BY 3, 4
    HAVING COUNT(*) > 50
    ORDER BY 2 DESC
    LIMIT 15;
''', con=conn)


# ### Most of them are round-trips (meaning, they were pickup and dropped off at the same location and on average they are somewhere between 1,5-2,5 hours
# ### Who rides more often, customer or subscriber?

# In[ ]:


pd.read_sql('''
    SELECT subscription_type,
           COUNT(*) AS sub_type_ct
    FROM trip
    GROUP BY 1;
''', con=conn)


# ### For weekdays and weekends (and also per each day of the week), what is the average duration per subscriber type?

# In[ ]:


pd.read_sql('''
    WITH t1 AS (SELECT DATE(CASE
                -- m/d/yyyy
               WHEN ((INSTR(start_date, ' ')-1) = 8 AND substr(start_date,2,1) = '/') 
               THEN substr(start_date,5,4)||'-0'||substr(start_date,1,1)||'-0'||substr(start_date,3,1)
               -- m/dd/yyyy
               WHEN ((INSTR(start_date, ' ')-1) = 9 AND substr(start_date,2,1) = '/') 
               THEN substr(start_date,6,4)||'-0'||substr(start_date,1,1)||'-'||substr(start_date,3,2)
               -- mm/d/yyyy
               WHEN ((INSTR(start_date, ' ')-1) = 9 AND substr(start_date,3,1) = '/') 
               THEN substr(start_date,6,4)||'-0'||substr(start_date,1,2)||'-'||substr(start_date,4,1)
               -- mm/dd/yyyy
               WHEN ((INSTR(start_date, ' ')-1) = 10 AND substr(start_date,3,1) = '/') 
               THEN substr(start_date,7,4)||'-'||substr(start_date,1,2)||'-'||substr(start_date,4,2)
               ELSE start_date
               END) AS trip_date,
               subscription_type, 
               (duration / 60) AS duration_min
        FROM trip
        INNER JOIN station
        ON station.id = start_station_id
        WHERE city='San Francisco' AND duration/60/60 <= 15)
    SELECT CASE 
           WHEN (trip_date IN (DATE(trip_date, 'weekday 6'), DATE(trip_date, 'weekday 0')))
           THEN 'weekends'
           ELSE 'weekdays'
           END AS weekday,
           subscription_type,
           ROUND(AVG(duration_min), 2) AS avg_dur_min
           
    FROM t1
    GROUP BY 1,2;
''', con=conn)


# In[ ]:


pd.read_sql('''
    WITH t1 AS (SELECT DATE(CASE
                -- m/d/yyyy
               WHEN ((INSTR(start_date, ' ')-1) = 8 AND substr(start_date,2,1) = '/') 
               THEN substr(start_date,5,4)||'-0'||substr(start_date,1,1)||'-0'||substr(start_date,3,1)
               -- m/dd/yyyy
               WHEN ((INSTR(start_date, ' ')-1) = 9 AND substr(start_date,2,1) = '/') 
               THEN substr(start_date,6,4)||'-0'||substr(start_date,1,1)||'-'||substr(start_date,3,2)
               -- mm/d/yyyy
               WHEN ((INSTR(start_date, ' ')-1) = 9 AND substr(start_date,3,1) = '/') 
               THEN substr(start_date,6,4)||'-0'||substr(start_date,1,2)||'-'||substr(start_date,4,1)
               -- mm/dd/yyyy
               WHEN ((INSTR(start_date, ' ')-1) = 10 AND substr(start_date,3,1) = '/') 
               THEN substr(start_date,7,4)||'-'||substr(start_date,1,2)||'-'||substr(start_date,4,2)
               ELSE start_date
               END) AS trip_date,
               subscription_type, 
               (duration / 60) AS duration_min
        FROM trip
        INNER JOIN station
        ON station.id = start_station_id
        WHERE city='San Francisco' AND duration/60/60 <= 15)
    SELECT CASE 
           WHEN trip_date = (DATE(trip_date, 'weekday 1'))
           THEN '1 - Monday'
           WHEN trip_date = (DATE(trip_date, 'weekday 2'))
           THEN '2 - Tuesday'
           WHEN trip_date = (DATE(trip_date, 'weekday 3'))
           THEN '3 - Wednesday'
           WHEN trip_date = (DATE(trip_date, 'weekday 4'))
           THEN '4 - Thursday'
           WHEN trip_date = (DATE(trip_date, 'weekday 5'))
           THEN '5 - Friday'
           WHEN trip_date = (DATE(trip_date, 'weekday 6'))
           THEN '6 - Saturday'
           ELSE '7 - Sunday'
           END AS weekday,
           subscription_type,
           ROUND(AVG(duration_min), 2) AS avg_dur_min
           
    FROM t1
    GROUP BY 1,2
    ORDER BY 2,1;
''', con=conn)


# ### Wow, customers on average tend to ride 4-5 times longer than subscribers!!! Now let's ask the same question but only for customers - does the popular routes in SF changes? And of course, we are expecting to see longer durations for each popular route

# In[ ]:


pd.read_sql('''
    SELECT COUNT(*) AS num_count,
           ROUND(AVG(duration/60), 2) AS avg_duration_mins,
           trip.start_station_name,
           trip.end_station_name
    FROM trip
    INNER JOIN station
    ON station.id = start_station_id
    WHERE station.city = 'San Francisco' AND duration/60/60 <= 15 AND subscription_type = 'Customer'
    GROUP BY 3, 4
    ORDER BY 1 DESC
    LIMIT 15;
''', con=conn)


# ### It changes quite a bit - both the popular locations and average duration time

# ## *It was quite easy to use SQLite with python but definately, missing the datetime object is the big minus. Maybe bigquery can solve the shortcomings of SQLite. In any case, bigquery is going to be my next challenge (with larger dataset and more in depth analysis). I hope reading my kernel was worthy of your time and attention. If you have any questions, suggestions or even disagreements, please feel free to comment it below. If you liked it, please upvote it. Thanks!*
