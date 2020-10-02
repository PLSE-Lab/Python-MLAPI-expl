#!/usr/bin/env python
# coding: utf-8

# # Business Intelligence Exercise
# ## XYZ Cover
# 
# ### Goal
#     We aim to provide answers to different questions requested in the Business Intelligence Exercise file provided. Those questions are to be leading each section of the present notebook.
#     
# ### Context
#     This is merely an exercise to showcase required skills for a BI analyst to join XYZ Cover, and neither the data nor the procedures here are final or reflect XYZ's intellectual ownership or processes. 
#     The data was provided in 5 different files by XYZ's team, via email. 
#      This is presented in a Kaggle notebook for 3 main reasons: The first being able to show in a linear fashion the thought process and technical actions are taken, the second is to allow reviewers to run it without installing any dependencies or the likes, and the third to allow me to practice and explore some techniques that I might now use on a daily basis most recently. 
# 
# ### Notes
#     Notes for assumptions and specific comments per sections are to be noted when needed. 
#     I will use many poorly name variables, which is not good Python practice, just to make sure every section of the notebook is nuclear and can be ran independently.
#     SQL and Python will be used all along this notebook to try different approaches. This is not by any means the ideal environment to work and produce every kind of report but works well for a proof of concept. 
#     Window functions are used differently in SQLite, therefore other ways to treat the data will be used due to time constraints. 

# Import all necessary libraries, give them aliases if needed.  We will use numpy for numerical analysis, pandas to treat data as data frames and sqlite3 to query our data provided in the files as an SQLite Database. 
# ### Note
# This step is required every new session we want to run the notebook. 

# In[ ]:


import numpy as np # linear algebra
import os
import pandas as pd # data processing
import sqlite3 as sq# enable creation and handling of a local db 


# Read the CSV files, create tables in a local DB and connect our notebook to said DB. 
# ### Note
# This step is required every new session we want to run the notebook. 

# In[ ]:


# set-up a connection to a newly named flock.db. 
# the connection will be called "conn"
conn = sq.connect('flock.db')
cursor = conn.cursor()


# Load all CSV files as SQL tables in our `flock.db`

# In[ ]:


for file in os.listdir('../input'):     # For all files in our directory
    df = pd.read_csv('../input/'+file)  # Read each CSV file
    df.to_sql(file[:-4],conn)           # Create the read file as a table in the database.


# ## Which aircraft has flown the most?
# 
# ### Assumption
# When saying "flown the most" we refer to the number of flights and not accumulated distance flown. 
# 
# ### Process
# Look at the data in the `individual_flights` table, count how many individual flights there were per aircraft, and then enhance the result with the name of the aircraft instead of the code. We will provide the number of flights as well the name of the aircraft. 

# In[ ]:


# Create the query string to then feed into the Pandas read_sql function. 
q1 = ('SELECT airc.aircraft_type, COUNT(indv.flight_id) AS number_of_flights '
      'FROM individual_flights AS indv '
      'JOIN aircraft AS airc '
      'ON indv.aircraft_id = airc.aircraft_id '
      'GROUP BY 1 '
      'ORDER BY number_of_flights DESC ')

# The function read_sql takes a query string and a database connection, and performs the query. 
r1 = pd.read_sql(q1,conn)

#We only need the first result, so we use iloc[[0]].
print(r1.iloc[[0]])


# ### Answer
# 
# The aircraft that has flown the most (number of flights) is **Goose**, with 1008 flights. 

# ## Which aircraft has carried the most passengers compared to the cost.
# 
# We lack specific data on how many passengers flew in each individual trip, therefore for simplicity we will assume it was the max capacity of each aircraft per flight.
# 
# The value we will consider as "most cost-efficient" or, "that has carried the most passengers compared to the cost" will be the smallest found value for "cost per passenger". There are 2 ways to express this answer:
# 
# > A: passengers / GBP or B: GBPs / passenger
# 
# By using A, we need to interpret "fraction of a passenger" because the values are likely to be decimals, even when mathematically correct, makes noise for general interpretation and business communications. On the other hand, by using B, will be more aligned with common logic and more naturally sounding. 
# 
# This is to say, if expressed as A, we are interested in the biggest value, if expressed as B, we are interested in the smallest one. 
# 
# We will provide both forms to illustrate.

# In[ ]:


#Answer form A. 

q2 = ('SELECT airc.aircraft_type, CAST((COUNT(indv.flight_id)*airc.capacity) AS FLOAT)/CAST(airc.cost AS FLOAT) AS passengers_per_cost '
      'FROM individual_flights AS indv '
      'JOIN aircraft AS airc '
      'ON indv.aircraft_id = airc.aircraft_id '
      'GROUP BY 1 '
      'ORDER BY passengers_per_cost DESC')

#Answer form B.
q3 = ('SELECT airc.aircraft_type, CAST(airc.cost AS FLOAT)/CAST((COUNT(indv.flight_id)*airc.capacity) AS FLOAT) AS cost_per_passenger '
      'FROM individual_flights AS indv '
      'JOIN aircraft AS airc '
      'ON indv.aircraft_id = airc.aircraft_id '
      'GROUP BY 1 '
      'ORDER BY cost_per_passenger ASC')

r2 = pd.read_sql(q2,conn)

r3 = pd.read_sql(q3,conn)

print('Form A, passengers/GBP: \n')
print(r2.iloc[:3])

print('\n')
print('Form B, GBP/passenger: \n')
print(r3.iloc[:3])


# ### Answer
# 
# Aircraft has carried the most passengers compared to the cost is **Goose**.

# ## Which airport has transported the most passengers through it?
# 
# ### Assumption
# Assume, again, max capacity per flight. This would be better expressed as "potentially most passengers transported"
# 
# ### Process
# Calculate the number of passengers transported as the number of flights of each aircraft type, times its capacity, all together grouped by airport. 
# 
# It is worth mentioning that every individual flight will transport N passengers, but that will double count the number of people because it will be attributed to both airports (departure and destination) per flight, therefore, N would be attributed to airport A and airport B.
# 
# We will leave the number in the answer query.  

# In[ ]:


# Adding an OR in the JOIN ON clause will account for inbound and outbound flights all alike.
q4 = ("""
    SELECT airport_name
      , SUM(n_passenger_per_aircraft) AS n_passengers
      FROM (
          SELECT indvf.aircraft_id
          , airp.airport_name 
          , COUNT(DISTINCT flight_id)*aic.capacity AS n_passenger_per_aircraft
          FROM individual_flights AS indvf
          JOIN airports AS airp
              ON airp.airport_code = indvf.destination_airport_code 
                  OR airp.airport_code = indvf.departure_airport_code
          JOIN aircraft AS aic
              ON aic.aircraft_id = indvf.aircraft_id    
          GROUP BY 1, 2)
      GROUP BY 1
      ORDER BY 2 DESC""")

r4 = pd.read_sql(q4,conn)

print('Number of passengers per airport (inbound and outbound) \n')
print(r4.iloc[:3])


# ### Answer
# 
# Airport that has transported more passengers through it is **Amazon Mothership**.

# ### Is there a difference when considering outbound and inbound passenger flow?
# 
# Parting from the previous results, analyse if slicing it by outbound or inbound would yield to different results. 

# In[ ]:


# Only count if outbound form airport X
q5 = ("""SELECT airport_name
      , SUM(n_outbound_passengers) AS outbound_passengers
      FROM (
            SELECT indvf.aircraft_id
          , airp.airport_name 
          , COUNT(DISTINCT flight_id)*aic.capacity AS n_outbound_passengers
          FROM individual_flights AS indvf
          JOIN airports AS airp
          ON airp.airport_code = indvf.departure_airport_code
          JOIN aircraft AS aic
          ON aic.aircraft_id = indvf.aircraft_id
          GROUP BY 1, 2)
       GROUP BY 1
       ORDER BY 2 DESC""")

# Only count if inbound form airport X
q6 = ("""SELECT airport_name
      , SUM(n_inbound_passengers) AS inbound_passengers
      FROM (
          SELECT indvf.aircraft_id
          , airp.airport_name 
          , COUNT(DISTINCT flight_id)*aic.capacity AS n_inbound_passengers
          FROM individual_flights AS indvf
          JOIN airports AS airp
          ON airp.airport_code = indvf.destination_airport_code
          JOIN aircraft AS aic
          ON aic.aircraft_id = indvf.aircraft_id
          GROUP BY 1, 2)
      GROUP BY 1
      ORDER BY 2 DESC""")

r5 = pd.read_sql(q5,conn)
print('Number of passengers per airport (outbound) \n')
print(r5.iloc[:3])
print('\n')

r6 = pd.read_sql(q6,conn)
print('Number of passengers per airport (inbound) \n')
print(r6.iloc[:3])


# ### Answer
# 
# For outbound flights, the airport that has transported more passengers through it is **Amazon Mothership**, but for inbound flights, it would be Flocktopia. 

# ### Is there a difference when considering the size of the airports?
# Parting from the first result, we need to account for the size of the airport and calculate a ratio. 
# 
# ### Process
# The total number of passengers per aircraft, sum for all aircraft that passed by each airport, and divided by the size of the airport. 

# In[ ]:


q7 = ("""SELECT airport_name
      , CAST(SUM(n_passenger_per_aircraft) AS FLOAT)/CAST(airport_size AS FLOAT) AS passagers_per_m2
      FROM (
          SELECT indvf.aircraft_id
          , airp.airport_name 
          , airp.airport_size
          , COUNT(DISTINCT flight_id)*aic.capacity AS n_passenger_per_aircraft
          FROM individual_flights AS indvf
          JOIN airports AS airp
              ON airp.airport_code = indvf.destination_airport_code 
                  OR airp.airport_code = indvf.departure_airport_code
          JOIN aircraft AS aic
              ON aic.aircraft_id = indvf.aircraft_id    
          GROUP BY 1, 2, 3)
      GROUP BY 1
      ORDER BY 2 DESC""")

r7 = pd.read_sql(q7,conn)
print(r7.iloc[[0]])


# ### Answer
# 
# **Amazon Mothership** is the airport with more potential passenger traffic in relationship to it's size. 

# ## What was the best year for Revenue Passenger-Miles for each airline? 
# 
# The fields that track international values have frequent null values. We will consider them as zeros for simplicity but that must be accounted for before doing any kind of analysis because that will bias the results. 
# 
# ### Process
# Calculate the SUM of all RPM (Domestic, International or both at the same time, so, Total), and then select the MAX per airline, which will yield us the year as well. 
# By selecting a specific type of RPM or both we will obtain different answers, therefore all three must be provided. 

# Considering RPM Domestic and RPM International separated:

# In[ ]:


q8 = ("""SELECT Airline_Name, Year, MAX(RPM_Domestic)  
    FROM (
        SELECT Airline_Code, CAST(substr(Date, -4) AS INT) AS Year
        , SUM(COALESCE(RPM_Domestic,0)) AS RPM_Domestic
        FROM flight_summary_data 
        GROUP BY 1, 2) AS subq
    JOIN
        airlines 
    ON subq.airline_code = airlines.airline_code
    GROUP BY 1""")

r8 = pd.read_sql(q8,conn)

print(r8)

print('\n')

q9 = ("""SELECT Airline_Name, Year, MAX(RPM_International)   
    FROM (
        SELECT Airline_Code, CAST(substr(Date, -4) AS INT) AS Year
        , SUM(COALESCE(RPM_International,0)) AS RPM_International
        FROM flight_summary_data 
        GROUP BY 1, 2) AS subq
    JOIN
        airlines 
    ON subq.airline_code = airlines.airline_code
    GROUP BY 1""")

r9 = pd.read_sql(q9,conn)

print(r9)


# Considering RPM Domestic and RPM International all together

# In[ ]:


q10 = ("""SELECT Airline_Name, Year, MAX(RPM_Total)   
    FROM (
        SELECT Airline_Code, CAST(substr(Date, -4) AS INT) AS Year
        , SUM((RPM_Domestic + COALESCE(RPM_International,0))) AS RPM_Total
        FROM flight_summary_data 
        GROUP BY 1, 2) AS subq
    JOIN
        airlines 
    ON subq.airline_code = airlines.airline_code
    GROUP BY 1""")


r10 = pd.read_sql(q10,conn)

print(r10)


# ### Answer
# The best years for each airline are the following, for the different types of RPM. 
# 
# **Domestic RPM:**
# 
# Amazon Airlines    2015
# 
# Flock Air                 2016
# 
# Goose Airways      2016
# 
# 
# **International RPM:**
# 
# Amazon Airlines   2016
# 
# Flock Air                2016
# 
# Goose Airways      2015
# 
# 
# **Total RPM:**
# 
# Amazon Airlines    2015
# 
# Flock Air                 2016
# 
# Goose Airways       2016

# ### What were the number of passenger miles each airline recorded?
# 
# ### Assumption
# When talking about number of passenger miles, it is refering to RPM, since that is the metric that most closely represents what asked. We will answer the question as if it was made "What was the total RPM each airline recorded?". 

# In[ ]:


q11 = ("""SELECT Airline_Name, SUM((RPM_Domestic + COALESCE(RPM_International,0))) AS RPM_Total
        FROM flight_summary_data 
        JOIN airlines
        ON flight_summary_data.airline_code = airlines.airline_code
        GROUP BY 1 ORDER BY RPM_Total""")

r11 = pd.read_sql(q11,conn)

print(r11)


# ### How does domestic passenger flow and international passenger flow vary by month of the year for Flocktopia airport?
# #### A single plot summarising this would be useful
# 
# ### Assumption
# Flow as the number of, because flow it is defined as volume / time unit, and in this case, we do not have that kind of specific information. 
# 
# Assume the question was asked as frame above, because otherwise, it would have required to answer "if a single plot summarising this would be useful", and in which case the answer would be no, but for reasons such as number of points and visibility, seasonality and other variants that might be affecting the variation of flow. 
# 
# ### Notes
# SQLite works differently and it is somehow more limited to deal with dates and formats, so the solution presented here is not elegant or best practice. 
# 

# In[ ]:


q12 = """SELECT SUM(passengers_domestic) AS total_passengers_domestic
, SUM(passengers_international) AS total_passengers_international
, CAST(substr(Date, -4) AS INT) AS Year
, CAST(substr(Date, 4, 2) AS INT) AS Month 
FROM flight_summary_data WHERE Airport_Code = "FKT" 
GROUP BY 3, 4
ORDER BY Year ASC, Month ASC"""

r12 = pd.read_sql(q12,conn)

r12 = r12.set_index(['Year','Month']).diff()

print(r12.plot(figsize=(18,10)))


# A more useful way to visualise this data is separating the domestic passengers and international passengers. 

# In[ ]:


q15 = """SELECT SUM(passengers_domestic) AS total_passengers_domestic
, CAST(substr(Date, -4) AS INT) AS Year
, CAST(substr(Date, 4, 2) AS INT) AS Month 
FROM flight_summary_data WHERE Airport_Code = "FKT" 
GROUP BY 2, 3
ORDER BY Year ASC, Month ASC"""

q16 = """SELECT SUM(passengers_international) AS total_passengers_international
, CAST(substr(Date, -4) AS INT) AS Year
, CAST(substr(Date, 4, 2) AS INT) AS Month 
FROM flight_summary_data WHERE Airport_Code = "FKT" 
GROUP BY 2, 3
ORDER BY Year ASC, Month ASC"""

r15 = pd.read_sql(q15,conn)
r16 = pd.read_sql(q16,conn)

r15 = r15.set_index(['Year','Month']).diff()
r16 = r16.set_index(['Year','Month']).diff()

mean_q = r15.mean(axis=0)

print(mean_q)
print(r15.plot(figsize=(18,10),title='Total Domestic Passengers by month'))
print(r16.plot(figsize=(18,10),title='Total International Passengers by month'))


# ### Note
# 
# Calculating the variation (delta) from each month would be a better way to represent this, but due to time constrains I will not do it for this analysis. 
# 
# ### Answer
# 
# The plot serves as an answer. We can tell that seasonality might be playing a huge factor, and once that is removed we could do more in-depth analysis. 

# ## What was the best year for growth for each airline? 
# 
# ### Process
# Define what is to be considered as "Growth", then split what the factors of Growth would be, compose an index for growth and then compare all growth indexes to see which one did better. 
# 

# We have ASM and RPM, as well the number of flights Domestic and for some flights International, per Airline and Airport. 
# To list some of the possible ways of evaluate growth, we have:
#     Decrease the difference between ASM - RPM, therefore captivating more passengers out of the ones that they had the capacity for. 
#     Increase on ASM, therefore increasing the fleet or the size of the planes, or the number of flights. 
#     Increase in number of flights alone. 
#     
# For simplicity and because it seems to be the one providing the best answer, we will use ASM as our growth indicator, and the more ASM an airline has over time, the more we will say they grew. 
# 
# To calculate the growth we will take the AVG(ASM_Domestic) per Airline per Year.
# 
# ### Assumption
# Assume that all airlines have ASM_Domestic for each the same number of months, because otherwise might be a little biased due to how average is calculated. 
# To simplify even further for time restrictions I will assume the MAX ASM will be the latest from 2017 and 2002 averages, and the min the other one. Seems logical but something of that sort should not be done lightly.
# 
# ### Note
# We would account only for ASM_Domestic, since ARM_International has a lot of null values and will bias the results, but the process would be similar to the one applied before for a prior question involving RPM. 
#     

# In[ ]:


q13 = ("""
SELECT SUM(ASM_Domestic) AS sum_asm_domestic
, Airline_Code
, CAST(substr(Date, -4) AS INT) AS Year
FROM flight_summary_data
GROUP BY 2, 3
HAVING CAST(substr(Date, -4) AS INT) <> 2002
""")

airportsquery = ("""
SELECT * FROM airports
""")

airportsdf = pd.read_sql(airportsquery,conn)

r13 = pd.read_sql(q13,conn)
# We get the difference between each year for each airline so we can see how much they "grew" from the past year. We group by Airline_Code first. 
r13['dif'] = r13.groupby(['Airline_Code'])['sum_asm_domestic'].diff()

# Find the indexes of each of the max values for dif. It will have negative values,but since we are talking about growth they will not be included. 
# According to the way we calculate the dif, we need to exclude the year 2002 because it only has a couple of months.  
idx = r13.groupby(['Airline_Code'], sort=False)['dif'].transform(max) == r13['dif']

# We print the years with the Max dif.
print(r13[idx])


# ### Answer
# **Goose Airways** 2010, **Flock Air** 2016 and **Amazon Airlines** 2012

# ### Which airport contributed the most to this? 
# 

# In[ ]:


q14 = ("""
SELECT SUM(ASM_Domestic) AS sum_asm_domestic
, Airline_Code
, Airport_Code
, CAST(substr(Date, -4) AS INT) AS Year
FROM flight_summary_data
GROUP BY 2, 3, 4
HAVING (Airline_Code='FA'AND CAST(substr(Date, -4) AS INT)=2016) 
OR (Airline_Code='AA'AND CAST(substr(Date, -4) AS INT)=2012)
OR (Airline_Code='GA'AND CAST(substr(Date, -4) AS INT)=2010)
""")
# Cherrypicking with very bad practice in the query the specific years in which I obtained the max growth. This should never be made like this. 

r14 = pd.read_sql(q14,conn)
print(airportsdf)

# Set and index frame to extract the max value'd airport per airline in the desired year. 
indx2 = r14.groupby(['Airline_Code'], sort=True)['sum_asm_domestic'].transform(max) == r14['sum_asm_domestic']

# Before printing, join with the airport table to get the airport names. 
print(pd.merge(r14[indx2], airportsdf, how='inner', left_on='Airport_Code', right_on='Airport_Code')[['Airline_Code','Airport_Name','sum_asm_domestic']])


# ### Answer
# For Amazon Airlines, it was Nestland Airport, for Flock Air it was Nestland Airport as well, and for Goose Airways it was Flocktopia. 
