#!/usr/bin/env python
# coding: utf-8

# It's Wednesday. I awoke shortly after 5am to a cold house; my breath condensing into twisting patterns that gave some context to the early morning air. As I reached into the dark recesses of the cupboard on the landing to turn on the heating, the door dragging last-night's carelessly-dumped cardigan across the floor as it opened, I thought to myself: "Hey, it's day 3 of Rachael's SQL Scavenger Hunt today. I wonder what it will be about today?"

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# The first challenge: **During which hours of the day do the most accidents occur?**

# Let's start by taking a peek at the head of [what we hope will be] the relevant table. I'm going with the 2015 data as that is the set used in the [Day 3 tutorial][1].
# 
# [1]: https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3

# In[ ]:


accidents.head("accident_2015")


# In[ ]:


# define query:

query1 = """SELECT COUNT(consecutive_number), 
            EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC

         """

# submit query using the safe, scan-size limited function
fatalitiesByHour = accidents.query_to_pandas_safe(query1)

# print the result
fatalitiesByHour


# Looks like that 1800 hour is a bad one; could that just be due to the sheer volume of traffic on the roads as people head home from work? Without normalising against the number of journeys, it's difficult to say...
# 
# The 1700 hour comes up in 3rd place, which might fit with that hypothesis, and the 2000, 2100 and 1900 being the other three hours in the top five might also fit with that, possibly with a quick beer or two after work?
# 
# If we plot that (sorry, I'm just more familiar with R right now!), we get:
# 
# 
# `ggplot(accidents, aes(hour, accidents)) + geom_line() + labs(x = 'Hour of Day', y = 'Count of Accidents', title = 'Accidents by Hour of Day')`
# 
# ![](https://i.imgur.com/HeVS9cf.png)
# 
# Does that look believable?

# The second challenge: **Which state has the most hit and runs?**

# In[ ]:


# define query:

query2 = """SELECT registration_state_name, COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
         """

# submit query using the safe, scan-size limited function
hitRun = accidents.query_to_pandas_safe(query2)

# print the result
hitRun


# Again, let's bring in ggplot2 in R to see what that looks like. Let's remove the unknowns, as that's going to make our figure look less pretty and scrunch everything up towards the bottom, while it hogs the limelight.
# 
# 
# `library(dplyr)
# stateHitRuns %>%
#   filter(stateName != 'Unknown') %>%
#   mutate(stateName = reorder(stateName, numHitRuns)) %>% 
#   ggplot(aes(stateName, numHitRuns)) + geom_col(fill = 'red') + coord_flip() + 
#   labs(x = 'State Name', y = 'Number of Hit and Runs', title = 'Hit and Runs in 2015 by State')`
#          
# ![](https://i.imgur.com/W25J3MG.png)         

# Okay, right, California, Texas and New York make up three of the top four. Given their populations, perhaps not unsurprising. It would be nice to spend some time taking these numbers and looking at them against some other information, but I've still got a working week to finish my alarm goes off in five hours.

# And that concludes the third day of the SQL challenge. Roll on Thursday and day 4...
