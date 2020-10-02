#!/usr/bin/env python
# coding: utf-8

# **FARS Analysis** by Keenan Komoto

# I give unto you, an exploratory analysis of the FARS system. If desired, you can sort through the FARS Analytical User's Manual (as I have) to get a better understanding of the specific tables and values contained within those tables:
# > https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812447
# 
# However, if you don't, no biggie. I will do my best to provide all the heavy lifting so you can just sit back and enjoy the ride.
# _______________________________________________________________________________________________________________________________________________
# 
# **Goals**
# 
# Through this analysis I would like to gain any useful information possible that may help reduce the amount of traffic related fatalities. It could be the slightest detail as in: "Results show that if you have a broken leg you are 0.6% more likely to be involved in a fatal accident"
# 
# Even the smallest details can make a difference. 
# 
# Now then, shall we begin?
# 

# In[ ]:


import bq_helper

fars = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                dataset_name="nhtsa_traffic_fatalities")
fars.head("accident_2016")


# **Drunk Driving**
# 
# First let's look at a feature that will most likely have a high correlation to traffic related fatalities, drunk drivers.

# In[ ]:


#Get data of fatalities, with/without drunk drivers and total fatalities
    #note: "dd" is being designated as "drunk driver" NOT "designated driver"
query_dd =        """
                     SELECT COUNT(number_of_fatalities) AS fatalities_with_drunk_drivers
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers = 1
                  """

query_no_dd =     """
                     SELECT count(number_of_fatalities) AS fatalities_no_drunk_drivers
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers = 0
                  """

query_all_fatal = """
                     SELECT count(number_of_fatalities) AS total_fatalities
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                  """

#run and print the queries
fatal_dd = fars.query_to_pandas_safe(query_dd)
print(fatal_dd)

print("_"*50)

fatal_no_dd = fars.query_to_pandas_safe(query_no_dd)
print(fatal_no_dd)

print("_"*50)

all_fatal = fars.query_to_pandas_safe(query_all_fatal)
print(all_fatal)


# Well this doesn't add up... 
# 
# >8474+25719=34193
# 
# We are off by 246. Hmmmm... 
# 
# NULL VALUES!

# In[ ]:


#check for null values in fatalities column
query_fatal_null = """
                      SELECT COUNT(number_of_fatalities) AS number_null_fatalities
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                      WHERE number_of_fatalities IS NULL
                   """

#check for null values in drunk drivers column
query_dd_null = """
                   SELECT COUNT(number_of_drunk_drivers) AS number_null_drunk_drivers
                   FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                   WHERE number_of_drunk_drivers IS NULL
                """

#run queries
null_fatal = fars.query_to_pandas_safe(query_fatal_null)
null_dd = fars.query_to_pandas_safe(query_dd_null)

#print queries
print(null_fatal)
print("_"*50)
print(null_dd)


# Well isn't this confusi- NEITHER!!!!!!!!!!!

# In[ ]:


#check for accident records with no drunk drivers and no fatalities
query_neither = """
                           SELECT COUNT(number_of_fatalities) AS nonFatal_noDrunkDriver
                           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                           WHERE number_of_fatalities = 0 AND number_of_drunk_drivers = 0
                        """
neither = fars.query_to_pandas_safe(query_neither)
print(neither)


# Well that is expected, especially since we are looking at data on **fatal** car crashes.
# 
# Hm. Looking back at the original set of code we have the number of drunk drivers as = 1. 
# 
# Maybe there is more than 1 drunk driver in a few accidents?

# In[ ]:


#check for accidents with more than one drunk driver
query_dd =        """
                     SELECT COUNT(number_of_fatalities) AS fatalities_with_drunk_drivers
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers >= 1
                  """
##next two are same queries as the first set we ran^^^
query_no_dd =     """
                     SELECT count(number_of_fatalities) AS fatalities_no_drunk_drivers
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers = 0
                  """

query_all_fatal = """
                     SELECT count(number_of_fatalities) AS total_fatalities
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                  """

#run and print the queries
fatal_dd = fars.query_to_pandas_safe(query_dd)
print(fatal_dd)

print("_"*50)

fatal_no_dd = fars.query_to_pandas_safe(query_no_dd)
print(fatal_no_dd)

print("_"*50)

all_fatal = fars.query_to_pandas_safe(query_all_fatal)
print(all_fatal)


# **HOORAY!!!!**
# 
# We figured it out!
# 
# So the big step here was going through all of the possible options, then once those ideas are exhausted (or hopefully sooner) we look back at the original idea and see what could of gone wrong with that. In this instance there were 246 accidents that involved more than 1 drunk driver (eeeeesh!). Hey, there is one more thing we know about this data set!
# 
# > "We don't make mistakes, just happy little accidents" -Bob Ross
# 
# 

# **Check List**
# 
# Okay, so far we know:
# 1. ~25% of accidents reported in the FARS system involve a drunk driver
# 2. 246 of those (~0.7%) involve more than 1 drunk driver
# 3. ~75% of accidents do not involve any drunk drivers
# 
# _______________________________________________________________________________________________________________________________________________
# **Next Steps**
# 
# Something that I am curious about is what caused most of the accidents?
# 
# But first, maybe we can figure out some more information about the accidents that involved drunk drivers.
# 
# Let's look at the drimpair_2016 file (file contains the physical impairments of the driver in an accident)
# 

# In[ ]:


#print the first few columns of the drimpair_2016 table
fars.head("drimpair_2016")


# Looking at our handy-dandy [FARS Analytical User's Manual](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812447), we see that the last two columns contain the condition of the driver in both integer and text form. Let's do some work and see what impairments the drunk drivers had.
# >I know this seems redundant but stick with me here, we might find something here that is useful (I swear that I haven't looked at this before. I like working, but not that much... Okay that was a lie, I actually do like working, BUT I haven't looked at this before. 

# In[ ]:


#Count number of rows in accident table
query_check_accident = """
                          SELECT COUNT(*) AS num_rows
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                       """

#Count number of distinct accident identifiers (consecutive_number) in accident table
query_distinct_accident = """
                             SELECT COUNT(DISTINCT consecutive_number) AS num_distinct_rows
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                          """

#run queries
check_accident = fars.query_to_pandas_safe(query_check_accident)
check_distinct_accident = fars.query_to_pandas_safe(query_distinct_accident)

#print
print(check_accident)
print("_"*50)
print(check_distinct_accident)


# In[ ]:


#Count number of rows in drimpair table
query_check_accident = """
                          SELECT COUNT(*) AS num_rows
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016`
                       """
#Count number of distinct identifiers in drimpair table
query_distinct_drimpair = """
                             SELECT COUNT(DISTINCT consecutive_number) AS num_distinct_rows
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016`
                          """

#run queries
check_accident = fars.query_to_pandas_safe(query_check_accident)
check_distinct_drimpair = fars.query_to_pandas_safe(query_distinct_drimpair)

#print
print(check_accident)
print("_"*50)
print(check_distinct_drimpair)


# Okay difference between the files is that the accident file contains information on all accidents reported in the FARS system while the drimpair file contains information about if a driver was impaired by any means (alcohol/drugs, deaf, vision, broken leg, etc.). So we need to match the consecutive_number (unique case number assigned to each crash) to see what impairments drunk drivers had (other than being under the influence)

# Currently I am checking to see how the two tables (drimpair_2016 and accident_2016) match up. 
# >I like to make sure my work is correct^^^

# In[ ]:


query_how_drunk = """
                     WITH fatal_drunk AS
                     (
                         SELECT DISTINCT consecutive_number AS match
                         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                         WHERE number_of_drunk_drivers >= 1
                     ) --grab identifiers with one or more drunk drivers
                     
                     SELECT condition_impairment_at_time_of_crash_driver_name AS driver_condition,
                            COUNT(condition_impairment_at_time_of_crash_driver_name) AS count_
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016` AS drimp
                     JOIN fatal_drunk ON
                         fatal_drunk.match = drimp.consecutive_number
                     GROUP BY driver_condition
                     ORDER BY count_ DESC
                  """
#run query
how_drunk = fars.query_to_pandas_safe(query_how_drunk)

#print
print(how_drunk)


# Right now our total count doesn't match up. The sum of the counts is 12,595 but the number of fatalities with drunk drivers is 8,720... What are the extra 3,875 values?

# Let's split this up and find the problem.

# In[ ]:


#CTE part of last query^
query_1stHalf = """
                           SELECT DISTINCT consecutive_number AS num_distinct
                           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                           WHERE number_of_drunk_drivers >= 1
                       """
#run
check_1stHalf = fars.query_to_pandas_safe(query_1stHalf)

#print
print(check_1stHalf)


# Our CTE section of the query gives us the correct number of values we want at 8,720. 
# 
# What about the rest of the query?

# In[ ]:


#non CTE part of query^^
query_2ndHalf = """
                     SELECT condition_impairment_at_time_of_crash_driver_name AS driver_condition,
                            COUNT(condition_impairment_at_time_of_crash_driver_name) AS count_
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016` /*AS drimp
                     INNER JOIN acc ON acc.num_distinct = drimp.consecutive_number*/ --commenting these
                     GROUP BY driver_condition
                     ORDER BY count_ DESC
                  """
#run
check_2ndHalf = fars.query_to_pandas_safe(query_2ndHalf)

#print
print(check_2ndHalf)


# Sums up to 52,436 values. 
# 
# Let's try something a little different:

# In[ ]:


#attempting original query^^^ with different clause/statements
query_draccident = """
                      SELECT condition_impairment_at_time_of_crash_driver_name AS driver_condition,
                             COUNT(condition_impairment_at_time_of_crash_driver_name) AS count_
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_2016` AS drimp
                      WHERE drimp.consecutive_number IN 
                      (
                          SELECT DISTINCT consecutive_number
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` AS acc
                          WHERE acc.number_of_drunk_drivers >= 1
                      )--Instead of using CTE we are trying the WHERE IN clause
                      GROUP BY driver_condition
                      ORDER BY count_ DESC
                   """

#run
draccident = fars.query_to_pandas_safe(query_draccident)

#print
print(draccident)


# Well....... I don't know why there are only 8,720 accidents with drunk drivers, but when we combine the two tables we get 12,595.
# 
# **If anyone reading this can weigh in on this issue I would be extremely grateful!**
# 
# Well can still make a pretty plot.

# In[ ]:


import matplotlib.pyplot as plt

plt.bar(draccident.driver_condition, draccident.count_)
plt.xticks(rotation=90)
plt.title("Driver Impairments in Drunk Driving Accidents")


# Well that's a bit ugly

# In[ ]:


import seaborn as sns

#set up plot
ax=plt.axes()
draccident_plot = sns.barplot(x="driver_condition", y="count_", data=draccident, palette="hls", ax=ax)
ax.set_title("Driver Impairments in Drunk Driving Accidents")
#rotate x-axis labels
for tick in draccident_plot.get_xticklabels():
    tick.set_rotation(90)


# **Ahhhh, that's better.**

# There is a surprising amount of "Other Physical Impairment" as well as "Asleep or Fatigued"
# Overall, there isn't much useful information here other than confirming that in accidents involving drunk drivers, the driver was under the influence. At least the FARS reporting system is accurate in this!

# >Until a superhero comes and saves me from the SQL nightmare I just had, let's figure out what other insights we can gain from this dataset

# **Vision Table**
# 
# I am curious about the vision table, looking at our handy-dandy manual (https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812447) we see that the vision table provides, guess what? Data about anything that obstructed a drivers **vision**.. Yes, thank you manual, I thought it was telling me if a driver had 20/20 vision or not. Actually, that might be interest- no Keenan, next time... next time.
# 
# Onwards!

# In[ ]:


#query to grab state (name from accident_2016 table), 
#obstruction, and count of obstructions from vision_2016 table
query_vision = """
                  WITH state AS
                  (
                      SELECT state_number AS s_num, state_name
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                  )
                  SELECT drivers_vision_obscured_by_name AS obstruction, state.state_name,
                         COUNT(drivers_vision_obscured_by_name) as obstruction_count
                  FROM `bigquery-public-data.nhtsa_traffic_fatalities.vision_2016` AS vision
                  JOIN state ON vision.state_number = state.s_num
                  GROUP BY state_name, obstruction
                  ORDER BY obstruction_count DESC
               """

#run query
vision_obstruct = fars.query_to_pandas_safe(query_vision)

#print
print(vision_obstruct)


# In[ ]:


#set up plot
obstruction_plot = sns.barplot(x="obstruction", y="obstruction_count", 
                               data=vision_obstruct, hue="state_name")

#move legend outside of figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#rotate x-axis labels
for tick in obstruction_plot.get_xticklabels():
    tick.set_rotation(90)


# **BAD PLOT!!!** 
# 
# This might be a bit overwhelming to find a good plot that could fit all of this information.
# 
# Looking back a bit we see that Texas has the most reports of obstructions, let's just plot Texas data. Using our handy-dandy manual we see that Texas is labeled as state number 48 (alphabetical). That makes things a bit easier

# In[ ]:


#query to grab obstruction data from Texas
query_texas_vis = """
                      SELECT drivers_vision_obscured_by_name AS obstruction,
                             COUNT(drivers_vision_obscured_by_name) as obstruction_count
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.vision_2016` AS vision
                      WHERE vision.state_number = 48
                      GROUP BY obstruction
                      ORDER BY obstruction_count DESC
                  """

#run query
texas_vis = fars.query_to_pandas_safe(query_texas_vis)

#plot
ax = plt.axes()
texas_vis_plot = sns.barplot(x="obstruction", y="obstruction_count", data=texas_vis, ax = ax)
ax.set_title("Obstructions in Texas")

#rotate x-axis labels
for tick in texas_vis_plot.get_xticklabels():
    tick.set_rotation(90)


# Yup, this confirms that there is nothing in Texas. Just kidding. Sort of. 

# _______________________________________________________________________________________________________________________________________________
# -------------------------------------------------------------**DISCLAIMER** ------------------------------------------------------------
# 
# I am still working on this next part, it seems like running a Python loop within a SQL query in BigQuery is going to pose a bit of a challenge for me. Will continue to update this notebook.
# _______________________________________________________________________________________________________________________________________________

# I am going to attempt to plot each state individually. First, we need to get all the information from the tables. Only problem is that we don't want to run 50 SQL queries by hand...
# 
# **PYTHON!!!**

# In[ ]:


number = 1
null_states = [3,7,14,43,52] #these numbers are missing or either Puerto Rico/Virgin Islands
                             #handy-dandy manual^ :)
#set up arrays
numbers = []
trash = []

#for loop to store numbers that indicate states
for number in range(55):
    number += 1
    if number in null_states:
        trash.append(number)
    else:
        numbers.append(number)

#quick check
#print(numbers)


# Now that our state numbers are all set and ready to go let's run **a** query.

# In[ ]:


#query to grab obstruction data from Texas
for i in numbers:
    i_allstates_vis =     """
                             SELECT drivers_vision_obscured_by_name AS obstruction,
                                    COUNT(drivers_vision_obscured_by_name) as obstruction_count
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vision_2016` AS vision
                             WHERE vision.state_number = (?)
                             GROUP BY obstruction
                             ORDER BY obstruction_count DESC
                          """, (i)

#run query
allstates_vis = fars.query_to_pandas_safe(i_allstates_vis)


# Uh oh...
