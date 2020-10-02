#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# # How to use this dataset

# ## Getting Started

# The CSV file for each month contains the scraped data of nearly every train (~98.5%+) ran on the NJ Transit track system that month, including Amtrak trains which run exclusively on the Northeast Corridor. There are CSVs for every month from March 2018 (2018_03) to September 2018 (2018_09), as of the writing of this notebook; CSVs will be added for every month moving forward.

# Let's take a look at the data for April 2018:

# In[ ]:


# use correct path here
df_april = pd.read_csv('../input/2018_04.csv', index_col=False)
df_april.head(2)


# Peeking at the CSV, we see that the first couple of rows have "train_id" #7837 and "date" 2018-04-01. So, let's take a look at the data for train #7837 on April 1, 2018 to get an idea for how to work with this data. The first two rows here will be identical to the cell above:

# In[ ]:


df_april[(df_april["train_id"] == "7837") & (df_april["date"] == "2018-04-01")]


# Train #7837 is an NJ Transit train with 14 stops, originating out of New York Penn Station and terminating at Trenton. For a map of the entire NJ Transit system, click [here](https://www.njtransit.com/pdf/rail/Rail_System_Map.pdf).
# 
# This specific slice of data represents the trip made by train 7837 on April 1, 2018. Each stop in the train's journey is represented as one row in the data. The first row is `from` New York Penn Station and `to` New York Penn Station; this indicates the train's originating departure from New York Penn Station. The second row is `from` New York Penn Station and `to` Secaucus Upper Lvl; this represents the train's journey from New York Penn Station to Secaucus Upper Lvl.

# ## Data Dictionary

# <table>
#     <tr>
#         <th>column</th>
#         <th>type</th>
#         <th>description</th>
#     </tr>
#     
#     <tr>
#         <td>train_id</td>
#         <td>string</td>
#         <td>Train number in the NJT or Amtrak system. These are unique on a daily basis and correspond to the same scheduled train across multiple days. If the train_id contains and "A", it is an Amtrak train.</td>
#     </tr>
#     
#     <tr>
#         <td>date</td>
#         <td>string</td>
#         <td>Date of operation according to the 27-hour NJ Transit schedule. e.g. trains originating between 02/09/18 4:00 to 02/09/18 27:00 (actually 02/10/18 3:00) are considered to run on 02/09/18. </td>
#     </tr>
#     
#     <tr>
#         <td>stop_sequence</td>
#         <td>int</td>
#         <td>Scheduled stop number (e.g. 1st stop, 2nd stop) for the stop in the current row. </td>
#     </tr>
#     
#     <tr>
#         <td>from</td>
#         <td>string</td>
#         <td>Station the train is traveling from for the stop in the current row.</td>
#     </tr>
#     
#     <tr>
#         <td>from_id</td>
#         <td>int</td>
#         <td>Station id for the "from" station. Refer to rail_stations and stops.txt in rail_data/. </td>
#     </tr>
#     
#     <tr>
#         <td>to</td>
#         <td>string</td>
#         <td>Station the train is arriving to for the stop in the current row.</td>
#     </tr>
#     
#     <tr>
#         <td>to_id</td>
#         <td>int</td>
#         <td>Station id for the "to" station. Refer to rail_stations and stops.txt in rail_data/.</td>
#     </tr>
#     
#     <tr>
#         <td>scheduled_time</td>
#         <td>datetime</td>
#         <td>If "type" equals "NJ Transit", the scheduled departure time out of the "to" stop. Else, none.</td>
#     </tr>
#     
#     <tr>
#         <td>actual_time</td>
#         <td>datetime</td>
#         <td>If the status field is "departed", the actual departure time out of the "to" stop. If the status field is "cancelled", the time at which this stop was cancelled. If the status field is "estimated", the estimated departure time out of the "to" stop. </td>
#     </tr>
#     
#     <tr>
#         <td>delay_minutes</td>
#         <td>decimal</td>
#         <td>Only populated when "type" equals "NJ Transit". The difference between actual_time and scheduled_time, in minutes. Pre-cleaned to be > 0 for stops where "actual_time" less than "scheduled_time". </td>
#     </tr>
#     
#     <tr>
#         <td>status</td>
#         <td>string</td>
#         <td>Can take the values "departed", "cancelled", or "estimated". "departed" if stop was explicitly marked departed. "Cancelled" if stop was marked cancelled. None if the stop wasn't explicitly marked departed due to a terminated data stream for the train; estimated times used.</td>
#     </tr>
#     
#     <tr>
#         <td>line</td>
#         <td>string</td>
#         <td>The train line on NJ Transit or Amtrak. See <a href="https://www.njtransit.com/pdf/rail/Rail_System_Map.pdf">here</a> for NJ Transit train lines. All Amtrak lines run on the Northeast Corridor NJ Transit line.</td>
#     </tr>
#     
#     <tr>
#         <td>type</td>
#         <td>string</td>
#         <td>Either "NJ Transit" or "Amtrak". "Amtrak" trains do not have "scheduled_time" values.</td>
#     </tr>
#     
#     
#         
# </table>

# ## Basic NJ Transit delay analysis

# Let's convert the `scheduled_time` and `expected_time` columns to datetimes:

# In[ ]:


df_april['scheduled_time'] = pd.to_datetime(df_april['scheduled_time'])
df_april['actual_time'] = pd.to_datetime(df_april['actual_time'])


# Now, let's try to view cumulative delays for a train. The cumulative delay for a train is simply the "delay" value for the last stop for the train:

# In[ ]:


cumu_delay = df_april.groupby(['date', 'train_id']).last()


# In[ ]:


cumu_delay.head(2)


# Note that all the "status" values are "estimated". This is because we (anecdotally, but repeatedly) noticed that NJ Transit's departure vision does not mark the last station as "departed" in a consistent fashion. However, the latest estimated time for the last station is accurate. For more documentation on the parsing logic used to generate these CSVs, check out the [GitHub repository](https://github.com/pranavbadami/njtransit).

# Finally, let's look at the distribution of delays for commuters going into New York Penn Station in April across all NJ Transit trains:

# In[ ]:


# Get cumulative delay for NJ Transit trains to New York Penn Station
njt_nyp = cumu_delay[(cumu_delay['type'] == "NJ Transit") & (cumu_delay['to'] == "New York Penn Station")]
njt_nyp.head(2)


# We can do some high level analysis here, such as finding the distribution of cumulative delays for trains that were delayed more than 5 minutes:

# In[ ]:


njt_nyp[njt_nyp["delay_minutes"] >= 5]["delay_minutes"].hist(bins=20)


# Next, let's take a look at delays for all NJ Transit trains to New York Penn Station on April 2, 2018:

# In[ ]:


# filter based on the "date" index, which level 0 of the multiindex
njt_nyp_0402 = njt_nyp.loc[njt_nyp.index.get_level_values(0) == "2018-04-02"]
njt_nyp_0402.head(2)


# We can plot the delay of trains vs when they were scheduled to arrive at New York Penn Station. This gives us a sense of overall train behavior into New York Penn Station, a major station, on April 2; delays are particularly severe by the end of the morning rush hour (~9 am).

# In[ ]:


njt_nyp_0402.plot(x="scheduled_time", y="delay_minutes", figsize=(8,6))


# ## Amtrak Trains

# This dataset also contains observed performance data for the various Amtrak lines that run on the Northeast Corridor tracks.

# In[ ]:


amtrak = df_april[df_april["type"] == "Amtrak"]
amtrak.head(2)


# Let's take a look at Amtrak train A2205 on April 1, 2018:

# In[ ]:


amtrak[(amtrak['train_id'] == "A2205") & (amtrak['date'] == '2018-04-01')]


# Note that there are NaN values here in the "scheduled_time" column. This is because Amtrak schedule data has not (yet, see [issues](https://github.com/pranavbadami/njtransit/issues)) been incorporated into the dataset. The "delay_minutes" column also contains NaNs since it is derived using "scheduled_time". 

# While we cannot calculate delay statistics for Amtrak (yet), we can still capture the rail network traffic caused by these trains. For example, we can look at the total trip times for Amtrak trains:

# In[ ]:


# get first and last stops for Amtrak trains
amtrak_first = amtrak.groupby(['date', 'train_id']).first()
amtrak_last = amtrak.groupby(['date', 'train_id']).last()

# calculate total trip times for Amtrak trains
amtrak_trip_times = amtrak_last["actual_time"] - amtrak_first["actual_time"]
amtrak_trip_times.head()


# Amtrak cancellations may also be of interest:

# In[ ]:


amtrak[amtrak["status"] == "cancelled"].head(5)


# Thank you for checking out the dataset! I hope you discover something interesting in it that can help NJ Transit and its riders in the future. And if we ever run into each other on an NJ Transit train, please say hi!
