#!/usr/bin/env python
# coding: utf-8

# # Stocking rental bikes

# ## Abstract 
# 
# This kernel is a study on how to do better BQML exercise on the Austin rental bikes proposed [here](https://www.kaggle.com/rtatman/bigquery-machine-learning-exercise?utm_medium=email&utm_source=intercom&utm_campaign=sql-summer-camp). 
# The outcome of the exercise was that using the out-of-the-box 2017 data to predict the number of rides in 2018 does not work. 
# It is obvious that 2018 saw an increase of bike rides with respect to 2017, but in this kernel I look into why this is happening in more detail. 
# 
# It is found that all the stations for which it is not possible to make an accurate prediction are geographically close and close to a campus of the Univeristy of Texas. These stations saw an increase in number of rides around mid February 2018. This lead me to look into what happen in the area in that period. 
# 
# I found out that the Austin B-cycle company launched a new initiative in the area, that included more stations and free passes for students (more details [here](https://www.bcycle.com/news/2018/02/14/austin-b-cycle-expands-service-to-ut-campus-and-west-campus-neighborhoods)). This is why it is not possible to make accurate prerictions for a lot of stations. 
# 
# If I had been working in the bike sharing company, this is something I would have for sure known from the beginning and I would have taken it into account. 

# ## Setup

# In[ ]:


# Set your own project id here
PROJECT_ID = "bqml-bike-rental" # a string, like 'kaggle-bigquery-240818'

import pandas as pd

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID


# In[ ]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# ## Take a quick look at the data

# In[ ]:


# quick look at the table
table = client.get_table('bigquery-public-data.austin_bikeshare.bikeshare_trips')
client.list_rows(table, max_results=5).to_dataframe()


# ## Differences between 2018 and previous years
# 

# ### Number of rides per year

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'rides_station_year', '\nSELECT start_station_name, EXTRACT(YEAR from start_time) as year, \n    COUNT(1) as num_rides\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nGROUP BY year, start_station_name\nORDER BY year, start_station_name')


# Look at how many rides are done each year

# In[ ]:


rides_station_year.groupby("year").sum().plot.bar()


# 2018 saw a huge increase in the number of rides. 
# This can be due to the increase in number of stations or to an increase of the number of rides from each station.

# ### Rides per day

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'rides_per_day', '\nWITH counts AS (\n    SELECT TIMESTAMP_TRUNC(start_time, DAY) as day,  \n        EXTRACT(YEAR from start_time) as year,\n        start_station_name, \n        COUNT(1) as num_rides\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    GROUP BY day, start_station_name, year\n), \navg_station AS(\n    SELECT year, start_station_name, \n        AVG(num_rides) as num_rides\n    FROM counts\n    GROUP BY start_station_name, year\n)\nSELECT year, AVG(num_rides) as avg_rides_per_day\nFROM avg_station\nGROUP BY year\nORDER BY year')


# In[ ]:


rides_per_day


# The average number of rides per station per year is indeed higher for 2018 than for the other years. 
# This information does not take into account the number of stations. For example, 2017 has a lower average per station per day than 2016, but the total nuber of rides is actually higher. It can be interesting to look also at the number of stations. 

# ### Number of stations

# To veryfy that, we can use pandas to group by the name of the station the datafeame we already have (`rides_station_year`), and check the year the station was used for the first time (the minimum of the years). 

# In[ ]:


rides_station = rides_station_year.groupby("start_station_name").agg({"num_rides":"sum","year":"min"})
rides_station.plot.scatter(x='year', y='num_rides')


# Quite a few stations were added in 2018, including one that alone had more than 70000 rides (cumulative in 2018 and 2019). 

# In[ ]:


rides_station.sort_values("num_rides", ascending=False).head()


# This new station is `21st & Speedway @PCL`, with 72799 rides. 

# ### Information on the stations
# To understand better the differences, we can focus on 2017 and 2018. 
# We can check which stations have been added or removed in 2018. 

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'rides_in_2017', "\nSELECT start_station_name, TIMESTAMP_TRUNC(start_time, HOUR) as start_hour, \n    COUNT(1) as num_rides_2017\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time BETWEEN '2017-01-01' and '2018-01-01'\nGROUP BY start_hour, start_station_name \nORDER BY num_rides_2017 DESC")


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'rides_in_2018', "\nSELECT start_station_name,start_station_id, TIMESTAMP_TRUNC(start_time, HOUR) as start_hour, \n    COUNT(1) as num_rides_2018\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time BETWEEN '2018-01-01' AND '2019-01-01'\nGROUP BY start_hour, start_station_name , start_station_id\nORDER BY num_rides_2018 DESC")


# In[ ]:


rides_in_2017_grouped = rides_in_2017.groupby("start_station_name").sum()
stations_2017 = rides_in_2017_grouped.index
rides_in_2018_grouped = rides_in_2018.groupby("start_station_name").sum()
stations_2018 = rides_in_2018_grouped.index
common_stations = sorted(list( set(stations_2017) & set(stations_2018) ))
stations_only_2017 = sorted(list( set(stations_2017) - set(stations_2018) ))
stations_only_2018 = sorted(list(set(stations_2018) - set(stations_2017)))

print(f"Total number of stations in 2017: {len(stations_2017)}\n")
print(f"Total number of stations in 2018: {len(stations_2018)}\n")
print(f"Common stations: {len(common_stations)}\n",common_stations,"\n")
print(f"Stations removed in 2018: {len(stations_only_2017)}\n", stations_only_2017,"\n")
print(f"Stations new in 2018: {len(stations_only_2018)}\n", stations_only_2018,"\n")


# Now we know which stations have changed between 2017 and 2018. 
# 
# 
# We now inspect the other table in the database, containing information on the individual stations. 
# Unfortunateely it is not specified when the table was updated. 

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'station_info', '\nSELECT name, status, latitude, longitude\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_stations`\nORDER BY name')


# In[ ]:


station_info.head()


# In[ ]:


stations_with_info = list(station_info["name"])
stations_with_info_common = sorted(list( set(common_stations) & set(stations_with_info) ))
stations_without_info_common = sorted(list( set(common_stations) - set(stations_with_info) ))
stations_with_info_only_2017 = sorted(list( set(stations_only_2017) & set(stations_with_info) ))
stations_without_info_only_2017 = sorted(list( set(stations_only_2017) - set(stations_with_info) ))
stations_with_info_only_2018 = sorted(list( set(stations_only_2018) & set(stations_with_info) ))
stations_without_info_only_2018 = sorted(list( set(stations_only_2018) - set(stations_with_info) ))

print(f"Number of stations for which we have info: {len(stations_with_info)} \n")
print(f"Common stations with info: {len(stations_with_info_common)} \n", stations_with_info_common, "\n")
print(f"Common stations without info: {len(stations_without_info_common)} \n", stations_without_info_common, "\n")
print(f"Stations with info that weree removed in 2018: {len(stations_with_info_only_2017)}\n", stations_with_info_only_2017,"\n")
print(f"Stations with info that were new in 2018: {len(stations_with_info_only_2018)}\n", stations_with_info_only_2018,"\n")


# Now let's try to obtain all of this information from SQL only. 
# I'm going to select only the stations common in 2017 and 2018 and with status 'active'

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'rides_common_stations_test', "\nWITH station_info AS (\n    SELECT name, status, latitude, longitude\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_stations`\n    WHERE status = 'active'\n    ORDER BY name\n), \nstations_2017 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2017-01-01' and '2018-01-01'\n    GROUP BY start_station_name\n),\nstations_2018 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2018-01-01' and '2019-01-01'\n    GROUP BY start_station_name\n),\ncounts AS (\n    SELECT COUNT(1) as num_rides, start_station_name, \n        TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2018-01-01' AND '2018-03-01' # check a few months\n    GROUP BY start_hour, start_station_name    \n),\ncommon_stations AS (\n    SELECT stations_2018.start_station_name \n    FROM stations_2018 INNER JOIN stations_2017 ON stations_2018.start_station_name = stations_2017.start_station_name\n), \ncommon_stations_info AS(\n    SELECT common_stations.start_station_name, station_info.latitude, station_info.longitude\n    FROM station_info INNER JOIN common_stations on station_info.name = common_stations.start_station_name\n)\nSELECT counts.start_station_name, counts.start_hour, common_stations_info.latitude as latitude, \n    common_stations_info.longitude as longitude, counts.num_rides as num_rides\nFROM counts INNER JOIN common_stations_info ON counts.start_station_name = common_stations_info.start_station_name\nORDER BY num_rides")


# In[ ]:


rides_common_stations_test.tail()


# ## Model creation

# We create the model and train it on 2017 data. We include also latitude and longitude of the station in the training. 

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\nCREATE OR REPLACE MODEL`model_dataset.bike_trips`\nOPTIONS(model_type='linear_reg') AS \nWITH station_info AS (\n    SELECT name, status, latitude, longitude\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_stations`\n    WHERE status = 'active'\n    ORDER BY name\n), \nstations_2017 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2017-01-01' and '2018-01-01'\n    GROUP BY start_station_name\n),\nstations_2018 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2018-01-01' and '2019-01-01'\n    GROUP BY start_station_name\n),\ncounts AS (\n    SELECT COUNT(1) as num_rides, start_station_name, \n        TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2017-01-01' AND '2018-1-01' # train on 2017 data\n    GROUP BY start_hour, start_station_name    \n),\ncommon_stations AS (\n    SELECT stations_2018.start_station_name \n    FROM stations_2018 INNER JOIN stations_2017 ON stations_2018.start_station_name = stations_2017.start_station_name\n), \ncommon_stations_info AS(\n    SELECT common_stations.start_station_name, station_info.latitude, station_info.longitude\n    FROM station_info INNER JOIN common_stations on station_info.name = common_stations.start_station_name\n)\nSELECT counts.start_station_name, counts.start_hour, counts.num_rides as label, common_stations_info.latitude as latitude, \n    common_stations_info.longitude as longitude \nFROM counts INNER JOIN common_stations_info ON counts.start_station_name = common_stations_info.start_station_name")


# ## Model evaluation
# 
# The performance of the model is evaluated on 2018 data

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\nSELECT\n  *\nFROM ML.EVALUATE(MODEL `model_dataset.bike_trips`, (\nWITH station_info AS (\n    SELECT name, status, latitude, longitude\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_stations`\n    WHERE status = 'active'\n    ORDER BY name\n), \nstations_2017 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2017-01-01' and '2018-01-01'\n    GROUP BY start_station_name\n),\nstations_2018 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2018-01-01' and '2019-01-01'\n    GROUP BY start_station_name\n),\ncounts AS (\n    SELECT COUNT(1) as num_rides, start_station_name, \n        TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2018-01-01' AND '2019-1-01' # evaluate on 2018 data\n    GROUP BY start_hour, start_station_name    \n),\ncommon_stations AS (\n    SELECT stations_2018.start_station_name \n    FROM stations_2018 INNER JOIN stations_2017 ON stations_2018.start_station_name = stations_2017.start_station_name\n), \ncommon_stations_info AS(\n    SELECT common_stations.start_station_name, station_info.latitude, station_info.longitude\n    FROM station_info INNER JOIN common_stations on station_info.name = common_stations.start_station_name\n)\nSELECT counts.start_station_name, counts.start_hour, counts.num_rides as label, common_stations_info.latitude as latitude, \n    common_stations_info.longitude as longitude \nFROM counts INNER JOIN common_stations_info ON counts.start_station_name = common_stations_info.start_station_name\n    \n))")


# The performance is already improved with respect to the original exercise, but still not completely satisfactory

# ## Looking at predictions
# 
# We can look at the predictions that our model makes for 2018. Remember that for now we are only looking at the stations that were present both in 2018 and 2017, and ignore the new stations in 2018. 

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'predict_2018', "\nSELECT \n    TIMESTAMP_TRUNC(start_hour, DAY) as start_day,    \n    predicted_label as predicted_num_rides,\n    label as num_rides, \n    start_station_name, \n    latitude, \n    longitude\nFROM ML.PREDICT( MODEL `model_dataset.bike_trips`,(\nWITH station_info AS (\n    SELECT name, status, latitude, longitude\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_stations`\n    WHERE status = 'active'\n    ORDER BY name\n), \nstations_2017 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2017-01-01' and '2018-01-01'\n    GROUP BY start_station_name\n),\nstations_2018 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2018-01-01' and '2019-01-01'\n    GROUP BY start_station_name\n),\ncounts AS (\n    SELECT COUNT(1) as num_rides, start_station_name, \n        TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2018-01-01' AND '2019-1-01' # evaluate on 2018 data\n    GROUP BY start_hour, start_station_name    \n),\ncommon_stations AS (\n    SELECT stations_2018.start_station_name \n    FROM stations_2018 INNER JOIN stations_2017 ON stations_2018.start_station_name = stations_2017.start_station_name\n), \ncommon_stations_info AS(\n    SELECT common_stations.start_station_name, station_info.latitude, station_info.longitude\n    FROM station_info INNER JOIN common_stations on station_info.name = common_stations.start_station_name\n)\nSELECT counts.start_station_name, counts.start_hour, counts.num_rides as label, common_stations_info.latitude as latitude, \n    common_stations_info.longitude as longitude \nFROM counts INNER JOIN common_stations_info ON counts.start_station_name = common_stations_info.start_station_name\n))\nORDER BY start_day, start_station_name\n ")


# In[ ]:


predict_2018.head()


# In[ ]:


predict_2018_by_station = predict_2018.groupby("start_station_name").sum()#agg({"num_rides":"sum", "predicted_num_rides":"sum",
                                                                            #"latitude":"mean","longitude":"mean"})
predict_2018_by_station[["predicted_num_rides","num_rides"]].plot()
predict_2018_by_station['rel-diff'] = (predict_2018_by_station["predicted_num_rides"] - predict_2018_by_station["num_rides"])/predict_2018_by_station["predicted_num_rides"]
print("Rows with largest relative difference\n",predict_2018_by_station.sort_values('rel-diff').head(),"\n")
print("Average values\n", predict_2018_by_station.mean(),"\n")


# The model seems to do reasonably well except for two stations where the difference is much larger: `Guadalupe & 21st` and `UT West Mall @ Guadalupe`. Let's focus on the first one to check if this difference is constant in time. 

# In[ ]:


predict_2018_guadalupe21 = predict_2018[predict_2018['start_station_name'] == 'Guadalupe & 21st']
predict_2018_guadalupe21[["predicted_num_rides","num_rides","start_day"]].groupby("start_day").sum().plot()


# It looks like the difference starts being large between February and March 2018. 

# ## Geographical location of the stations
# 
# It looks like the two stations that have the largest difference between predicted and observed number of rides are close geographically (they have similar values of latitude and longitude). 
# It's interesting to see if they are also close to the new stations that have been added in 2018. This could be related to an event attracting many people in the area or an initiative of the bike rental company. 

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'new_stations_2018', "\nWITH station_info AS (\n    SELECT name, status, latitude, longitude\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_stations`\n    WHERE status = 'active'\n    ORDER BY name\n), \nstations_2017 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2017-01-01' and '2018-01-01'\n    GROUP BY start_station_name\n),\nstations_2018 AS(\n    SELECT start_station_name\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    WHERE start_time BETWEEN '2018-01-01' and '2019-01-01'\n    GROUP BY start_station_name\n),\nonly_2018 AS (\n    SELECT stations_2018.start_station_name \n    FROM stations_2018 LEFT JOIN stations_2017 ON stations_2018.start_station_name = stations_2017.start_station_name\n    WHERE stations_2017.start_station_name is null \n)\nSELECT only_2018.start_station_name, station_info.latitude, station_info.longitude\nFROM only_2018 INNER JOIN station_info on only_2018.start_station_name = station_info.name")


# In[ ]:


new_stations_2018


# Let's look at this in a plot (this could be done better with geopandas, but for the moment I just want to take a quick look)

# In[ ]:


ax1 = station_info.plot.scatter(y='latitude', x='longitude', color='blue', label='all stations')
new_stations_2018.plot.scatter(y='latitude', x='longitude', ax=ax1, color='green', label='2018 new stations')
station_info[station_info.name.str.contains('Guadalupe & 21st|UT West Mall @ Guadalupe')].plot.scatter(y='latitude', x='longitude', color='red', 
                                                      label='bad prediction', ax=ax1)


# Yes, the two stations with the highest disagreement between predicted and observed rides are close to the new stations. This suggest that something happened in the area. 

# ## Conclusion
# 
# After a google search for what happened in Austin between February and March 2018, I found that 
# in February 2018 Austin B-cycle announced  an expansion of the service to UT Campus and West Campus Neighborhoods. 
# This included new stations and also an annual membership free of charge for the UT students (all the details can be found [here](https://www.bcycle.com/news/2018/02/14/austin-b-cycle-expands-service-to-ut-campus-and-west-campus-neighborhoods). 
# 
# The data from 2018 was so hard to predict because the combination of new stations and free membership for the students lead to a massive increase in the number of rides, which was impossible to predict with 2017 data. 
# 
# A person working for the bike sharing company would have for sure know this, and could have included in the model additional information. For example data related to other cities where this offer has been proposed before, the number of students who registered for the free membership, and so on. 
