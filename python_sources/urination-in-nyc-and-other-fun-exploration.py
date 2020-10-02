#!/usr/bin/env python
# coding: utf-8

# This is a bit of fun I had messing around with the NYC data along with some weather correlation. Thanks Kaggle team for pulling this together for us to use.

# **Public urination...**
# 
# This section pulls together complaints about public urination in NYC to show the correlation of higher complaint counts to nice weather and lower compliant counts to cold weather.
# 
# 1) Prep the BigQuery connectors.

# In[1]:


import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import matplotlib.pyplot as plt
import os
import bq_helper

# Connect to BigQuery datasets
ny_data_set = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "new_york")
noaa_data_set = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                         dataset_name = "noaa_gsod")

#noaa_data_set.list_tables()
#ny_data_set.table_schema("311_service_requests")
#ny_data_set.head("311_service_requests")
#ny_data_set.head("311_service_requests",selected_columns="location", num_rows=10)


# 2) Retrieve the NYC 311 data and pivot it so there is a column for every complaint type.

# In[2]:


# Define query 
query = """
SELECT 
 Extract(DATE from created_date) AS creation_date, 
 REPLACE(UPPER(complaint_type), "HEATING", "HEAT/HOT WATER") as complaint_type, 
 COUNT(*) AS count 
FROM        `bigquery-public-data.new_york.311_service_requests` 
WHERE
 Extract(YEAR from created_date) = 2016
GROUP BY creation_date, complaint_type 
ORDER BY creation_date ASC, count DESC 
"""
#ny_data_set.estimate_query_size(query)

# Run query 
complaint_counts = ny_data_set.query_to_pandas_safe(query, max_gb_scanned=0.5)

# Pivot complaint data to create new columns for all of the complaint types 
complaint_counts = complaint_counts.pivot(index='creation_date', columns='complaint_type', values='count')
complaint_counts.columns = [c.lower()
                            .replace(' ', '_')
                            .replace('-', '_') 
                            .replace('/', '_') 
                            for c in complaint_counts.columns]
# Fill zeros for missing values
complaint_counts = complaint_counts.fillna(0)
# Reset index to numeric values for later trending
complaint_counts["creation_date"] = complaint_counts.index
complaint_counts.index = range(len(complaint_counts.index))

#print(complaint_counts.head())


# 3) Retrieve weather data for New York.

# In[26]:


# Define query 
query = """
SELECT 
 CAST(CONCAT(w.year,'-',w.mo,'-',w.da) AS date) AS date,
 AVG(w.temp) AS avg_temp
FROM        `bigquery-public-data.noaa_gsod.gsod2016`  w
INNER JOIN  `bigquery-public-data.noaa_gsod.stations`  s
 ON w.stn=s.usaf
 AND w.wban=s.wban
WHERE
 s.country='US'
 AND s.state = 'NY'
GROUP BY date
ORDER BY date
"""
#noaa_data_set.estimate_query_size(query)

# Run query 
weather_by_day = noaa_data_set.query_to_pandas_safe(query, max_gb_scanned=0.5)

#print(weather_by_day.head(365))


# 4) Plot two independent axes both displayed on the x-axis.

# In[38]:


# Create first axis
color = 'tab:orange'
X = complaint_counts.index
y = complaint_counts.urinating_in_public
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.set_xlabel('time (days)')
ax1.set_ylabel('urine', color=color)
ax1.plot(X, y, color=color)
coefs = poly.polyfit(X, y, 4)
ffit = poly.polyval(X, coefs)
ax1.plot(X, ffit, dashes=[6, 2], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create second axis
color = 'tab:green'
X = weather_by_day.index
y = weather_by_day.avg_temp
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('temp', color=color)
ax2.plot(y, color=color)
coefs = poly.polyfit(X, y, 4)
ffit = poly.polyval(X, coefs)
ax2.plot(X, ffit, dashes=[6, 2], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Display plot
plt.show()


# 5) Determine the Pearson correlation coefficient between hot weather and higher urinating in public.

# In[5]:


correlation = np.corrcoef(complaint_counts.urinating_in_public, 
                          weather_by_day.avg_temp)[0, 1]
print("Correlation coefficient: {}".format(correlation))


# **Complaint types most affected by weather**
# 
# Realizing that there is a correlation between public urination and weather made me want to take this a step further and discover the top 311 complaint types overall that are affected by weather.

# 1) Calculate heating and cooling "degree days" to normalize the temperature into something a little easier to compare.

# In[37]:


# Calculate heating and cooling degree days 
change_point = 65
weather_by_day['raw_degree_day_calc'] = change_point - weather_by_day['avg_temp']
weather_by_day['HDD'] = abs(weather_by_day.loc[weather_by_day.raw_degree_day_calc>0,'raw_degree_day_calc'])
weather_by_day['HDD'] = weather_by_day['HDD'].fillna(0)
weather_by_day['CDD'] = abs(weather_by_day.loc[weather_by_day.raw_degree_day_calc<0,'raw_degree_day_calc'])
weather_by_day['CDD'] = weather_by_day['CDD'].fillna(0)
weather_by_day['total_degree_days'] = weather_by_day['HDD'] + weather_by_day['CDD']
#print(weather_by_day.head(200))


# 2) Chart the degree days to see how they turned out.

# In[39]:


plt.figure(figsize=(12, 5))
plt.plot(weather_by_day.HDD, color='tab:red')
plt.plot(weather_by_day.CDD, color='tab:blue')
plt.plot(weather_by_day.avg_temp, color='tab:green')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
plt.show()


# **Other ideas for future exploration:**
# - top 10 complaint types for 311 that are affected by weather (either pure temp or degree days)
# - find cases of repeated disorderly youth complaints in 311 to find grumpy people
# - buildings with sustained and extreme noise issues
