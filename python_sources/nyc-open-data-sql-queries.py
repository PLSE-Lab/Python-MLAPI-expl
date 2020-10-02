#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Any results you write to the current directory are saved as output.
import bq_helper
import matplotlib.pyplot as plt
ny_data_set = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")


# ## Sample queries from NYC Open Data
# The following are sample queries usig NYC Open Data to demonstrate basic understanding of SQL using an open platform. 

# In[ ]:


maple_query = """
WITH tree_list AS
(SELECT spc_common
FROM `bigquery-public-data.new_york.tree_census_2015`
INTERSECT DISTINCT
SELECT spc_common
FROM `bigquery-public-data.new_york.tree_census_2015`)
SELECT COUNT(*) AS total_number_tree_species
FROM tree_list 
"""
maples = ny_data_set.query_to_pandas_safe(maple_query)
maples


# In[ ]:


# MAPLE SYRUP problem - Find the number of red, black, and sugar maple trees in NYC.
maple_query = """
SELECT 
    spc_common AS tree_type,
    COUNT(spc_common) AS total_count
FROM `bigquery-public-data.new_york.tree_census_2015`
WHERE
    status != 'Dead' AND
    spc_common IN ('sugar maple' ,'black maple', 'red maple')
GROUP BY spc_common
"""
maples = ny_data_set.query_to_pandas_safe(maple_query)
maples


# So the red maple is clearly most common in NYC. These maple types can produce maple syurp, if they are fairly close together one could potentially bottle the syurp. 
# 
# Now let's amend this to find the total the sugar and black maples compared to the number of red maples. 

# In[ ]:


maple_query_2 = """
WITH maple_count AS 
    (
    SELECT spc_common AS tree_type,
           COUNT(spc_common) AS count
    FROM   `bigquery-public-data.new_york.tree_census_2015`
    WHERE  status != 'Dead' 
           AND spc_common IN ('sugar maple', 'black maple', 'red maple')
    GROUP BY spc_common
    ) 
SELECT SUM(count) AS total,
CASE
    WHEN tree_type IN ('black maple', 'sugar maple') 
    THEN 'black & sugar maple'
    ELSE 'red maple'
END AS sub_class
FROM maple_count
GROUP BY sub_class
"""
maples_2 = ny_data_set.query_to_pandas_safe(maple_query_2)
maples_2


# ## Pedestrian Fatalities from Motor Vehicles

# As an vivid biker I always wonder about pedestrian fatalities and where they occur the most. I'll start by looking at which boroughs have a high level of fatalities by filtering for high monthly and yearly numbers by borough. 

# In[ ]:


# Most dangerous borough for pedestrians since 2015 
peds_query = """
WITH totals AS 
(
    SELECT  borough,
            SUM(number_of_pedestrians_killed) AS total_deaths,
            EXTRACT(MONTH FROM timestamp) AS month,
            EXTRACT(YEAR FROM timestamp) AS year
    FROM `bigquery-public-data.new_york.nypd_mv_collisions`
    WHERE borough NOT IN ('')
    GROUP BY borough, year, month
    ORDER BY year, month
),
averages AS 
(
    SELECT  borough, month, year, total_deaths, 
            AVG(total_deaths) over (partition by year) AS yearly_average,
            ROUND( AVG(total_deaths) over (partition by month),2) AS monthly_average
    FROM totals
)
SELECT borough, month,
       year, total_deaths, 
       monthly_average, yearly_average
FROM   averages
WHERE  total_deaths > 3*monthly_average
       AND monthly_average > yearly_average
ORDER BY year, month
"""
ped_death = ny_data_set.query_to_pandas_safe(peds_query)
ped_death


# Brooklyn in 2015 and 2017 seems pretty high, so does Manhattan in 2017. Next let's look at total deaths by borough. 

# In[ ]:


peds_query_2 = """
SELECT  borough,
        SUM(number_of_pedestrians_killed) AS total_deaths
FROM `bigquery-public-data.new_york.nypd_mv_collisions`
GROUP BY borough
"""

ped_death = ny_data_set.query_to_pandas_safe(peds_query_2)
ped_death


# So many unassigned boroughs - how do the police not know what borough they are in?
# 
# Let's see if we can understand where these unmarked fatalities are coming from. 

# In[ ]:


peds_query_two = """
SELECT  borough, zip_code, 
        on_street_name, off_street_name, 
        cross_street_name, latitude, longitude
FROM `bigquery-public-data.new_york.nypd_mv_collisions`
WHERE EXTRACT(YEAR FROM timestamp) > 2015 
        AND borough IN ('')
        AND number_of_pedestrians_killed > 0
        LIMIT 20
"""
ped_death_two = ny_data_set.query_to_pandas_safe(peds_query_two)
ped_death_two


# It looks like it is possible to determine many of the boroughs based on the given data. Using GPS position, or street names could yield almost all of the unassigned borough fields. It is baffling how on a police report for a pedestrain death no one can be bothered to fill out the street name, or borough. 

# In[ ]:


#All the noise complaints on Irving Place. 
address_query = """
                 WITH bus_noise AS
                    (
                        SELECT  incident_address,
                                complaint_type 
                        FROM    `bigquery-public-data.new_york.311_service_requests`
                        WHERE   complaint_type like '%Noise%' 
                        
                    )
                 SELECT count(complaint_type) AS count, incident_address AS address
                 FROM bus_noise
                 WHERE incident_address like '%IRVING PLACE%'
                 GROUP BY address
                 HAVING count >= 5
                 ORDER BY count DESC
                """
noise = ny_data_set.query_to_pandas_safe(address_query)
noise


# I really wanted 1 Irving Place to have the most, but it seems that someone is very good about calling 311 around 56/55 Irving Place though I do not think that address is nearly as noisy as 1 Irving Place. There are three bars along the street at 53-56 Irving Place, one of which usually has a line of people in front (it's a pseudo-secret bar), so I could see why those addresses might get several 311 calls. 
