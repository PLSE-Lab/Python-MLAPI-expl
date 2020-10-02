#!/usr/bin/env python
# coding: utf-8

# Lead is a useful metal, but is a neurotoxin and  can be especially harmful to children ([Wikipedia](https://en.wikipedia.org/wiki/Lead)). The 2008 (renewed 2016) limit for airborne Lead is 15 micrograms per cubic meter of air averaged over a three month period ( [current EPA Airborne Lead Standard](https://www.epa.gov/lead-air-pollution/national-ambient-air-quality-standards-naaqs-lead-pb) ), 
# 
# Let's take a look at the Lead levels in the air around Los Angeles for the EPA Historical Air Quality data set.
# 
# Note: I use "pb" as part of column names throughout this analysis - Pb is the atomic symbol for Lead

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
epa_aq_helper = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="epa_historical_air_quality")


# I'm not sure what the data looks like, so let's get some basic dimensions of the data figured out. I'm asking BigQuery to run the numbers, so the summary will be quick.

# In[ ]:


pb_dimensions_query = """
    SELECT
        COUNT(DISTINCT method_name),
        COUNT(DISTINCT units_of_measure) AS uom_count,
        MIN(arithmetic_mean) AS min_mean,
        MAX(arithmetic_mean) AS max_mean,
        MIN(EXTRACT(YEAR FROM date_local)) AS min_year,
        MAX(EXTRACT(YEAR FROM date_local)) AS max_year,        
        COUNT(DISTINCT city_name) AS city_count        
    FROM
        `bigquery-public-data.epa_historical_air_quality.lead_daily_summary`
    """
pb_dimensions_df = epa_aq_helper.query_to_pandas(pb_dimensions_query)
print(pb_dimensions_df)


# Let's do the same, but for Los Angeles. We'll expand out method_name and units_of_measure just to see what is there.

# In[ ]:


la_pb_dimensions_query = """
    SELECT
        method_name,
        units_of_measure,
        MIN(arithmetic_mean) AS min_mean,
        MAX(arithmetic_mean) AS max_mean,
        MIN(extract(YEAR FROM date_local)) AS min_year,
        MAX(extract(YEAR FROM date_local)) AS max_year,        
        COUNT(city_name) AS point_count
    FROM
        `bigquery-public-data.epa_historical_air_quality.lead_daily_summary`
    WHERE 
      city_name = "Los Angeles"
      AND state_name = "California"
    GROUP BY
        method_name, units_of_measure
    ORDER BY
        min_year
    """
la_pb_dimensions_df = epa_aq_helper.query_to_pandas(la_pb_dimensions_query)
print(la_pb_dimensions_df)


# Looks like the unit_of_measure is fine, and as expected, there were different ways of measuring airborne Lead over the time period. If this was a industrial, academic, or governmental analysis, we'd have to confirm with an analytical chemist that the data is equivalent across measurement methods. We'll assume they are ok to compare and merge, and move on for now.
# 
# Let's take a look at how Lead in the air in Los Angeles has changed over time.

# In[ ]:


pb_la_yr_query = """
    SELECT
        EXTRACT(YEAR FROM date_local) AS year,
        arithmetic_mean AS pb_ug_per_m3
    FROM
      `bigquery-public-data.epa_historical_air_quality.lead_daily_summary`
   WHERE
     city_name = "Los Angeles"
      AND state_name = "California"
        """
pb_la_yr_df = epa_aq_helper.query_to_pandas(pb_la_yr_query)
pb_la_yr_df.boxplot(by='year', column='pb_ug_per_m3')


# Wow, didn't expect that bump in Lead levels between 2010 and 2013. Let's take a closer look. Since we've already got the data, we'll just use Python to pull out that section and replot.

# In[ ]:


pb_2008_2017 = pb_la_yr_df[pb_la_yr_df["year"] > 2007]
pb_2008_2017.boxplot(by='year', column='pb_ug_per_m3');


# How close did LA come to EPA guideline violation? Let's pull out the monthly data for the range in question...

# In[ ]:


pb_month_query = """
    SELECT
        EXTRACT(YEAR FROM date_local) AS year,
        EXTRACT(MONTH FROM date_local) AS month,
        arithmetic_mean AS pb_ug_per_m3
    FROM
      `bigquery-public-data.epa_historical_air_quality.lead_daily_summary`
    WHERE
      city_name = "Los Angeles"
      AND state_name = "California"
      AND EXTRACT(YEAR FROM date_local) BETWEEN 2008 AND 2013
    ORDER BY year ASC, month ASC
        """
pb_month_df = epa_aq_helper.query_to_pandas(pb_month_query)


# Let's create an artificial index spanning the years and months, and then use pandas calculate the rolling average over the three month period, then plot the result.

# In[ ]:


pb_month_df['ymidx'] = (pb_month_df['year']-2008)*12+pb_month_df['month']
pb_month_df.sort_values(by='ymidx')

pb_month_df["rollingavg"] = pb_month_df['pb_ug_per_m3'].rolling(window=3).mean()
pb_month_df.plot(x='ymidx',y='rollingavg',kind='line')


# So the EPA standards were not broken, but there were some significant peaks during that period. Here is the 2013 drop off.

# In[ ]:


pb_2013_wk_query = """
    SELECT
        EXTRACT(WEEK FROM date_local) AS week,
        arithmetic_mean AS pb_ug_per_m3
    FROM
      `bigquery-public-data.epa_historical_air_quality.lead_daily_summary`
    WHERE
      city_name = "Los Angeles"
      AND state_name = "California"
      AND EXTRACT(YEAR FROM date_local) = 2013
    ORDER BY week
        """
pb_2013_wk_df = epa_aq_helper.query_to_pandas(pb_2013_wk_query)
pb_2013_wk_df.boxplot(by='week',column='pb_ug_per_m3');


# Maybe these LA Times articles have something to do with it [1](http://articles.latimes.com/print/2013/sep/18/local/la-me-exide-20130919) [2](http://timelines.latimes.com/exide-technologies-history/) , but we'd have to get airborne Lead monitoring for the affected ineighborhood to make sure. As we all know, correlation does not imply causation.
# 
# What other interesting data can you find in this set?
# 
