# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# sample query from:
# https://cloud.google.com/bigquery/public-data/openaq#which_10_locations_have_had_the_worst_air_quality_this_month_as_measured_by_high_pm10
QUERY = """
        SELECT location, city, country, value, timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = "pm10" AND timestamp > "2017-04-01"
        ORDER BY value DESC
        LIMIT 1000
        """
        
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')

# print all the tables in this dataset (there's only one!)
bq_assistant.list_tables()

df = bq_assistant.query_to_pandas(QUERY)
df.head(3)

QUERY2 = """
        SELECT distinct country
        FROM `bigquery-public-data.openaq.global_air_quality`
        where unit != 'ppm'
        """
df2 = bq_assistant.query_to_pandas(QUERY2)
df2.head(100)

bq_assistant.estimate_query_size(QUERY2)

QUERY2a = """
        SELECT distinct country
        FROM `bigquery-public-data.openaq.global_air_quality`
        """
bq_assistant.estimate_query_size(QUERY2a)

QUERY2b = """
        SELECT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        """

bq_assistant.estimate_query_size(QUERY2b)

QUERY4 = """select * from `bigquery-public-data.openaq.global_air_quality`"""
bq_assistant.estimate_query_size(QUERY4)

QUERY3 = """
        SELECT distinct pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        where value != 0
        order by pollutant
        """
df3 = bq_assistant.query_to_pandas(QUERY3)
df3.head(100)