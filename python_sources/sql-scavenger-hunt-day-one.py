import pandas as pd
from google.cloud import bigquery
import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
#open_aq.list_tables()
#open_aq.head("global_air_quality")

#Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value isn't something, use "!=")

query1="""SELECT distinct country 
          FROM `bigquery-public-data.openaq.global_air_quality`
          where unit != 'ppm'
          """
open_aq.estimate_query_size(query1)
open_aq.query_to_pandas(query1)

#Which pollutants have a value of exactly 0?

query2="""SELECT city,country,pollutant
          FROM `bigquery-public-data.openaq.global_air_quality`
          where value=0
          group by city,country,pollutant"""
          
open_aq.estimate_query_size(query2)
open_aq.query_to_pandas(query2)
