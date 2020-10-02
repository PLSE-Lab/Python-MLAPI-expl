#D1 import package with helper functions 
import bq_helper

#D1 create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

#D1 print all the tables in this dataset (there's only one!)
open_aq.list_tables()

#D1 print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

#1 query to select all the items from the "city" column where the "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
#2 the query_to_pandas_safe will only return a result if it's less than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)

#3 What five cities have the most measurements taken there?
us_cities.city.value_counts().head()

#4
query1 = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

#5
non_ppm_cities = open_aq.query_to_pandas_safe(query1)

#6
non_ppm_cities.city.value_counts().head()

#D1 save our dataframe as a .csv 
non_ppm_cities.to_csv("non_ppm_cities.csv")



