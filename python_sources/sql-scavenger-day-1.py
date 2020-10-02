import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name="openaq")
                                    
open_aq.list_tables()

open_aq.head("global_air_quality")

query = """SELECT city 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
            """
us_cities = open_aq.query_to_pandas_safe(query)

us_cities.city.value_counts().head()


query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
            """
            
not_ppm = open_aq.query_to_pandas_safe(query1)
not_ppm.country.value_counts().head()

query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
            """
            
exactly_zero = open_aq.query_to_pandas_safe(query2)

exactly_zero.pollutant.value_counts().head()