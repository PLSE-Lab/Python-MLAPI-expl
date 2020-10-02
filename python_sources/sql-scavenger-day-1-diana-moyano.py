#Import package with helper functions
import bq_helper
#create a helper object for this dataset
open_aq= bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
#print all the tables in this dataset
open_aq.list_tables()
#print first rows to see how it is constucted
open_aq.head("global_air_quality")


#1.Which countries use a unit other than ppm to measure any type of pollution? 
#(Hint: to get rows where the value isn't something, use "!=")
Query1=""" SELECT DISTINCT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit<>"ppm"
"""
country_noppm=open_aq.query_to_pandas_safe(Query1)
country_noppm.country.value_counts()

#The countries that use a unit other than ppm are FR, ES, DE, US, AT, CZ, Etc
country_noppm.to_csv("country_noppm")

#2. Which pollutants have a value of exactly 0?
QueryP=""" SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY pollutant
HAVING sum(value)<0.001
"""
poll_zero=open_aq.query_to_pandas_safe(QueryP)
poll_zero.pollutant.value_counts()
#The only pollutant that has a value of exactly 0 is so2
poll_zero.to_csv("poll_zero")
