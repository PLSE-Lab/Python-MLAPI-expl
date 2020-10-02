
### Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")

# import our bq_helper package
import bq_helper

#create a helper object for our bigquery dataset
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")

# Query for countries that do not use a ppm unit to measure pollution 

query1 = """select distinct country
           from `bigquery-public-data.openaq.global_air_quality`
           where LOWER(unit) != 'ppm'
           order by country
           """
           
Countries_without_ppm = open_aq.query_to_pandas_safe(query1)

# The countries without ppm
Countries_without_ppm

### Which pollutants have a value of exactly 0?

# Query for pollutants with 0 value 

query2 = """select distinct pollutant
           from `bigquery-public-data.openaq.global_air_quality`
           where value = 0
           order by pollutant
           
           """
Pollutants_with_zero_value = open_aq.query_to_pandas_safe(query2)

# the pollutants with zero value
Pollutants_with_zero_value