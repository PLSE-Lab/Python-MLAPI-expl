#import bq_helper Package
import bq_helper

# create a helper object for Open Air Quality dataset
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "openaq")
                                       
# print a list of all the tables in the open_aq dataset
open_aq.list_tables()

open_aq.head("global_air_quality")

# this query looks in the full table in the open_aq
# dataset, then gets the country column from every row where 
# the type column has "ppm" in it.
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
            
# Executing the query
openaq_countries = open_aq.query_to_pandas_safe(query)

# Saving the results in a csv file
openaq_countries.to_csv("countries_without_PPM.csv")

# craeting the results as a list
countries_without_ppm = openaq_countries.country.tolist()

# Printing the lists
print('There are {} countries without unit PPM: '
      .format(len(countries_without_ppm)))
print(countries_without_ppm)

query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
        
# Executing the query
openaq_poll_val_0 = open_aq.query_to_pandas_safe(query)

# Saving the results in a csv file
openaq_poll_val_0.to_csv("pollutant_val_zero.csv")

# craeting the results as a list
pollutant_with_zero= openaq_poll_val_0.pollutant.tolist()

# Printing the lists
print('There are {} pollutant which has the value zero : '
      .format(len(pollutant_with_zero)))
print(pollutant_with_zero)