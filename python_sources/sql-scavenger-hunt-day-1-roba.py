import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "openaq")
# query for selecting the countries where the unit of measurement is not ppm
query1 = """SELECT DISTINCT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        """
country_unit = open_aq.query_to_pandas_safe(query1)
print(country_unit)
# query for selecting the pollutants where the value is 0
query2 = """SELECT DISTINCT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """
pollutant_value = open_aq.query_to_pandas_safe(query2)
print(pollutant_value)