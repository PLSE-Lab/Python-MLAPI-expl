import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
d1_q1 = """SELECT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
"""
no_ppm2 = open_aq.query_to_pandas_safe(d1_q1)
country_list = no_ppm2.country.unique()
print(country_list,country_list.size)
d1_q2 = """SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0.0
"""
Pollutants_0 = open_aq.query_to_pandas_safe(d1_q2)
Pollutants_count = Pollutants_0.pollutant.value_counts()
print(Pollutants_count)
