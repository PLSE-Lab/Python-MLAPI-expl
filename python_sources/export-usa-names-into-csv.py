# import package with helper functions 
import bq_helper

# create a helper object for this dataset
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

# query and export data 
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
agg_names = usa_names.query_to_pandas_safe(query)
agg_names.to_csv("usa_names_agg.csv")





