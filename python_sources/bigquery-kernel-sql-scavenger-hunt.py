# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# create a helper object for our bigquery dataset
aq_data = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "openaq")

# print a list of all the tables in the hacker_news dataset
print (aq_data.list_tables())

# print information on all the columns in the "full" table
# in the hacker_news dataset
print (aq_data.table_schema("global_air_quality"))

# preview the first couple lines of the "full" table
print (aq_data.head("global_air_quality"))

# preview the first ten entries in the country column of the global_air_quality table
print (aq_data.head("global_air_quality", selected_columns="country", num_rows=10))

# this query looks in the full table in the openaq
# dataset, then gets the country column from every row where 
# the unit column has "µg/m³" in it.
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = "µg/m³" """

# check how big this query will be
# (The query size is returned in gigabytes.)
print ( (aq_data.estimate_query_size(query) )

# query to pandas - only run this query if it's less than 100 MB
pandas1 = aq_data.query_to_pandas_safe(query, max_gb_scanned=0.1)

# query to pandas - check out the country of unit postings (if the query is smaller than 1 gig)
country = aq_data.query_to_pandas_safe(query)

# Query 2
query2 = """SELECT averaged_over_in_hours
            FROM `bigquery-public-data.openaq.global_air_quality` """

# check how big this query2 will be
# (The query size is returned in gigabytes.)
print(aq_data.estimate_query_size(query2))

# query2 to pandas
average_aq = aq_data.query_to_pandas_safe(query2)

# average of all the averaged_over_in_hours for query2
average_aq.averaged_over_in_hours.mean()

