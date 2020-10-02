
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

import bq_helper 

# create a helper object for our bigquery dataset
bqh = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name= "noaa_gsod")

# build and run a series of queries to get annual temperatures for the US
# WARNING: each year takes 5+ mins to run and the resultant dataset is about 100MB!

START_YEAR = 2009
END_YEAR = 2019

for year in range(START_YEAR, END_YEAR):
    query = "SELECT * FROM `bigquery-public-data.noaa_gsod.gsod{}`".format(year)

    df_wthr = bqh.query_to_pandas_safe(query, max_gb_scanned=5)
    filename = 'US_weather_{}.csv'.format(year)
    df_wthr.to_csv(filename, index = False)
    print ("Saved {}".format(filename))

# build and run a query to get the stations contributing to US weather

stn_query = "SELECT usaf AS stn, name, country, state, lat, lon, elev FROM `bigquery-public-data.noaa_gsod.stations` WHERE country = 'US' ORDER BY stn"

df_stn = bqh.query_to_pandas_safe(stn_query, max_gb_scanned=5)
filename = 'weather_stations.csv'
df_stn.to_csv(filename, index = False)
print ("Saved {}".format(filename))

