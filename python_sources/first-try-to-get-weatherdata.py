# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import bq_helper 

# create a helper object for our bigquery dataset


weather_hi = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name= "noaa_gsod")
                                       
#weather_hi.list_tables()          

#weather_hi.table_schema("gsod2017")

#weather_hi.list_tables()


query = """ SELECT stn AS stn,mo AS month,da AS day,temp AS temp,stp AS stp,visib AS visib,wdsp AS wdsp,gust AS gust,prcp AS prcp,sndp AS sndp FROM `bigquery-public-data.noaa_gsod.gsod2018` """

weather_hi.estimate_query_size(query)

result = weather_hi.query_to_pandas_safe(query, max_gb_scanned=0.1)

#result.temp.mean()
result.to_csv('weather2018_visib_wdsp.csv', index = False)
print("hello world")


#stations : bigquery-public-data.noaa_gsod.stations usaf, name, country, lat, long

#stationsQuery = """SELECT usaf AS stationenNummer, country AS land, lat As latitude, lon AS longitude FROM `bigquery-public-data.noaa_gsod.stations` WHERE country = 'US' AND lat IS NOT NULL AND lon IS NOT NULL AND NOT (lat = 0.0 AND lon = 0.0) ORDER BY usaf"""

#weather_hi.estimate_query_size(stationsQuery)

#stations = weather_hi.query_to_pandas_safe(stationsQuery, max_gb_scanned=0.1)


#stations.to_csv('WetterStationen_US.csv', index = False)

                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                                       