#!/usr/bin/env python
# coding: utf-8

# # Loading data and preprocessing
# ## Loading datasets

# In[ ]:


# import most basic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# list input directory
import os
print(os.listdir("../input"))

# prepare datatype for flights.csv to prevent wrong datatypes being assigned
dt = np.dtype([("YEAR", "int64"),                 ("MONTH", "int64"),              ("DAY", "int64"),              ("DAY_OF_WEEK", "int64"),          ("AIRLINE", "object"),
               ("FLIGHT_NUMBER", "int64"),        ("TAIL_NUMBER", "object"),       ("ORIGIN_AIRPORT", "object"),  ("DESTINATION_AIRPORT", "object"), ("SCHEDULED_DEPARTURE", "int64"),
               ("DEPARTURE_TIME", "float64"),     ("DEPARTURE_DELAY", "float64"),  ("TAXI_OUT", "float64"),       ("WHEELS_OFF", "float64"),         ("SCHEDULED_TIME", "float64"),
               ("ELAPSED_TIME", "float64"),       ("AIR_TIME", "float64"),         ("DISTANCE", "int64"),         ("WHEELS_ON", "float64"),          ("TAXI_IN", "float64"),
               ("SCHEDULED_ARRIVAL", "int64"),    ("ARRIVAL_TIME", "float64"),     ("ARRIVAL_DELAY", "float64"),  ("DIVERTED", "int64"),             ("CANCELLED", "int64"),
               ("CANCELLATION_REASON", "object"), ("AIR_SYSTEM_DELAY", "float64"), ("SECURITY_DELAY", "float64"), ("AIRLINE_DELAY", "float64"),      ("LATE_AIRCRAFT_DELAY", "float64"),
               ("WEATHER_DELAY", "float64")])

# load flights delays dataset
airlines = pd.read_csv('../input/flight-delays/airlines.csv')
flights = pd.read_csv('../input/flight-delays/flights.csv', dtype=dt)
airports = pd.read_csv('../input/flight-delays/airports.csv')
# remove flights with wrong airport codes
# airport codes should be IATA (three letter codes), but some are numbers, these cannot be combined with the weather data
flights = flights[flights.ORIGIN_AIRPORT.str.contains("^[A-Z]+$")]
# make date in datetime format from year, month and day
flights['DATE'] = pd.to_datetime(flights[['YEAR','MONTH', 'DAY']])
# load airportCodes
airportCodes = pd.read_csv('../input/airports-train-stations-and-ferry-terminals/airports-extended.csv')[["GKA", "AYGA"]]
airportCodes.columns =  ["IATA_CODE", "ICAO_CODE"]


# ### Convert timeformats
# In the flight delays dataset times are written as four-digit integers as `hhmm`. First alle times get formated to python datetime. Because the random forest does not accept datetime objects, `DEPARTURE_TIME` is created as base10 floating point number. E.g. 10:30 becomes 10,5 etc.

# In[ ]:


import datetime

def getTime(time):
    if pd.isnull(time):
        return np.nan
    else:
        if time == 2400: time = 0
        time = "{0:04d}".format(int(time))
        return datetime.time(int(time[0:2]), int(time[2:4]))
def getTimeNum(inTime):
    
    return time.mktime(getTime(inTime).timetuple())
def combineDateTime(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])

dates = []
times = []
lenght = flights.iloc[[-1]].index.values[0]
for index, cols in flights[['DATE', 'SCHEDULED_DEPARTURE']].iterrows():
    if(index%1000==0): print(str((index*100)/lenght) + "%    ", end="\r")
    if pd.isnull(cols[1]):
        dates.append(np.nan)
    elif float(cols[1]) == 2400:
        cols[0] += datetime.timedelta(days=1)
        cols[1] = datetime.time(0,0)
        dates.append(combineDateTime(cols))
        break
    else:
        cols[1] = getTime(cols[1])
        dates.append(combineDateTime(cols))
    if(not pd.isnull(dates[-1])):
        times.append(dates[-1].time().hour + (dates[-1].time().minute / 60))
    else:
        times.append(dates[-1])
print("100%                                  ")


flights["SCHEDULED_DEPARTURE_TIME"] = times
flights['SCHEDULED_DEPARTURE'] = dates
flights['DEPARTURE_TIME'] = flights['DEPARTURE_TIME'].apply(getTime)
flights['SCHEDULED_ARRIVAL'] = flights['SCHEDULED_ARRIVAL'].apply(getTime)
flights['ARRIVAL_TIME'] = flights['ARRIVAL_TIME'].apply(getTime)


# ## Get weather information
# First the ICAO codes from the `Airports, Train Stations, and Ferry Terminals` dataset are merged into the airports to get from IATA to ICAO

# In[ ]:


airportsNew = pd.merge(airports, airportCodes, on="IATA_CODE", how="left")


# In[ ]:


# testing big query syntax
import bq_helper
from bq_helper import BigQueryHelper

noaa_tables = BigQueryHelper(active_project="bigquery-public-data", dataset_name="noaa_gsod")
#noaa_tables.list_tables()

query1 = """SELECT
    usaf, call
    FROM
        `bigquery-public-data.noaa_gsod.stations`
    WHERE
        `call` LIKE 'KSFO' AND
        CAST(`begin` AS INT64) <= 20150101 AND
        CAST(`end` AS INT64) >= 20151231"""

response1 = noaa_tables.query_to_pandas_safe(query1)
response1


# To get to the weather the ICAO-codes have to be converted to station numbers (`usaf`), which can be found in the `stations` table of the `NOAA GSOD` dataset (=weather dataset)
# 
# Then they are merged into the airport variable to associate the station numbers with the airports

# In[ ]:


icao_codes = airportsNew["ICAO_CODE"]
code_string = "("
for code in icao_codes:
    code_string += "'" + code + "', "
code_string = code_string[:-2]
code_string += ")"
query = "SELECT DISTINCT usaf, call AS `ICAO_CODE` FROM `bigquery-public-data.noaa_gsod.stations` WHERE `call` in " + code_string + " AND CAST(`begin` AS INT64) <= 20150101 AND CAST(`end` AS INT64) >= 20151231"
#print(query)
response = noaa_tables.query_to_pandas(query)
airportsNew = airportsNew.merge(response, on="ICAO_CODE", how="left")


# All weather from 2015 from all station codes in `airportsNew` are put into a new variable `weather`

# In[ ]:


stations = airportsNew["usaf"]
code_string = "("
for code in stations:
    code_string += "'" + code + "', "
code_string = code_string[:-2]
code_string += ")"
query = "SELECT * FROM `bigquery-public-data.noaa_gsod.gsod2015` WHERE `stn` in " + code_string
#print(query)
weather = noaa_tables.query_to_pandas(query)


# In[ ]:


weather["mo"] = pd.to_numeric(weather["mo"])
weather["da"] = pd.to_numeric(weather["da"])
weather = weather.merge(airportsNew[["usaf", "ICAO_CODE", "IATA_CODE"]], left_on="stn", right_on="usaf")


# In[ ]:


weather[:5]


# #### Airports with <300 days of weather recordings: 

# In[ ]:


stations = airportsNew["usaf"]
code_string = "("
for code in stations:
    code_string += "'" + code + "', "
code_string = code_string[:-2]
code_string += ")"
query = "SELECT * FROM (SELECT COUNT(stn) AS `days`, `stn` FROM `bigquery-public-data.noaa_gsod.gsod2015` WHERE `stn` in " + code_string + " GROUP BY `stn`) WHERE `days` < 300"
#print(query)
weatherTest = noaa_tables.query_to_pandas(query)
weatherTest


# * 723020: Wilmington International Airport
# * 785260: San Juan Intl Airport
# * 740035: Yuma MCAS
# * 726764: Yellowstone Airport

# # Getting to know the Dataset

# In[ ]:


flights[:3]


# In[ ]:


np.shape(airports)


# In[ ]:


airlines[:3]


# In[ ]:


airports[:3]


# #### Cancelled flights in current `flights` variable

# In[ ]:


np.sum(flights["CANCELLED"])


# # Tests
# Using only American Airlines with about 600.000 flights instead of the full 5.5 Mio for performance reasons

# In[ ]:


carrier = "AA"
carrier_flights = flights[flights["AIRLINE"]==carrier]
carrier_flights[:5]


# Remove all flights without a `DEPARTURE_TIME`

# In[ ]:


carrier_flights = carrier_flights[pd.notnull(carrier_flights.DEPARTURE_TIME)]


# * Drop flights with `NaN` as delay
# * Merge weather into flights
# * Split dataset into x and y (ground thruth and features)
# * Transform String features into numbers

# In[ ]:


from sklearn.preprocessing import LabelEncoder
all_delay = carrier_flights.dropna(axis=0, how="any", subset=["DEPARTURE_DELAY"], inplace=False)
all_delay = all_delay.merge(right=weather[["stn", "mo", "da", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]], left_on=["ORIGIN_AIRPORT", "MONTH", "DAY"], right_on=["IATA_CODE", "mo", "da"])
y_delay = all_delay[["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DATE", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]]
x_delay = all_delay["DEPARTURE_DELAY"]
le = LabelEncoder()
y_delay["ORIGIN_AIRPORT"] = le.fit_transform(y_delay["ORIGIN_AIRPORT"])
y_delay["DESTINATION_AIRPORT"] = le.fit_transform(y_delay["DESTINATION_AIRPORT"])
y_delay["IATA_CODE"] = le.fit_transform(y_delay["IATA_CODE"])


# Now unused code that we used for testing effect of different estimator numbers. Results showed no increase in accuracy by increasing estimator count
# 
# ```python
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# configurations = [
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp", "mxpsd", "gust",],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust",],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "sndp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "snow_ice_pellets"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "hail"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
#                  ]
# estimatorVar = [10, 50, 100, 200]
# 
# for configuration in configurations:
#     y_delay_train, y_delay_test, x_delay_train, x_delay_test = train_test_split(y_delay[configuration], x_delay)
#     print("MODEL " + str(configuration))
#     for estimators in estimatorVar:
#         time = datetime.datetime.now()
#         print("ESTIMATORS: " + str(estimators))
#         model = RandomForestRegressor(n_estimators=estimators)
#         model.fit(y_delay_train, x_delay_train)
#         predicted = model.predict(y_delay_test)
#         print("SCORE: " + str(model.score(y_delay_test, x_delay_test)))
#         print("MEAN: " + str(np.mean(np.abs(predicted - x_delay_test))))
#         print("MEDIAN: " + str(np.median(np.abs(predicted - x_delay_test))))
#         timediff = datetime.datetime.now() - time
#         print("TOOK " + str(timediff.total_seconds()) + " SECONDS")
#         print("-------------------------------------")
# ```

# Showing different results for varying configurations
# 
# Note: some configurations are disabled to ensure the kernel doesn't take longer than 9h to run (the maximum allowed by kaggle)

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
configurations = [
    ["ORIGIN_AIRPORT", "SCHEDULED_DEPARTURE_TIME"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp", "mxpsd", "gust",],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust",],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp"],
    #["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "sndp"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog"],
    #["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "snow_ice_pellets"],
    #["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "hail"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
                 ]
estimatorVar = [10]

for configuration in configurations:
    y_delay_train, y_delay_test, x_delay_train, x_delay_test = train_test_split(y_delay[configuration], x_delay)
    print("MODEL " + str(configuration))
    for estimators in estimatorVar:
        time = datetime.datetime.now()
        print("ESTIMATORS: " + str(estimators))
        model = RandomForestRegressor(n_estimators=estimators)
        model.fit(y_delay_train, x_delay_train)
        predicted = model.predict(y_delay_test)
        print("SCORE: " + str(model.score(y_delay_test, x_delay_test)))
        print("MEAN: " + str(np.mean(np.abs(predicted - x_delay_test))))
        print("MEDIAN: " + str(np.median(np.abs(predicted - x_delay_test))))
        timediff = datetime.datetime.now() - time
        print("TOOK " + str(timediff.total_seconds()) + " SECONDS")
        print("-------------------------------------")


# # Tests with full dataset
# ### Classification
# Classification is intended to predict wether a flight is cancelled or not

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


flights_cancellation = flights.merge(right=weather[["stn", "mo", "da", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]], left_on=["ORIGIN_AIRPORT", "MONTH", "DAY"], right_on=["IATA_CODE", "mo", "da"])


# In[ ]:


len(flights_cancellation)


# In[ ]:


y_cancellation = flights_cancellation[["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK"]]
x_cancellation = flights_cancellation["CANCELLED"]
le = LabelEncoder()
y_cancellation["ORIGIN_AIRPORT"] = le.fit_transform(y_cancellation["ORIGIN_AIRPORT"])
y_cancellation["DESTINATION_AIRPORT"] = le.fit_transform(y_cancellation["DESTINATION_AIRPORT"])
y_cancellation_train, y_cancellation_test, x_cancellation_train, x_cancellation_test = train_test_split(y_cancellation, x_cancellation)
model = RandomForestClassifier()
model.fit(y_cancellation_train, x_cancellation_train)
print(model.score(y_cancellation_test, x_cancellation_test))
predicted = model.predict(y_cancellation_test)


# In[ ]:


np.sum([predicted == 0])


# In[ ]:


len(predicted)


# In[ ]:


np.sum([1 for i, j in zip(predicted, x_cancellation_test) if i == j])


# In[ ]:


model.score(y_cancellation_test, x_cancellation_test)


# In[ ]:


y_cancellation = flights_cancellation[["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"]]
x_cancellation = flights_cancellation["CANCELLED"]
le = LabelEncoder()
y_cancellation["ORIGIN_AIRPORT"] = le.fit_transform(y_cancellation["ORIGIN_AIRPORT"])
y_cancellation["DESTINATION_AIRPORT"] = le.fit_transform(y_cancellation["DESTINATION_AIRPORT"])
y_cancellation_train, y_cancellation_test, x_cancellation_train, x_cancellation_test = train_test_split(y_cancellation, x_cancellation)
model = RandomForestClassifier()
model.fit(y_cancellation_train, x_cancellation_train)
print(model.score(y_cancellation_test, x_cancellation_test))
predicted = model.predict(y_cancellation_test)


# ### Regression
# Prediction of the exact delay per flight. Again some configurations are disabled for performance reasons

# In[ ]:


from sklearn.preprocessing import LabelEncoder
all_delay = flights.dropna(axis=0, how="any", subset=["DEPARTURE_DELAY"], inplace=False)
all_delay = all_delay.merge(right=weather[["stn", "mo", "da", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]], left_on=["ORIGIN_AIRPORT", "MONTH", "DAY"], right_on=["IATA_CODE", "mo", "da"])
y_delay = all_delay[["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DATE", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]]
x_delay = all_delay["DEPARTURE_DELAY"]
le = LabelEncoder()
y_delay["ORIGIN_AIRPORT"] = le.fit_transform(y_delay["ORIGIN_AIRPORT"])
y_delay["DESTINATION_AIRPORT"] = le.fit_transform(y_delay["DESTINATION_AIRPORT"])
y_delay["IATA_CODE"] = le.fit_transform(y_delay["IATA_CODE"])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
configurations = [
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp", "mxpsd", "gust",],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust",],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp"],
    #["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "sndp"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog"],
    #["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "snow_ice_pellets"],
    #["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "hail"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
    ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
                 ]
estimatorVar = [10]

for configuration in configurations:
    y_delay_train, y_delay_test, x_delay_train, x_delay_test = train_test_split(y_delay[configuration], x_delay)
    print("MODEL " + str(configuration))
    for estimators in estimatorVar:
        time = datetime.datetime.now()
        print("ESTIMATORS: " + str(estimators))
        model = RandomForestRegressor(n_estimators=estimators)
        model.fit(y_delay_train, x_delay_train)
        predicted = model.predict(y_delay_test)
        print("SCORE: " + str(model.score(y_delay_test, x_delay_test)))
        print("MEAN: " + str(np.mean(np.abs(predicted - x_delay_test))))
        print("MEDIAN: " + str(np.median(np.abs(predicted - x_delay_test))))
        timediff = datetime.datetime.now() - time
        print("TOOK " + str(timediff.total_seconds()) + " SECONDS")
        print("-------------------------------------")


# # Misc. code from development
# Most of the following code was repurposed above

# ```python
# carrier = "AA"
# carrier_flights = flights[flights["AIRLINE"]==carrier]
# carrier_flights[:5]
# ```

# ```python
# carrier_flights = carrier_flights[pd.notnull(carrier_flights.DEPARTURE_TIME)]
# ```

# ```python
# from sklearn.preprocessing import LabelEncoder
# all_delay = carrier_flights.dropna(axis=0, how="any", subset=["DEPARTURE_DELAY"], inplace=False)
# all_delay = all_delay.merge(right=weather[["stn", "mo", "da", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]], left_on=["ORIGIN_AIRPORT", "MONTH", "DAY"], right_on=["IATA_CODE", "mo", "da"])
# y_delay = all_delay[["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DATE", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]]
# x_delay = all_delay["DEPARTURE_DELAY"]
# le = LabelEncoder()
# y_delay["ORIGIN_AIRPORT"] = le.fit_transform(y_delay["ORIGIN_AIRPORT"])
# y_delay["DESTINATION_AIRPORT"] = le.fit_transform(y_delay["DESTINATION_AIRPORT"])
# y_delay["IATA_CODE"] = le.fit_transform(y_delay["IATA_CODE"])
# ```

# ```python
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# configurations = [
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp", "mxpsd", "gust",],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust",],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "sndp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "snow_ice_pellets"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "hail"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
#                  ]
# estimatorVar = [10, 50, 100, 200]
# 
# for configuration in configurations:
#     y_delay_train, y_delay_test, x_delay_train, x_delay_test = train_test_split(y_delay[configuration], x_delay)
#     print("MODEL " + str(configuration))
#     for estimators in estimatorVar:
#         time = datetime.datetime.now()
#         print("ESTIMATORS: " + str(estimators))
#         model = RandomForestRegressor(n_estimators=estimators)
#         model.fit(y_delay_train, x_delay_train)
#         predicted = model.predict(y_delay_test)
#         print("SCORE: " + str(model.score(y_delay_test, x_delay_test)))
#         print("MEAN: " + str(np.mean(np.abs(predicted - x_delay_test))))
#         print("MEDIAN: " + str(np.median(np.abs(predicted - x_delay_test))))
#         timediff = datetime.datetime.now() - time
#         print("TOOK " + str(timediff.total_seconds()) + " SECONDS")
#         print("-------------------------------------")
# ```

# ```python
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# configurations = [
#     ["ORIGIN_AIRPORT", "SCHEDULED_DEPARTURE_TIME"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "wdsp", "mxpsd", "gust",],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust",],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "sndp"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "snow_ice_pellets"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "hail"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "wdsp", "mxpsd", "gust", "prcp", "fog", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
#     ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud"],
#                  ]
# estimatorVar = [10]
# 
# for configuration in configurations:
#     y_delay_train, y_delay_test, x_delay_train, x_delay_test = train_test_split(y_delay[configuration], x_delay)
#     print("MODEL " + str(configuration))
#     for estimators in estimatorVar:
#         time = datetime.datetime.now()
#         print("ESTIMATORS: " + str(estimators))
#         model = RandomForestRegressor(n_estimators=estimators)
#         model.fit(y_delay_train, x_delay_train)
#         predicted = model.predict(y_delay_test)
#         print("SCORE: " + str(model.score(y_delay_test, x_delay_test)))
#         print("MEAN: " + str(np.mean(np.abs(predicted - x_delay_test))))
#         print("MEDIAN: " + str(np.median(np.abs(predicted - x_delay_test))))
#         timediff = datetime.datetime.now() - time
#         print("TOOK " + str(timediff.total_seconds()) + " SECONDS")
#         print("-------------------------------------")
# ```

# ```python
# np.median(np.abs(model.predict(y_delay_test) - x_delay_test))
# ```

# ```python
# np.max(model.predict(y_delay_test))/60```

# # REAL

# ```python
# flights = flights[pd.notnull(flights.DEPARTURE_TIME)]```

# ```python
# from sklearn.preprocessing import LabelEncoder
# all_delay = flights.dropna(axis=0, how="any", subset=["DEPARTURE_DELAY"], inplace=False)
# all_delay = all_delay.merge(right=weather[["stn", "mo", "da", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]], left_on=["ORIGIN_AIRPORT", "MONTH", "DAY"], right_on=["IATA_CODE", "mo", "da"])
# y_delay = all_delay[["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DATE", "SCHEDULED_DEPARTURE_TIME", "temp", "dewp", "stp", "visib", "wdsp", "mxpsd", "gust", "max", "min", "prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets", "hail", "thunder", "tornado_funnel_cloud", "IATA_CODE"]]
# x_delay = all_delay["DEPARTURE_DELAY"]
# le = LabelEncoder()
# y_delay["ORIGIN_AIRPORT"] = le.fit_transform(y_delay["ORIGIN_AIRPORT"])
# y_delay["DESTINATION_AIRPORT"] = le.fit_transform(y_delay["DESTINATION_AIRPORT"])
# y_delay["IATA_CODE"] = le.fit_transform(y_delay["IATA_CODE"])
# ```

# ```python
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# configuration = ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE_TIME', 'wdsp']
# estimators = 10
# y_delay_train, y_delay_test, x_delay_train, x_delay_test = train_test_split(y_delay[configuration], x_delay)
# time = datetime.datetime.now()
# print("ESTIMATORS: " + str(estimators))
# model = RandomForestRegressor(n_estimators=estimators)
# model.fit(y_delay_train, x_delay_train)
# predicted = model.predict(y_delay_test)
# print("SCORE: " + str(model.score(y_delay_test, x_delay_test)))
# print("MEAN: " + str(np.mean(np.abs(predicted - x_delay_test))))
# print("MEDIAN: " + str(np.median(np.abs(predicted - x_delay_test))))
# timediff = datetime.datetime.now() - time
# print("TOOK " + str(timediff.total_seconds()) + " SECONDS")
# print("-------------------------------------")
# ```

# ```python
# np.std(np.abs(predicted-x_delay_test))```

# ```python
# np.mean(np.abs(predicted-x_delay_test))```

# ```python
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# configuration = ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE_TIME', 'temp', 'wdsp', 'mxpsd', 'gust', 'prcp', 'snow_ice_pellets']
# estimators = 10
# y_delay_train, y_delay_test, x_delay_train, x_delay_test = train_test_split(y_delay[configuration], x_delay)
# time = datetime.datetime.now()
# print("ESTIMATORS: " + str(estimators))
# model = RandomForestRegressor(n_estimators=estimators)
# model.fit(y_delay_train, x_delay_train)
# predicted = model.predict(y_delay_test)
# print("SCORE: " + str(model.score(y_delay_test, x_delay_test)))
# print("MEAN: " + str(np.mean(np.abs(predicted - x_delay_test))))
# print("MEDIAN: " + str(np.median(np.abs(predicted - x_delay_test))))
# timediff = datetime.datetime.now() - time
# print("TOOK " + str(timediff.total_seconds()) + " SECONDS")
# print("-------------------------------------")```

# ```python
# model```
