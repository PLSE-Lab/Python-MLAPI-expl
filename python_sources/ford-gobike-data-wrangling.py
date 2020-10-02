#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install swifter')


# In[ ]:


# import shutil
# shutil.rmtree("/kaggle/working")


# # Project 5 - (Data Visualization)
# ## Ford GoBike
# ### Phase 1: Data Wrangling
# 
# <a href="#Introduction:">Introduction</a><br/>
# <a href="#Data-Gathering:">Data Gathering</a><br/>
# <a href="#Data-Assessment:">Data Assessment</a><br/>
# <a href="#Data-Cleansing">Data Cleansing</a><br/>
# <a href="#Data-Storing">Data Stroing</a><br/>
# <a href="">Project 5 - (Data Visualization) - Ford GoBike - Phase 2: Data Exploratory</a>
# 
# #### Introduction:
# 
# Ford GoBike has shared puplic bikes trips in main 3 cities in Florida. the data is between the period of June 2017 to October 2019. the data are distributed in multible csv files based on the month and the year. The data has more than 4 millions observations and 15 to 17 features. _check readme.md file to know more about the data_
# 
# 

# In[ ]:


import requests, zipfile, io
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import swifter
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
pd.set_option('float_format', '{:f}'.format)


# #### Data Gathering:
# 
# The data are spreaded on miltible files on the cloud. For 2017, it contains all months, but starting 2018 and 2019, it has a file for every month. some of these files has different naming conventions. so i had to make a dictionary that it generate the file url. All files are zipped csv file except 2017 is csv.
# 

# In[ ]:


files = [
        {"year": "2018",
        "months": ['01','02','03','04','05','06','07','08', '09', '10','11','12'],
        "template_url" : 'https://s3.amazonaws.com/baywheels-data/%s%s-fordgobike-tripdata.csv.zip',
        "filetype":'zip'},
        {"year": "2019",
        "months": ['01','02','03','04'],
        "template_url" : 'https://s3.amazonaws.com/baywheels-data/%s%s-fordgobike-tripdata.csv.zip',
        "filetype":'zip'},
        {"year": "2019",
        "months": ['05','06','07','08', '09', '10'],
        "template_url" : 'https://s3.amazonaws.com/baywheels-data/%s%s-baywheels-tripdata.csv.zip',
        "filetype":'zip'},
    {"year":"2017",
        "months": [''], 
        "template_url" : 'https://s3.amazonaws.com/baywheels-data/%s%s-fordgobike-tripdata.csv',
         "filetype":'csv'},
]

# looping on the files based on the dictionary, read the stream in the memory, and extract it into data folder.

for file in files:
    for month in file["months"]:
        url = file["template_url"] % (file["year"], month)
        response = requests.get(url)
        if file["filetype"] == 'zip':
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall('source')
        else:
            with open(os.path.join('source', "%s%s-fordgobike-tripdata.csv" % (file["year"], month)), mode='wb') as file:
                file.write(response.content)
        


# I have renamed the file to a better naming

# In[ ]:


# data_files = []
# for (dirpath, dirnames, filenames) in os.walk('/kaggle/working/source'):
#     data_files.extend(filenames)
#     break
# for file in data_files:
#     file_structure = file.split('-')
#     old = os.path.join("/kaggle/working/source", file)
#     print(file_structure)
#     new = os.path.join("/kaggle/working/source", file_structure[0]+'_'+file_structure[2])
#     os.rename(old,new)


# I have looped on all the files under data folder and loaded them into an array of dataframes

# In[ ]:


path = r'/kaggle/working/source' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    if filename == '/kaggle/working/source/201907-fordgobike-tripdata.csv':
        df = pd.read_csv(filename, index_col=None, sep=";",low_memory=False)
    else:
        df = pd.read_csv(filename, index_col=None,low_memory=False)
    li.append(df)


# By have a quick glance on all the data frames columns, the data has the bellow columns only, some dataframes has some columns are missing. Hence, i had to loop on all the data frames and find the missing columns and concat the missing columns and fill it with empty values.

# In[ ]:


li_columns = ['duration_sec',
           'start_time','start_station_id','start_station_name', 'start_station_latitude','start_station_longitude',
           'end_time'  ,'end_station_id'  ,'end_station_name'  ,'end_station_latitude'   ,'end_station_longitude'  ,
           'bike_id','user_type','member_birth_year','member_gender','bike_share_for_all_trip','rental_access_method'
          ]


# In[ ]:


for i in li:
    deltacolumns = list(set(li_columns) - set(i.columns.tolist()))
    for column in deltacolumns:
        i[column] = np.nan


# After the dataframes match in term of the colomns, and all of them contains the timestamp of the trip, i did not find any any column to indicate each dataframe originally came from which source. Therfore i have merged all the data frames into one.

# In[ ]:


df = pd.concat(li, axis=0, ignore_index=True, sort=True)


# I have sorted the dataframe based on `start_time` and `end_time` then i have reset the index column and stored the dataframe which contains the gathered data into one big csv and called it `all_tripdata.csv`

# In[ ]:


df = df[li_columns].sort_values(by=['start_time', 'end_time'])


# In[ ]:


df.reset_index(inplace=True)


# In[ ]:


df[li_columns].to_csv('all_tripdata.csv')


# #### Data Assessment:
# 
# I have imported the csv file `all_tripdata.csv` which contains bike trips for all months, and i have performed some type changes as the following:
# * converted start and end station ids into an `Int64` to accept null values and to remove deimal digits, then I have converted them into a `string` again. 
# * converted start and end stations with string as `"nan"` into numpy nan value `np.nan`
# * converted trips start and end time into `datetime` format.
# * converted `bike_id` into string
# * converted `member_birth_year` into `Int64` then into `string`.

# In[ ]:


df = pd.read_csv('all_tripdata.csv',low_memory=False)[li_columns]


# In[ ]:


df['start_station_id'] = df.start_station_id.astype('Int64').astype(str)
df['end_station_id'] = df.end_station_id.astype('Int64').astype(str)
df.loc[df['start_station_id'] == 'nan',['start_station_id']] = np.nan
df.loc[df['end_station_id'] == 'nan',['end_station_id']]  = np.nan


# In[ ]:


df['start_time'] = pd.to_datetime(df['start_time'], format="%Y-%m-%d %H:%M:%S.%f")
df['end_time'] = pd.to_datetime(df['end_time'], format="%Y-%m-%d %H:%M:%S.%f")


# In[ ]:


df['bike_id'] = df.bike_id.astype(str)
df['member_birth_year'] = df.member_birth_year.astype('Int64').astype(str)


# Now i want to have a quick glance on the data.
# * I have printed the 10 first records
# * checked features datatypes
# * printed out features with null values
# * printed out description of countative features. 

# In[ ]:


display(df.head(10))
display(df.info())
display((df.isna().mean() * df.shape[0]).astype(int))
display(df.describe())


# From above results below are list of assessments I would perform on the data:
# 
# * check stations with 0 lat/long
# * check each station has unique location
# * check trip duration in second that matches the start and end of the trip
# * check null values for any member information
# * check members birth year values
# * check null values for bike_id bike_share_for_all_trip
# * check null value for rental_access_method

# ***Assessment 1:*** print our station ids which has 0 longitude or latitude for start and end stations.

# In[ ]:


display(df.query('start_station_latitude == 0 or start_station_longitude == 0').start_station_id.unique())
display(df.query('end_station_latitude == 0 or end_station_longitude == 0').end_station_id.unique())


# ***Assessment 2:*** excluding stations from `Assessment 1`, I will perform the following:
# * combine each station with existing latitude and longitude recorded in start or end station
# * record the maximum and the minimum latitude and longitude recorded on each station.
# * compare the result of the maximum latitude and longitude with minimum latitude and longitude

# In[ ]:


start_stations_locations = df.query("start_station_id not in ('420', '449')")[['start_station_id', 'start_station_latitude',"start_station_longitude"]]
end_stations_locations = df.query("end_station_id not in ('420', '449')")[['end_station_id', 'end_station_latitude',"end_station_longitude"]]
start_stations_locations.rename(columns={"start_station_id": "station_id", "start_station_latitude": "station_latitude", "start_station_longitude":"station_longitude"},inplace=True)
end_stations_locations.rename(columns={"end_station_id": "station_id", "end_station_latitude": "station_latitude", "end_station_longitude":"station_longitude"},inplace=True)
stations_locations = pd.concat([start_stations_locations,end_stations_locations])
stations_locations_max = stations_locations.groupby('station_id').max()
stations_locations_min = stations_locations.groupby('station_id').min()
stations_locations_max_min = stations_locations_max.join(stations_locations_min, lsuffix='_max', rsuffix='_min')
stations_locations_max_min_wo_nan = stations_locations_max_min.query('station_id != "nan"')
stations_locations_max_min_wo_nan = stations_locations_max_min_wo_nan.round(4)
stations_locations_max_min_wo_nan['matched_latitude'] = stations_locations_max_min_wo_nan.station_latitude_max == stations_locations_max_min_wo_nan.station_latitude_min
stations_locations_max_min_wo_nan['matched_longitude'] = stations_locations_max_min_wo_nan.station_longitude_max == stations_locations_max_min_wo_nan.station_longitude_min
stations_locations_max_min_wo_nan_unmatched = stations_locations_max_min_wo_nan.query('matched_latitude == False or matched_longitude == False')
stations_locations_max_min_wo_nan_unmatched.plot.scatter('station_longitude_max','station_latitude_max',figsize=(10,5), c="blue")
stations_locations_max_min_wo_nan_unmatched.plot.scatter('station_longitude_min','station_latitude_min',figsize=(10,5), c="red")


# One of the locations above is very far from the rest of locations, let me evalute it

# In[ ]:


stations_locations_max_min_wo_nan_unmatched.query('station_latitude_max > 44')


# station 408 requires some attention. let us exclude it and plot the locations again.

# In[ ]:


stations_locations_max_min_wo_nan_unmatched_filtered = stations_locations_max_min_wo_nan_unmatched.query('station_latitude_max < 44')
stations_locations_max_min_wo_nan_unmatched_filtered.plot.scatter('station_longitude_max','station_latitude_max',figsize=(10,5),c="blue")
stations_locations_max_min_wo_nan_unmatched_filtered.plot.scatter('station_longitude_min','station_latitude_min',figsize=(10,5),c="red")


# Good result, Mostly each station has very close locations registered. 

# ***Assessment 3*** validate `duration_sec` equals to the delta between trip `start_time` and `end_time` 

# In[ ]:


actual_duration_second = df[['duration_sec', 'start_time', 'end_time']].copy()
actual_duration_second['actual_duration_sec'] = (actual_duration_second['end_time'] - actual_duration_second['start_time']).dt.seconds
actual_duration_second.query('duration_sec != actual_duration_sec').shape[0]


# ***Assessment 4:*** print out unique values of `member_gender`, `bike_share_for_all_trip` and `rental_access_method`

# In[ ]:


display(
    df.member_gender.unique(),
    df.bike_share_for_all_trip.unique(),
    df.rental_access_method.unique())


# ***Assessment 5:*** evaluate `member_birth_year` distribution. 

# In[ ]:


df[["member_birth_year"]].astype(float).boxplot(column="member_birth_year")


# exploring the `member_birth_year` yonger then 1960

# In[ ]:


df[["member_birth_year"]].astype(float).query('member_birth_year > 1960').boxplot(column="member_birth_year")


# exploring `member_birth_year` between 1960 adn 1940

# In[ ]:


display(df[["member_birth_year"]].astype(float).query('member_birth_year <= 1960 and member_birth_year > 1940').boxplot(column="member_birth_year"))


# exploring `member_birth_year` older than 1940

# In[ ]:


display(df[["member_birth_year"]].astype(float).query('member_birth_year <= 1940').boxplot(column="member_birth_year"))


# #### Data Cleansing
# 
# From the assessments above i have found the following quality issues and data tidenss required to be applied on the data
# 
# **Quality**
# 
# * Fix lat/long for station id 420 => testing station, dropped
# * Fix lat/long for station id 449 => unrecoverable, dropped
# * Fix lat/long for station id 408 => assigned correct lat/long for this station
# * Trips with no station id should be filled with nearest station per lat/long => solved as possible, dropped the rest
# * Assign median (rounded upto 4 digits) lat/long for every station id
# * Assign station names which has null values and fix stations that has multible stations names => require visual assessment and fixed the naming in csv file and imported. 
# * Fix 45942 rows with wrong trip duration => cleaned trup duration by calculating the differnce between end and start time
# * Fix member genders column => fixed the genders to be Female, Male and Others where Others are any null values or O, or Other
# * Fix weird members birth years => reassign members birth years which are less 1945 to nulls, and kept the rest.
# * Fix null values for bike_share_for_all_trip => could not fix null values, kept it to try extract some insights
# * Fix null values for rental_access_method => could not fix null values, kept it to try extract some insights
# 
# **Tideness**
# * Classify start_stations_id and end_station_id into cities (SAN FRANSISCO, SAN JOSE, OAKLAND)
# * Classify member birth year into age groups (18-26, 27-39,40,57,older than 57)
# 

# In[ ]:


df_clean = df.copy()


# ***Cleaning 1***: clean station 420

# In[ ]:


display(df_clean.query('start_station_id == "420"'),df_clean.query('end_station_id == "420"'))


# station 420 is a test station, I am going to exclude it from the data

# In[ ]:


df_clean = df_clean.query('start_station_id != "420" and end_station_id != "420"')


# ***Cleaning 2***: clean station 449

# In[ ]:


display(df_clean.query('start_station_id == "449"'),df_clean.query('end_station_id == "449"'))


# station 490 is an actual station, but I couldnot recover station location, I am going to exclude it

# In[ ]:


df_clean = df_clean.query('start_station_id != "449" and end_station_id != "449"')


# ***Cleaning 3***: clean station 408

# In[ ]:


display(df_clean.query('start_station_id == "408"'),df_clean.query('end_station_id == "408"'))


# station 408 has multiple locations, location of -122.388320, 37.18513 is closer to the truth, I am going to assign this location for station 408

# In[ ]:


df_clean.loc[df_clean["end_station_id"] == "408", 
             ["start_station_latitude","start_station_longitude",
              "end_station_latitude","end_station_longitude"]] = [37.718513,-122.388320,37.718513, -122.388320 ]


# ***Cleaning 4*** : Fill trips start and end stations with nearest station based on locations

# To achieve this, first i will calculate the median locations for each station and store it in `stations_locations_median`

# In[ ]:


start_stations_locations = df_clean[['start_station_id', 'start_station_latitude',"start_station_longitude"]].copy()
end_stations_locations = df_clean[['end_station_id', 'end_station_latitude',"end_station_longitude"]].copy()
start_stations_locations.rename(columns={"start_station_id": "station_id", "start_station_latitude": "station_latitude", "start_station_longitude":"station_longitude"},inplace=True)
end_stations_locations.rename(columns={"end_station_id": "station_id", "end_station_latitude": "station_latitude", "end_station_longitude":"station_longitude"},inplace=True)
stations_locations = pd.concat([start_stations_locations,end_stations_locations])
stations_locations_median = stations_locations.groupby('station_id').median()


# I will have two subsets of trips with missing starting stations (`start_nan_staitons`) and ending stations (`end_nan_stations`)

# In[ ]:


start_nan_stations = df_clean.query('start_station_id != start_station_id').reset_index()[['index', 'start_station_latitude', 'start_station_longitude']]
end_nan_stations = df_clean.query('end_station_id != end_station_id').reset_index()[['index', 'end_station_latitude', 'end_station_longitude']]


# I will apply on each subset the corrsponding function `start_distance_calculation` and `end_distance_calculation` which will assign return the nearest station id and its distance of every trip

# In[ ]:


def start_distance_calculation(row):
    delta_lat = (stations_locations_median['station_latitude'] - row['start_station_latitude']) ** 2
    delta_long = (stations_locations_median['station_longitude'] - row['start_station_longitude']) ** 2
    distance = np.sqrt(delta_lat + delta_long)
    new_cols = pd.Series(data=[distance.min(), distance.idxmin()], index=['start_min_distance_value', 'start_min_distance_station']) 
    result = pd.concat([row, new_cols])
    return result


start_nan_stations_nearest = start_nan_stations.swifter.apply(start_distance_calculation, axis=1)
start_nan_stations_nearest["index"] = start_nan_stations_nearest["index"].astype(int)


# In[ ]:


def end_distance_calculation(row):
    delta_lat = (stations_locations_median['station_latitude'] - row['end_station_latitude']) ** 2
    delta_long = (stations_locations_median['station_longitude'] - row['end_station_longitude']) ** 2
    distance = np.sqrt(delta_lat + delta_long)
    new_cols = pd.Series(data=[distance.min(), distance.idxmin()], index=['end_min_distance_value', 'end_min_distance_station']) 
    result = pd.concat([row, new_cols])
    return result


end_nan_stations_nearest = end_nan_stations.swifter.apply(end_distance_calculation, axis=1)
end_nan_stations_nearest["index"] = end_nan_stations_nearest["index"].astype(int)


# I will only accept a station to be considered the nearest if the distance is less than 0.01 KM, and I will store the results in `start_nan_stations_accepted` and `end_nan_stations_accepted`

# In[ ]:


start_nan_stations_accepted = start_nan_stations_nearest.query('start_min_distance_value < 0.01').set_index("index")[["start_min_distance_value", "start_min_distance_station"]]
end_nan_stations_accepted = end_nan_stations_nearest.query('end_min_distance_value < 0.01').set_index("index")[["end_min_distance_value", "end_min_distance_station"]]


# Now, I will assign every missing stations Id based on the accepted nearest stations calculated from above.

# In[ ]:


df_clean.index.name = "index"


# In[ ]:


df_clean = df_clean.join(start_nan_stations_accepted)
df_clean.loc[df_clean["start_min_distance_value"] == df_clean["start_min_distance_value"], ["start_station_id"]]= df_clean["start_min_distance_station"]
df_clean.drop(columns=['start_min_distance_value', 'start_min_distance_station'],inplace=True)


# In[ ]:


df_clean = df_clean.join(end_nan_stations_accepted)
df_clean.loc[df_clean["end_min_distance_value"] == df_clean["end_min_distance_value"], ["end_station_id"]]= df_clean["end_min_distance_station"]
df_clean.drop(columns=['end_min_distance_value', 'end_min_distance_station'],inplace=True)


# I will drop any observation which remains their `start_station_id` or `end_station_id` as nulls

# In[ ]:


df_clean = df_clean.query('start_station_id == start_station_id')
df_clean = df_clean.query('end_station_id == end_station_id')


# ***Cleaning 5***: assign stations locations with their medians locations

# using `stations_locations_median`, I will assign each station with corresponding median location (latitude, longitude)

# In[ ]:


def fix_station_lat_long(row):
    df_clean.loc[df_clean["start_station_id"] == row["station_id"], ["start_station_latitude","start_station_longitude"]]  =  [row.station_latitude,row.station_longitude]
    df_clean.loc[df_clean["end_station_id"]   == row["station_id"], ["end_station_latitude","end_station_longitude"]]    =  [row.station_latitude, row.station_longitude]

stations_locations_median_rounded_4 = stations_locations_median.reset_index().round(4)    
_ignore = stations_locations_median_rounded_4.swifter.apply(fix_station_lat_long, axis=1)


# validate if any stations has different locations by calculating the minumum and maximum locations (latitude, longitude)

# In[ ]:


start_stations_locations_clean = df_clean[['start_station_id', 'start_station_latitude',"start_station_longitude"]].copy()
end_stations_locations_clean = df_clean[['end_station_id', 'end_station_latitude',"end_station_longitude"]].copy()
start_stations_locations_clean.rename(columns={"start_station_id": "station_id", "start_station_latitude": "station_latitude", "start_station_longitude":"station_longitude"},inplace=True)
end_stations_locations_clean.rename(columns={"end_station_id": "station_id", "end_station_latitude": "station_latitude", "end_station_longitude":"station_longitude"},inplace=True)
stations_locations_clean = pd.concat([start_stations_locations_clean,end_stations_locations_clean])
stations_locations_max_clean = stations_locations_clean.groupby('station_id').max()
stations_locations_min_clean = stations_locations_clean.groupby('station_id').min()
stations_locations_max_min_clean = stations_locations_max_clean.join(stations_locations_min_clean, lsuffix='_max', rsuffix='_min')
stations_locations_max_min_wo_nan_clean = stations_locations_max_min_clean.query('station_id != "nan"').round(4).copy()
stations_locations_max_min_wo_nan_clean['matched_latitude'] = stations_locations_max_min_wo_nan_clean.station_latitude_max == stations_locations_max_min_wo_nan_clean.station_latitude_min
stations_locations_max_min_wo_nan_clean['matched_longitude'] = stations_locations_max_min_wo_nan_clean.station_longitude_max == stations_locations_max_min_wo_nan_clean.station_longitude_min
stations_locations_max_min_wo_nan_unmatched_clean = stations_locations_max_min_wo_nan_clean.query('matched_latitude == False or matched_longitude == False')
stations_locations_max_min_wo_nan_unmatched_clean


# > ***Cleaning 6:*** Class[](http://)ify stations by cities

# I have imported a map image using the following technique (<a href="https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db">click here</a>) to represent in which cities each station are in

# In[ ]:


main_map_edges = (-122.5,-121.8,37.2,37.9)
main_map_image = plt.imread('../input/map-images/map.png')
fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(stations_locations_max_min_clean.station_longitude_max,stations_locations_max_min_clean.station_latitude_max,zorder=1, alpha= 0.8, c='b', s=20)
ax.set_xlim(main_map_edges[0],main_map_edges[1])
ax.set_ylim(main_map_edges[2],main_map_edges[3])
ax.imshow(main_map_image, zorder=0, extent = main_map_edges, aspect= 'equal')


# all stations are located in 3 different cities:
# * SAN FRANSISCO
# * SAN JOSE
# * OAKLAND
# 
# I am going to classify every trip with `start_station_city` and `end_station_city` by using start and end station longitude

# In[ ]:


df_clean["start_station_city"] = np.nan
df_clean["end_station_city"] = np.nan
df_clean.loc[df_clean['start_station_longitude'] < -122.35, "start_station_city"] = "SAN FRANSISCO"
df_clean.loc[(df_clean['start_station_longitude'] > -122.35) & (df_clean['start_station_longitude'] < -122.1 ), "start_station_city"] = "OAKLAND"
df_clean.loc[df_clean['start_station_longitude'] > -122.1 , "start_station_city"] = "SAN JOSE"
df_clean.loc[df_clean['end_station_longitude'] < -122.35, "end_station_city"] = "SAN FRANSISCO"
df_clean.loc[(df_clean['end_station_longitude'] > -122.35) & (df_clean['end_station_longitude'] < -122.1 ), "end_station_city"] = "OAKLAND"
df_clean.loc[df_clean['end_station_longitude'] > -122.1 , "end_station_city"] = "SAN JOSE"


# In[ ]:


df_clean.query('start_station_city != end_station_city').start_station_city.count()


# from above, i have noticed that some trips are actually started and ended in two different cities. I am going to explore this area in exploration section.

# ***Cleaning 7***: cleaning stations which has multiple names

# I have stored station_ids for any station which have multiple station names into `stations_many_names`

# In[ ]:


start_stations_names = df_clean[['start_station_id', 'start_station_name']].copy()
end_stations_names = df_clean[['end_station_id', 'end_station_name']].copy()
start_stations_names.rename(columns={"start_station_id": "station_id", "start_station_name": "station_name"},inplace=True)
end_stations_names.rename(columns={"end_station_id": "station_id", "end_station_name": "station_name"},inplace=True)
stations_names = pd.concat([start_stations_names,end_stations_names])
stations_names_wo_duplicates = stations_names.drop_duplicates()
stations_many_names = stations_names_wo_duplicates.groupby('station_id').count().query('station_name > 1')


# In[ ]:


stations_many_names.shape[0]


# i have only 31 stations, I am going to perform a visual assessment on them, I have exported them into a csv file `stations_name_error.csv`

# In[ ]:


stations_names_requiers_offline_fix = stations_names_wo_duplicates.query(f'station_id in {stations_many_names.index.tolist()}').sort_values('station_id').set_index('station_id').query('station_name == station_name')
stations_names_requiers_offline_fix.to_csv('stations_names_error.csv')


# I have performed a visual assessment by fixing some spelling mistakes, or descriping the same place with some abbriviations. but some of the stations mentioning complete different names, for those i had to look it up on the map and select the correct name. I have done the fixing of the 31 stations in `stations_names_fixed.csv` then i have imported it to fix the stations names.

# In[ ]:


station_names_fixed = pd.read_csv("../input/visual-assessment/stations_names_fixed.csv")


# In[ ]:


station_names_fixed["station_id"] = station_names_fixed.station_id.astype(str)


# I have used the function `fix_stations_names` to update all trips with correct `start_station_name` and `end_station_name` based on the fixed stations.

# In[ ]:


stations_names_wo_duplicates_fixed = stations_names_wo_duplicates.query('station_name == station_name').set_index('station_id').join(station_names_fixed.set_index('station_id'),rsuffix=('_fixed')).reset_index()
stations_names_wo_duplicates_fixed.loc[stations_names_wo_duplicates_fixed['station_name_fixed'] == stations_names_wo_duplicates_fixed['station_name_fixed'], "station_name"] = stations_names_wo_duplicates_fixed['station_name_fixed'] 
stations_names_wo_duplicates_fixed = stations_names_wo_duplicates_fixed.drop_duplicates().set_index("station_id")[["station_name"]]


# In[ ]:


def fix_station_names(row):
    df_clean.loc[df_clean["start_station_id"] == row["station_id"], ["start_station_name"]]  =  [row.station_name]
    df_clean.loc[df_clean["end_station_id"]   == row["station_id"], ["end_station_name"]]    =  [row.station_name]

_ignore = stations_names_wo_duplicates_fixed.reset_index().swifter.apply(fix_station_names, axis=1)


# ***Cleaning 8***: clean `duration_sec` by finding delta between `end_time` and `start_time`

# In[ ]:


df_clean.loc[:,['duration_sec']] = (df_clean['end_time'] - df_clean['start_time']).dt.seconds


# validate the fix `duration_sec` is fixed

# In[ ]:


actual_duration_second = df_clean[['duration_sec', 'start_time', 'end_time']].copy()
actual_duration_second['actual_duration_sec'] = (actual_duration_second['end_time'] - actual_duration_second['start_time']).dt.seconds
actual_duration_second.query('duration_sec != actual_duration_sec').shape[0]


# In[ ]:


df_clean.duration_sec.describe()


# ***Cleaning 9:*** clean `member_gender` with proper values

# In[ ]:


df_clean.member_gender.replace('M','Male', inplace=True)
df_clean.member_gender.replace('F','Female', inplace=True)
df_clean.member_gender.replace('O','Other', inplace=True)
df_clean.member_gender.replace('?','Other', inplace=True)
df_clean.member_gender.fillna("Other", inplace=True)


# In[ ]:


df_clean.member_gender.unique()


# ***Cleaning 10***: cleaning `member_birth_year`

# evaluate the distribution of `member_birth_year` column, it has a lot of wide range of outlier values which is `member_birth_year` is less than 1960

# In[ ]:


df_clean.member_birth_year = df_clean.member_birth_year.astype(float)
df_clean.boxplot(column="member_birth_year")


# In[ ]:


df_clean.query('member_birth_year < 1960').member_birth_year.hist(bins=30)


# from above, you can clearly notice that still we can consider `member_birth_year` who are between 1960 and 1945, let us view the distribution of it

# In[ ]:


df_clean.query('member_birth_year < 1960 and member_birth_year > 1945').boxplot(column="member_birth_year")


# Let us view the distribution for `member_birth_year` who are below 1945

# In[ ]:


df_clean.query('member_birth_year <= 1945').boxplot(column="member_birth_year")


# In[ ]:


df_clean.query('member_birth_year < 1945').member_birth_year.hist(bins=20)


# In[ ]:


df_clean.query('member_birth_year < 1945').shape[0]


# From above, as the records which have `member_birth_year` less than 1945 only 7856 records, and their not well distributed, and by looking at at least 1200 records has default `member_birth_year` equals to 1900. then i have decided to clean this data by assigning `np.nan` value to them

# In[ ]:


df_clean.loc[df_clean["member_birth_year"] < 1945, "member_birth_year" ] = np.nan
df_clean.query('member_birth_year < 1945').shape[0]


# Now the range of `member_birth_year` is between 2001 and 1945, with an outliers values between 1960 and 1945, which is quite resonable.

# In[ ]:


df_clean.boxplot(column="member_birth_year")


# In[ ]:


df_clean["member_age_group"] = np.nan


# In[ ]:


df_clean.loc[(df_clean["member_birth_year"] <= 2001) & (df_clean["member_birth_year"] > 1992), "member_age_group"] = "18-26"
df_clean.loc[(df_clean["member_birth_year"] <= 1992) & (df_clean["member_birth_year"] > 1979), "member_age_group"] = "27-39"
df_clean.loc[(df_clean["member_birth_year"] <= 1979) & (df_clean["member_birth_year"] > 1962), "member_age_group"] = "40-57"
df_clean.loc[(df_clean["member_birth_year"] <= 1862), "member_age_group"] = "older than 57"


# In[ ]:


df_clean.head(10)


# #### Data Storing
# 
# Check null values again, and then store them into `all_tripdata_cleaned.csv`

# In[ ]:


(df_clean.isna().mean() * df_clean.shape[0]).astype(int)


# In[ ]:


df_clean.to_csv('all_tripdata_cleaned.csv', index=False)


# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Project5-Data Visualization _ Wrangling.ipynb'])

