#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('html', '', '<style>\n.rendered_html tr, .rendered_html th, .rendered_html td    {\n    text-align: left\n    }\n</style>')


# In[ ]:


get_ipython().system('pip install swifter')


# # Project 5 - (Data Visualization)
# ## Ford GoBike
# ### Phase 2: Data Exploration
# 
# <a href="#Introduction:">Introduction</a><br/>
# <a href="#Importing-Data">Importing Data</a><br/>
# <a href="#Data-Exploration:">Data Exploration</a><br/>
# <a href="#Conclusion:">Conclusion</a><br/>
# <a href="#Recommendations:">Recommendations</a><br/>

# #### Introduction:
# 
# Ford GoBike has shared puplic bikes rips in main 3 cities in Florida. the data is between the period of June 2017 to October 2019. The exploration will be applied on a data frame which went through wrangling process and it is ready to be explored. the data is more than 4 Million records of trip observations. A bike Trip is happenning between starting point and end point, each starting and ending point is a station. Each trip has starting time and ending time. each station has longitude and latitude represents the starting location and ending location on a map. each group of stations are located in a city, some trips can be outside the city. The trip has bike ID which represents the bike was used in the trip. Also the data contains some information about the user as gender, age. below you can find list of features of every observation.
# 

# <table>
#     <tr>
#         <th>
#             Trip Starting Details
#         </th>
#         <th>
#             Trip Ending Details
#         </th>
#         <th>
#             Trip Member & Other Information
#         </th>
#     </tr>
#     <tr>
#         <td>start_time</td>
#         <td>end_time</td>
#         <td>duration_sec</td>
#     </tr>
#         <tr>
#         <td>start_station_city</td>
#         <td>end_station_city</td>
#         <td>bike_id</td>
#     </tr>
#         <tr>
#         <td>start_station_id</td>
#         <td>end_station_id</td>
#         <td>user_type</td>
#     </tr>
#         <tr>
#         <td>start_station_name</td>
#         <td>end_station_name</td>
#         <td>member_birth_year</td>
#     </tr>
#             <tr>
#         <td>start_station_latitude</td>
#         <td>end_station_latitude</td>
#         <td>member_age_group</td>
#     </tr>
#         <tr>
#         <td>start_station_longitude</td>
#         <td>end_station_longitude</td>
#         <td>member_gender</td>
#     </tr>
#     <tr>
#         <td>
#         </td>
#         <td>
#         </td>
#         <td>
#             bike_share_for_all_trip
#         </td>
#     </tr>
# </table>

# #### Importing Data

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


# Importing date from `all_tripdata_cleaned.csv`and setup some type converstions for features.

# In[ ]:


df = pd.read_csv('../input/ford-gobike-data-wrangling/all_tripdata_cleaned.csv',dtype={"start_station_id": "str", "end_station_id": "str", "bike_id" : "str"}, parse_dates = ["start_time", "end_time"], low_memory=False)


# In[ ]:


columns = ['duration_sec',
           'start_time','start_station_city','start_station_id','start_station_name','start_station_latitude','start_station_longitude',
           'end_time'  ,'end_station_city','end_station_id'  ,'end_station_name'  ,'end_station_latitude'  ,'end_station_longitude'  ,
           'bike_id','user_type','member_birth_year','member_age_group','member_gender','bike_share_for_all_trip','rental_access_method'
          ]
df = df[columns]


# Confirm features data types and breif descriptions of it.

# In[ ]:


display(df.info())
display(df.describe())


# #### Data Exploration:
# 
# In my exploration, I am trying to find answers to the following questions:
# 
# * <a href="#exploration1" style="color:black">Which city has more trips? is there any reason for that?</a>
# * <a href="#exploration2" style="color:black">Is there any relations between age groups and trip duration? what about genders and trip duration?</a>
# * <a href="#exploration3" style="color:black">Who uses the bikes more Casual Customers or Subscribed Members?</a>
# * <a href="#exploration4" style="color:black">Is number of trips increasing monthly? Which cities has more increase in number of trips?</a>
# * <a href="#exploration5" style="color:black">Which day of the week has more number of trips? which day hour has more number of trips?</a>
# * <a href="#exploration6" style="color:black">Do bikers usually use the bikes within the city? or outside the city?</a>
# * <a href="#exploration7" style="color:black">What are 10 top stations considered to be a starting point for trips? what about the ending point?</a>

# <span id="exploration1"></span>
# ***Exploration 1:*** Which city has more trips? is there any reason for that?

# To achieve this, I would like to introduce a new features called `within_city_trip`, this feature will hold the city name in case the start and end stations cities are the same. Otherwise, it will hold a value of `OTHERS` which means that starting city of the trip is not the same city where it ended. 

# In[ ]:


df["within_city_trip"] = np.nan
df.loc[(df["start_station_city"] == "SAN FRANSISCO") & (df["end_station_city"] == "SAN FRANSISCO"),"within_city_trip"] = "SAN FRANSISCO"
df.loc[(df["start_station_city"] == "OAKLAND"      ) & (df["end_station_city"] == "OAKLAND"      ),"within_city_trip"] = "OAKLAND"
df.loc[(df["start_station_city"] == "SAN JOSE"     ) & (df["end_station_city"] == "SAN JOSE"     ),"within_city_trip"] = "SAN JOSE"
df.loc[(df["start_station_city"] != df["end_station_city"]                                       ),"within_city_trip"] = "OTHERS"


# By excluding trips that occured outside the city by filtering out feature `within_city_trip` that equals to `OTHERS`

# In[ ]:


df_trips_within_city = df.query('within_city_trip != "OTHERS"')
plt.figure(figsize = [12, 7])
plt.subplot(1, 2, 1)
plt.suptitle('Number of Within-City Trips',fontsize=16);
graph = sns.countplot(data = df_trips_within_city, x = "within_city_trip",order = df_trips_within_city.within_city_trip.value_counts().index);
graph.set_xlabel('Cities')
graph.set_ylabel('Number of Bike Trips')

trips_per_city = df_trips_within_city.groupby('within_city_trip')[['within_city_trip']].count()
trips_per_city["within_city_trip_prop"] = trips_per_city["within_city_trip"] / trips_per_city["within_city_trip"].sum()
trips_per_city.sort_values(by="within_city_trip_prop",inplace=True,ascending=False)

plt.subplot(1, 2, 2)
plt.pie(trips_per_city.within_city_trip_prop.tolist(), explode = (0,0,0),labels = trips_per_city.index.tolist(), autopct='%1.1f%%', startangle=90, textprops={'color':'white', "fontsize":16})
plt.show()


# ***Observations***
# ```
# San fransisco has most number of trips compared to Oakland and San Jose. San Fransisco has more than 3 Millions (around 75%) of trips during 2017 to 2019. Oakland has almost third amount of trips as San Fransisco, it is almost 1 Million (20.9%) trips during the same period. San Jose has the least number of trips, it only has less than 5% of the trips.
# ```

# Let us go further and check number of unique stations in every city.

# In[ ]:


city_start_stations = df_trips_within_city.groupby(["within_city_trip","start_station_id"])["start_station_id"].count().reset_index(name="start_station_id_count")
city_end_stations = df_trips_within_city.groupby(["within_city_trip","end_station_id"])["end_station_id"].count().reset_index(name="end_station_id_count")
city_start_stations = city_start_stations.rename(columns={"start_station_id":"station_id"})[["within_city_trip","station_id"]]
city_end_stations = city_end_stations.rename(columns={"end_station_id":"station_id"})[["within_city_trip","station_id"]]
city_stations = pd.concat([city_start_stations, city_end_stations], axis=0, ignore_index=True, sort=True)
city_stations = city_stations.drop_duplicates()


# In[ ]:


plt.figure(figsize = [12, 7])
plt.subplot(1, 2, 1)
plt.suptitle('Number of Stations per City',fontsize=16);
graph = sns.countplot(data = city_stations, x = "within_city_trip",order = city_stations.within_city_trip.value_counts().index);
graph.set_xlabel('Cities')
graph.set_ylabel('Number of Stations')

stations_per_city= city_stations.groupby('within_city_trip')[["station_id"]].count()
stations_per_city["stations_prop"] = stations_per_city["station_id"] / stations_per_city["station_id"].sum()
stations_per_city.sort_values(by="stations_prop",inplace=True,ascending=False)

plt.subplot(1, 2, 2)
plt.pie(stations_per_city.stations_prop.tolist(), explode = (0,0,0),labels = stations_per_city.index.tolist(), autopct='%1.1f%%', startangle=90, textprops={'color':'white',"fontsize":16})
plt.show()


# Let us go deeper and check Number of bikes available in every city

# In[ ]:


plt.figure(figsize = [12, 7])
plt.subplot(1, 2, 1)
plt.suptitle('Number of Bikes per City',fontsize=16);
graph = sns.countplot(data = df_trips_within_city, x = "within_city_trip",order = df_trips_within_city.within_city_trip.value_counts().index);
graph.set_xlabel('Cities')
graph.set_ylabel('Number of Bikes')

city_bikes = df_trips_within_city.groupby(["within_city_trip","bike_id"])["bike_id"].count().reset_index(name="bike_id_count")
city_bikes = city_bikes[["within_city_trip","bike_id"]].groupby("within_city_trip")[["bike_id"]].count()
city_bikes["bike_prop"] = city_bikes["bike_id"] / city_bikes["bike_id"].sum()
city_bikes.sort_values(by="bike_prop",inplace=True,ascending=False)

plt.subplot(1, 2, 2)
plt.pie(city_bikes.bike_prop.tolist(), explode = (0,0,0),labels = city_bikes.index.tolist(), autopct='%1.1f%%', startangle=90, textprops={'color':'white',"fontsize":16})
plt.show()


# ***Observations***
# ```
# From previous three graphs, It shows that number of stations and number of bikes could be a reason behind number of trips varies in every city. You can see that San fransisco has the most frequent rate of trips and that might be because of it has the most number of stations (around 50%) and the most number of bikes (around 56%). In the opposite San Jose has the least number of trips and that also might be because of San Jose has the lowest number of stations and bikes (19% and 27%).
# ```

# <span id="exploration2"></span>
# ***Exploration 2:*** Is there any relations between genders and trip duration? what about age groups and trip duration?

# I am going to explore number of trips per gender.

# In[ ]:


trips_per_gender = df.groupby('member_gender')[['member_gender']].count().rename(columns={"member_gender":"member_gender_count"}).sort_values("member_gender_count",ascending=False)
trips_per_gender.plot.bar()
plt.show()


# For a better representation i am going to view trips rate per each gender. 

# In[ ]:


plt.figure(figsize = [12, 7])
plt.suptitle('Trips Rate per Gender',fontsize=16);
plt.pie(trips_per_gender.member_gender_count, explode = (0,0,0),labels = trips_per_gender.index.tolist(), autopct='%1.1f%%', startangle=90, textprops={'color':'white',"fontsize":16})
plt.legend(loc="upper right")
plt.show()


# ***Observations***
# ```
# Most of bike riders are males they represent around 67% of the riders. Females are only 22% of the ride bikers. and around 11% are unknown.
# ```

# Now I am going check if there are any relation between gender and trip duration, as i have the duration in seconds, i will make a new feature taht holds duration in minutes

# In[ ]:


df["duration_min"] = df["duration_sec"]/60


# In[ ]:


sns.violinplot(data = df, x = 'member_gender', y = 'duration_min',order=df.member_gender.value_counts().index)


# the graph above was not very helpful, I will be zooming for the trips below 200 minutes

# In[ ]:


sns.violinplot(data = df.query("duration_min < 200"), x = 'member_gender', y = 'duration_min',order=df.member_gender.value_counts().index)


# I am going to explore the data on three intervals of trip durations:
# * first interval : below or equal to 60 minutes
# * second interval: between 60 and 200 minutes
# * third interval: more than 200 minutes

# In[ ]:


plt.figure(figsize = [15, 7])
plt.suptitle('Trip Duration (Minutes) Vs Gender',fontsize=16);

plt.subplot(1, 3, 1)
graph1 = sns.violinplot(data = df.query("duration_min <= 60"), x = 'member_gender', y = 'duration_min',order=df.member_gender.value_counts().index)
graph1.set_xlabel('Gender')
graph1.set_ylabel('Trip Duration (Minutes)')
plt.subplot(1, 3, 2)
graph2 = sns.violinplot(data = df.query("duration_min > 60 and duration_min <= 200"), x = 'member_gender', y = 'duration_min',order=df.member_gender.value_counts().index)
graph2.set_xlabel('Gender')
graph2.set_ylabel('Trip Duration (Minutes)')
plt.subplot(1, 3, 3)
graph3 = sns.violinplot(data = df.query("duration_min > 200"), x = 'member_gender', y = 'duration_min',order=df.member_gender.value_counts().index)
graph3.set_xlabel('Gender')
graph3.set_ylabel('Trip Duration (Minutes)')
plt.tight_layout()
plt.show()


# ***Observations***
# ```
# Clearly, Gender is not affecting the trip duration. most of the trips are usually below the 10 minutes regardless of the gender type
# ```

# Let us see if there is any relation between age and trip duration

# In[ ]:


plt.figure(figsize=(20, 10), dpi=80)
plt.scatter(df.duration_min,(2019 - df.member_birth_year))


# INTRESTING !!, there might be some relation between age and duration. as the younger the biker the more chance to have longer trips

# In[ ]:


df.loc[(df["member_birth_year"] <= 2001) & (df["member_birth_year"] > 1992), "member_age_group"] = "18-26"
df.loc[(df["member_birth_year"] <= 1992) & (df["member_birth_year"] > 1979), "member_age_group"] = "27-39"
df.loc[(df["member_birth_year"] <= 1979) & (df["member_birth_year"] > 1962), "member_age_group"] = "40-57"
df.loc[(df["member_birth_year"] <= 1962), "member_age_group"] = "older than 57"


# In[ ]:


plt.figure(figsize = [15, 7])
plt.suptitle('Trip Duration (Minutes) Vs Biker Age',fontsize=16);

plt.subplot(1, 3, 1)
graph1 = sns.violinplot(data = df.query("duration_min <= 60"), x = 'member_age_group', y = 'duration_min',order=["18-26", "27-39","40-57","older than 57"])
graph1.set_xlabel('Biker Age')
graph1.set_ylabel('Trip Duration (Minutes)')
plt.subplot(1, 3, 2)
graph2 = sns.violinplot(data = df.query("duration_min > 60 and duration_min <= 200"), x = 'member_age_group', y = 'duration_min',order=["18-26", "27-39","40-57","older than 57"])
graph2.set_xlabel('Biker Age')
graph2.set_ylabel('Trip Duration (Minutes)')
plt.subplot(1, 3, 3)
graph3 = sns.violinplot(data = df.query("duration_min > 200"), x = 'member_age_group', y = 'duration_min',order=["18-26", "27-39","40-57","older than 57"])
graph3.set_xlabel('Biker Age')
graph3.set_ylabel('Trip Duration (Minutes)')
plt.tight_layout()
plt.show()


# ***Observations***
# ```
# All bikers age groups have duration of trips to be less than 10 minutes. All age groups are some how equivelent for long duration trips. 
# ```

# <span id="exploration3"></span>
# ***Exploration 3:*** Who uses the bikes more Casual Customers or Subscribed Members?

# In[ ]:


plt.figure(figsize = [12, 7])
plt.subplot(1, 2, 1)
plt.suptitle('Subscribers vs Customers',fontsize=16);
graph = sns.countplot(data = df, x = "user_type",order = df.user_type.value_counts().index);
graph.set_xlabel('User Types')
graph.set_ylabel('Number of Trips')

user_types_trips= df.groupby('user_type')[["user_type"]].count().rename(columns={"user_type":"user_type_count"})
user_types_trips["user_type_prop"] = user_types_trips["user_type_count"] / user_types_trips["user_type_count"].sum()
user_types_trips.sort_values(by="user_type_prop",inplace=True,ascending=False)

plt.subplot(1, 2, 2)
plt.pie(user_types_trips.user_type_prop.tolist(), explode = (0,0),labels = user_types_trips.index.tolist(), autopct='%1.1f%%', startangle=90, textprops={'color':'white',"fontsize":16})
plt.show()


# ***Observations***
# ```
# More than 83% of the trips done by subscribed riders. and only around 16% who are casual customers. 
# ```

# to better understand the above findings, let us see what genders are more subscribed to our services and who from them are usually a casual customer. 

# In[ ]:


user_types_per_gender_trips= df.groupby(['user_type', 'member_gender'])[["user_type"]].count().rename(columns={"user_type":"user_type_count"})
user_types_per_gender_trips["user_type_prop"] = user_types_per_gender_trips[["user_type_count"]] / user_types_per_gender_trips.groupby("user_type").sum()


# In[ ]:


user_types_per_gender_trips.unstack().user_type_prop.plot.bar(stacked=True,figsize = (15, 7))
plt.suptitle('User Type VS Gender',fontsize=16);
plt.legend(loc="upper right")
plt.show()


# ***Observations***
# ```
# We can observe that around 70% of the subscribers are male, and Females only represent around 22%. For casual customers, they usually dont reveal their gender with a percentage close to 45%. Men also represent more propotion than female customers. 
# ```

# <span id="exploration4"></span>
# ***Exploration 4:*** Is number of trips increasing monthly? Which cities has more increase in number of trips?

# To achieve this, i had to add new two features (`start_month` and `start_year`)  extraxted from `start_time`.

# In[ ]:


df_month_year = df.copy()
df_month_year["start_month"] = df.start_time.dt.month
df_month_year["start_year"] = df.start_time.dt.year
df_month_year_grouped = df_month_year.groupby(['start_year','start_month'])[['start_year']].count()
df_month_year_grouped.plot.line(figsize=(20,10))
plt.xticks(np.arange(len(df_month_year_grouped.index)), df_month_year_grouped.index.tolist(),rotation=90)
plt.legend().remove()
plt.suptitle('Number of Trips per Month - Between June 2017 and October 2019',fontsize=16);
plt.xlabel('(Year, Month)')
plt.ylabel('Number of Trips')
plt.show()


# ***Observations***
# ```
# Overall, Number of trips are getting increases every month. there is a drop in number of trips in November and December of 2018 abd again it happened May and June 2019. The maximum number of trips happened twice, in March and July 2019. 
# ```

# Let us compare the increase of trips over the months per city

# In[ ]:


df_month_year_per_city = df_month_year.query('within_city_trip != "OTHERS"').groupby(['within_city_trip','start_year','start_month'])[['bike_id']].count().rename(columns={"bike_id":"trips_count"})
df_month_year_san_fransisco = df_month_year_per_city.query('within_city_trip == "SAN FRANSISCO"').groupby(['start_year','start_month'])[["trips_count"]].sum()
df_month_year_san_jose = df_month_year_per_city.query('within_city_trip == "SAN JOSE"').groupby(['start_year','start_month'])[["trips_count"]].sum()
df_month_year_oakland = df_month_year_per_city.query('within_city_trip == "OAKLAND"').groupby(['start_year','start_month'])[["trips_count"]].sum()


# In[ ]:


ax = df_month_year_san_fransisco.plot.line(figsize=(20,10))
df_month_year_san_jose.plot.line(ax=ax,figsize=(20,10))
df_month_year_oakland.plot.line(ax=ax,figsize=(20,10))
plt.legend(["SAN FRANSISCO","SAN JOSE","OAKLAND"])
plt.xticks(np.arange(len(df_month_year_grouped.index)), df_month_year_grouped.index.tolist(),rotation=90)
plt.suptitle('Number of Trips per Month per City - Between June 2017 and October 2019',fontsize=16);
plt.xlabel('(Year, Month)')
plt.ylabel('Number of Trips')
plt.show()


# ***Observations***
# ```
# We can clearly see that San Fransisco has the most increase in number of trip, where San Jose almost no change during the months. there is a slight increase for oakland. Also, we can observe the drop down in number of trips between October and December 2018 , and April to June in 2019. San Fransisco possesses the higher percentage in this drop follows it by Oakland. It is good if we can find a reason for that.
# ```

# <span id="exploration5"></span>
# ***Exploration 5:*** Which day of the week has more number of trips? which day hour has more number of trips?

# To achieve this, I have to create a new feature which represents the day of the week

# In[ ]:


df_week_day = df.copy()
df_week_day["weekday"] = df.start_time.dt.weekday
df_weekday_grouped = df_week_day.groupby(['weekday'])[['weekday']].count()
df_weekday_grouped.plot.bar(figsize=(10,5))
plt.xticks(df_weekday_grouped.index,["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
plt.suptitle('Number of Trips',fontsize=16);
plt.legend().remove()


# ***Observations***
# ```
# Weekdays has more number of trips than weekends, as it drop around 40% on weekends. Usually this means subscribers are using the bike as a transportation to work.
# ```

# To support our finding above, let us view at what hours the trips happens.

# To achieve this I have to extract Hours from `start_time` feature.

# In[ ]:


df_trip_per_hour = df.copy()
df_trip_per_hour["start_hour"] = df_trip_per_hour.start_time.dt.hour
df_trip_per_hour_grouped = df_trip_per_hour.groupby(['start_hour'])[["duration_sec","duration_min"]].agg({"duration_sec":"count", "duration_min":"mean"}).reset_index()
pal = sns.color_palette("Greens_d", 5)
df_trip_per_hour_grouped.describe()


# In[ ]:


def duration_min_paletter_classifier(duration_min):
    if(duration_min >10 and duration_min <= 13):
        return pal[4]
    elif(duration_min >13 and duration_min <= 15):
        return pal[3]
    elif(duration_min >15 and duration_min <= 17):
        return pal[2]
    elif(duration_min >17 and duration_min <= 20):
        return pal[1]
    else:
        return pal[0]
df_trip_per_hour_grouped["duration_min_palette"] = df_trip_per_hour_grouped.duration_min.swifter.apply(duration_min_paletter_classifier)


# In[ ]:


import matplotlib.patches as mpatches
legend_patches = [None,None,None,None,None]
legend_text = ["10-14 min","14-18 min","18-22 min","22-26 min","over 26 min"]
for i in range(5):
    legend_patches[i] = mpatches.Patch(color=pal[i], label=legend_text[4-i])


# In[ ]:


plt.figure(figsize = [12, 7])
graph = sns.barplot(data=df_trip_per_hour_grouped,x="start_hour", y="duration_sec",palette=df_trip_per_hour_grouped.duration_min_palette)
plt.suptitle('Number of Trips',fontsize=16);
plt.xlabel('Day Hour')
plt.ylabel('Number of Trips')
plt.legend(handles=legend_patches,title="Duration (min)")
plt.show()


# ***Observations***
# ```
# Actually that supports our finding previously, the trips usually happens on workday peek hours. The usage are either    between 7 and 9 o'clock in the morning, or between 4 and 6 o'clock in the afternoon. Also, you can clearly notice that on those hours the trip duration are the lowest, it is between 10 to 18 minutes. In the Other times trips duration increases between the peek hours between 10 AM and 3 PM to reach a duration of 22 to 26 minutes. the trips reaches more than 26 minutes in time between 1 to 4 AM.
# ```

# <span id="exploration6"></span>
# ***Exploration 6:*** Do bikers usually use the bikes within the city? or outside the city?

# In[ ]:


df.query('within_city_trip == "OTHERS"').shape[0]


# In[ ]:


trips_outside_city = df.query('within_city_trip == "OTHERS"').groupby(["start_station_city", "end_station_city"])["duration_sec"].count().reset_index(name="trip_count")


# In[ ]:


trips_outside_city["from_to"] = "( " + trips_outside_city["start_station_city"] +","+ trips_outside_city["end_station_city"]+")"
fig, ax = plt.subplots()
barlist = ax.barh(y="from_to",width="trip_count", data=trips_outside_city.sort_values(by="trip_count"))
plt.suptitle('Number of Outside-City Trips',fontsize=16);
plt.ylabel('(From City, To City)')
plt.legend().remove()
barlist[-1].set_color('r')
barlist[-2].set_color('r')
plt.show()


# ***Observations***
# ```
# Number of outside trips its very low compared to within-city trips, during almost 3 years, only 272 trips occured. Most of these trips was between San Fransisco and Oakland.
# ```

# <span id="exploration7"></span>
# ***Exploration 7:*** What are 10 top stations considered to be a starting point for trips? what about the ending point?

# In[ ]:


df_start_stations_per_city = df.groupby(["within_city_trip", "start_station_name", "start_station_latitude", "start_station_longitude"])["start_station_name"].count().reset_index(name="trip_count").sort_values("trip_count")
df_top10_start_stations_per_city = df_start_stations_per_city.groupby(["within_city_trip"]).tail(10).set_index('start_station_name')
df_end_stations_per_city = df.groupby(["within_city_trip", "end_station_name","end_station_latitude", "end_station_longitude"])["end_station_name"].count().reset_index(name="trip_count").sort_values("trip_count")
df_top10_end_stations_per_city = df_end_stations_per_city.groupby(["within_city_trip"]).tail(10).set_index('end_station_name')


# In[ ]:


main_map_edges = (-122.5,-121.8,37.2,37.9)
sanfransisco_map_edges = (-122.4650,-122.3846,37.7470,37.8109)
sanjose_map_edges = (-121.94,-121.86,37.3, 37.36)
oakland_map_edges = (-122.32,-122.23,37.78,37.88)



def plot_city_trips_map(df1_long, df1_lat, df1_size, df1_color,
                        df2_long, df2_lat, df2_size, df2_color,
                        city_image, city_edges):
    map_image = plt.imread(city_image)
    fig, ax = plt.subplots(figsize = (15,10))
    ax.scatter(df1_long,df1_lat, zorder=1, alpha= 0.8, c=df1_color, s=df1_size)
    ax.scatter(df2_long,df2_lat, zorder=2, alpha= 0.8, c=df2_color, s=df2_size)
    ax.set_xlim(city_edges[0],city_edges[1])
    ax.set_ylim(city_edges[2],city_edges[3])
    ax.imshow(map_image, zorder=0, extent = city_edges, aspect= 'equal')
def plot_city_trips_map_extend(df1_long, df1_lat, df1_size, df1_color,
                        df2_long, df2_lat, df2_size, df2_color,
                        city_image, city_edges,title_map):
    map_image = plt.imread(city_image)
    ax = plt.subplot(1, 1, 1,title=title_map)
    ax.scatter(df1_long,df1_lat, zorder=1, alpha= 0.8, c=df1_color, s=df1_size)
    ax.scatter(df2_long,df2_lat, zorder=2, alpha= 0.8, c=df2_color, s=df2_size)
    ax.set_xlim(city_edges[0],city_edges[1])
    ax.set_ylim(city_edges[2],city_edges[3])
    ax.legend(["Start Station", "End Station"])
    ax.imshow(map_image, zorder=0, extent = city_edges, aspect= 'equal')


# In[ ]:


df_top10_sanfransisco_start_stations = df_top10_start_stations_per_city.query('within_city_trip == "SAN FRANSISCO"')
df_top10_sanfransisco_end_stations = df_top10_end_stations_per_city.query('within_city_trip == "SAN FRANSISCO"')

df_top10_sanfransisco_start_stations[["trip_count"]].plot.barh(figsize=(15,5))
plt.suptitle('Top 10 Stations in San Fransisco',fontsize=16);
plt.ylabel('Start Stations')
plt.legend().remove()
df_top10_sanfransisco_end_stations[["trip_count"]].plot.barh(figsize=(15,5))
plt.ylabel('End Stations')
plt.legend().remove()


# In[ ]:


sanfransisco_trips_start_stations = df_start_stations_per_city.query('within_city_trip == "SAN FRANSISCO"')
plot_city_trips_map(sanfransisco_trips_start_stations.start_station_longitude,
                    sanfransisco_trips_start_stations.start_station_latitude,
                   sanfransisco_trips_start_stations.trip_count/300,
                   'b',
                    df_top10_sanfransisco_start_stations.start_station_longitude,
                    df_top10_sanfransisco_start_stations.start_station_latitude,
                   df_top10_sanfransisco_start_stations.trip_count/300,
                   'r',
                    "../input/map-images/sanfransisco.png",sanfransisco_map_edges)
sanfransisco_trips_end_stations = df_end_stations_per_city.query('within_city_trip == "SAN FRANSISCO"')
plot_city_trips_map(sanfransisco_trips_end_stations.end_station_longitude,
                    sanfransisco_trips_end_stations.end_station_latitude,
                   sanfransisco_trips_end_stations.trip_count/300,
                   'b',
                    df_top10_sanfransisco_end_stations.end_station_longitude,
                    df_top10_sanfransisco_end_stations.end_station_latitude,
                   df_top10_sanfransisco_end_stations.trip_count/300,
                   'r',
                    "../input/map-images/sanfransisco.png",sanfransisco_map_edges)


# In[ ]:


fig, ax = plt.subplots(figsize = (15,10))
plot_city_trips_map_extend(df_top10_sanfransisco_start_stations.start_station_longitude,
                    df_top10_sanfransisco_start_stations.start_station_latitude,
                   400,
                   'b',
                    df_top10_sanfransisco_end_stations.end_station_longitude,
                    df_top10_sanfransisco_end_stations.end_station_latitude,
                   100,
                   'g',
                    "../input/map-images/sanfransisco.png",sanfransisco_map_edges,"Top 10 Stations in San Fransisco")


# ***Observations***
# ```
# San Fransisco top 10 stations as trip starting point are the same for trip ending point. Most of the top stations located on Market Street.
# ```

# In[ ]:


df_top10_oakland_start_stations = df_top10_start_stations_per_city.query('within_city_trip == "OAKLAND"')
df_top10_oakland_end_stations = df_top10_end_stations_per_city.query('within_city_trip == "OAKLAND"')
df_top10_oakland_start_stations[["trip_count"]].plot.barh(figsize=(15,5))
plt.suptitle('Top 10 Stations in Oakland',fontsize=16);
plt.ylabel('Start Stations')
plt.legend().remove()
df_top10_oakland_end_stations[["trip_count"]].plot.barh(figsize=(15,5))
plt.ylabel('End Stations')
plt.legend().remove()


# In[ ]:


oakland_trips_start_stations = df_start_stations_per_city.query('within_city_trip == "OAKLAND"')
plot_city_trips_map(oakland_trips_start_stations.start_station_longitude,
                    oakland_trips_start_stations.start_station_latitude,
                   oakland_trips_start_stations.trip_count/200,
                   'b',
                    df_top10_oakland_start_stations.start_station_longitude,
                    df_top10_oakland_start_stations.start_station_latitude,
                   df_top10_oakland_start_stations.trip_count/200,
                   'r',
                    "../input/map-images/oakland.png",oakland_map_edges)
oakland_trips_end_stations = df_end_stations_per_city.query('within_city_trip == "OAKLAND"')
plot_city_trips_map(oakland_trips_end_stations.end_station_longitude,
                    oakland_trips_end_stations.end_station_latitude,
                   oakland_trips_end_stations.trip_count/200,
                   'b',
                    df_top10_oakland_end_stations.end_station_longitude,
                    df_top10_oakland_end_stations.end_station_latitude,
                   df_top10_oakland_end_stations.trip_count/200,
                   'r',
                    "../input/map-images/oakland.png",oakland_map_edges)


# In[ ]:


fig, ax = plt.subplots(figsize = (15,10))
plot_city_trips_map_extend(df_top10_oakland_start_stations.start_station_longitude,
                    df_top10_oakland_start_stations.start_station_latitude,
                   400,
                   'b',
                    df_top10_oakland_end_stations.end_station_longitude,
                    df_top10_oakland_end_stations.end_station_latitude,
                   100,
                   'g',
                    "../input/map-images/oakland.png",oakland_map_edges,"Top 10 Stations in Oakland")


# ***Observations***
# ```
# Oakland top 10 staions as trip starting point are mostly the same as trip ending point. But they are distributed a cross the city.
# ```

# In[ ]:


df_top10_sanjose_start_stations = df_top10_start_stations_per_city.query('within_city_trip == "SAN JOSE"')
df_top10_sanjose_end_stations = df_top10_end_stations_per_city.query('within_city_trip == "SAN JOSE"')
df_top10_sanjose_start_stations[["trip_count"]].plot.barh(figsize=(15,5))
plt.suptitle('Top 10 Stations in San Jose',fontsize=16);
plt.ylabel('Start Stations')
plt.legend().remove()
df_top10_sanjose_end_stations[["trip_count"]].plot.barh(figsize=(15,5))
plt.ylabel('End Stations')
plt.legend().remove()


# In[ ]:


sanjose_trips_start_stations = df_start_stations_per_city.query('within_city_trip == "SAN JOSE"')
plot_city_trips_map(sanjose_trips_start_stations.start_station_longitude,
                    sanjose_trips_start_stations.start_station_latitude,
                   sanjose_trips_start_stations.trip_count/100,
                   'b',
                    df_top10_sanjose_start_stations.start_station_longitude,
                    df_top10_sanjose_start_stations.start_station_latitude,
                   df_top10_sanjose_start_stations.trip_count/100,
                   'r',
                    "../input/map-images/sanjose.png",sanjose_map_edges)
sanjose_trips_end_stations = df_end_stations_per_city.query('within_city_trip == "SAN JOSE"')
plot_city_trips_map(sanjose_trips_end_stations.end_station_longitude,
                    sanjose_trips_end_stations.end_station_latitude,
                   sanjose_trips_end_stations.trip_count/100,
                   'b',
                    df_top10_sanjose_end_stations.end_station_longitude,
                    df_top10_sanjose_end_stations.end_station_latitude,
                   df_top10_sanjose_end_stations.trip_count/100,
                   'r',
                    "../input/map-images/sanjose.png",sanjose_map_edges)


# In[ ]:


fig, ax = plt.subplots(figsize = (15,10))
plot_city_trips_map_extend(df_top10_sanjose_start_stations.start_station_longitude,
                    df_top10_sanjose_start_stations.start_station_latitude,
                   400,
                   'b',
                    df_top10_sanjose_end_stations.end_station_longitude,
                    df_top10_sanjose_end_stations.end_station_latitude,
                   100,
                   'g',
                    "../input/map-images/sanjose.png",sanjose_map_edges,"Top 10 Stations in San Jose")


# ***Observations***
# ```
# Top 10 Stations in San Jose for trip starting point are exactly the ones for the trip ending point. Also they are distributed a cross the city.
# ```

# #### Conclusion:
# 
# * Male riders represent 67% of the riders, where female riders covers only around 22% of overall trips.
# * Subscribed riders covers 83% of the trips.
# * San Fransisco has the most bike trips compared to San Jose and Oakland, it covers around 75% of the trips, and the might be because it has the most number of stations and bikes and that could be a reason.
# * San Fransisco has the highest number of increase in number of trips in compare to San Jose and Oakland.
# * Riders usually use the bikes as a transportaion to work, they use it on weekdays and on peek hours the most to avoid traffic.
# * The trips that occures outside the city, mostly happens between Oakland and San Fransisco as there is only a bridge between the two cities.

# #### Recommendations:
# * I recommend the service provider to provide special offers for Female riders.
# * Increase in number of bikes and stations in San Jose. as it shows no satisfing increase during the 3 years.
# * Provide huge discounts during late night hours between 1 AM and 4 AM as mostly the bikes are idle during these hours
# * Support the outside trips between Oakland and San Fransisco by starting a competitions.

# Thank You <br/>
# Analysis Prepared By ***Nadeem Tabbaa*** <br/>
# Linkedin: <a href="https://www.linkedin.com/in/msytnadeem/">@msytnadeem</a>

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Project5-Data Visualization _ Exploration.ipynb'])

