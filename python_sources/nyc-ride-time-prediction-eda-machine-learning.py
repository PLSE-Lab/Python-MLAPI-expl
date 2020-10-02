#!/usr/bin/env python
# coding: utf-8

# # New York City taxi trip duration

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import folium
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read the train data set file 

# In[2]:


df = pd.read_csv("../input/nyc-taxi-trip-duration/train.csv", index_col = "id")
# read the input data from file with the "id" as index


# In[3]:


df.head(n = 5)
# display top 5 rows of the data


# ## Initial analysis 
# 
# From the inital view of the data, lets get the initial understanding of the data,
# 
# 1. **id**: A unique ID given to each ride made by the passenger.
# 2. **vendor_id**: The ID of the vendor who owns the taxi.
# 3. **pickup_datetime**: The start date and the time of the ride.
# 4. **dropoff_datetime**: The end date and time of the ride.
# 5. **passenger_count**: The number of passengers traveled in the ride.
# 6. **pickup_longitude and pickup_latitude**: This specifies the location of the passenger pickup. Latitude and longitude both help us measure the location of pickup. (These both has to be taken together in order to make sense).
# 7. **dropoff_longitude and dropoff_latitude**: This specifies the location of the passenger dropoff. Similar measures are used as the above one.
# 8. **store_and_fwd_flag**:
# 9. **trip_duration**: The total duration of the trip measures in seconds.
# 
# 
# Apart from these features, we can extract the following features which will help for further analysis.
# 
# 1. **Distance between pickup and dropoff**: We exactly don't know the distance that was traveled during a given ride, but we can get a rough estimate of that using the GoogleMaps API.
# 2. **Decoding the date**: My decoding the pickup date, we can get features such as the month, hour, weeekday or weekend, holiday or not, event happening or not and more.
# 3. **Better understanding of the routes**: Here we try to see the number of turns (left or right), exits taken during the ride. This again is going to be a rough estimate done through GoogleMaps API.
# 4. **Weather condition**: The weather can be a pretty good factor influencing the traffic, we get this data from Mr. Mathijs Waegemakers(https://www.kaggle.com/mathijs/datasets).
# 5. **Accidents**: Accidents can have a high impact on the traffic as well, this data is obtained from Mr. Oscarleo (https://www.kaggle.com/oscarleo).

# ## Decoding the date 

# In[4]:


df["pickup_datetime"] = pd.DatetimeIndex(df["pickup_datetime"], dtype = pd.DatetimeIndex)
# convert to datetime object


# In[5]:


holidays_df = pd.read_csv("../input/nycholidays/NYC_holidays.csv")
# read the holidays data
holidays_df["Date"] = pd.DatetimeIndex(holidays_df["Date"])
# convert to datetime object
HOLIDAYS = holidays_df["Date"].apply(lambda x : x.date()).values
# get the holiday/event dates


# In[6]:


def get_datetime_details(pickup_datetime, df):
    """
    Get more details related date and time
    which will be useful for our analysis.
    """
    
    df = df.assign(hour = pickup_datetime.dt.hour)
    # add hours column to the dataframe
    df = df.assign(minute = pickup_datetime.dt.minute)
    # add minute column to the dataframe
    df = df.assign(second = pickup_datetime.dt.second)
    # add second column to the dataframe
    df = df.assign(date = pickup_datetime.dt.date)
    # add date column to the dataframe
    df = df.assign(day = pickup_datetime.dt.dayofweek)
    # add the day column to the dataframe
    df = df.assign(weekend_or_not = df["day"].apply(lambda x: x >= 5))
    df["weekend_or_not"] = df["weekend_or_not"].astype(int)
    # check for weekend
    df = df.assign(holiday_or_not = df["date"].apply(lambda x: x in HOLIDAYS))
    df["holiday_or_not"] = df["holiday_or_not"].astype(int)
    # check if its a holiday
    return df


# In[7]:


df = get_datetime_details(df["pickup_datetime"], df)
# add details


# In[8]:


df.head()


# Now we have all the data that we could extract from the pcikup date and time. We will see how these data influences the ride time. 
# 
# Next, we will use https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm dataset to add more useful features for our analysis. This will help us improving the quality of the model and analysis.

# ## Exploring the NYC route data 

# In[4]:


route_df1 = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv",
                      index_col = "id")
route_df2 = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv",
                      index_col = "id")
route_df = pd.concat([route_df1, route_df2])
# read the routes data which will aid our analysis


# In[5]:


route_df.head()


# In[10]:


maneuver_count_df = pd.read_csv("../input/nycholidays/various_maneuver_counts.csv", index_col="id")
# get the various maneuver counts


# In[11]:


def add_route_details(df, route_df):
    """
    This function helps to add various route details
    to the main dataframe. This will assist us in 
    further analysis.
    """
    
    df = df.assign(distance = route_df["total_distance"])
    # add the ride distance as a feature
    df = df.assign(best_travel_time = route_df["total_travel_time"])
    # get the best travel time
    return df


# In[12]:


df = add_route_details(df, route_df)
# add the distance and best time possible


# In[13]:


df = df.join(maneuver_count_df)
# get the maneuver counts as well


# ## Adding the weather data

# In[14]:


weather_df = pd.read_csv("../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv")
# load the weather data
weather_df.head()


# We observe the value "T" in our data, which means corresponding column has occured in trace. To facilitate our graphs and calculations we will make these to 0.

# In[15]:


weather_df.replace(to_replace = "T", value = 0.0, inplace = True)


# In[16]:


weather_df["snow fall"] = weather_df["snow fall"].apply(lambda x : pd.to_numeric(x))
weather_df["snow depth"] = weather_df["snow depth"].apply(lambda x : pd.to_numeric(x))
weather_df["precipitation"] = weather_df["precipitation"].apply(lambda x : pd.to_numeric(x))
# make the column datatypes as float


# In[17]:


def convert_to_date_format(row):
    """
    Convert given string to datetime format
    """
    
    row_lst = row.split("-")
    return datetime.date(year = int(row_lst[2]), month = int(row_lst[1]), day = int(row_lst[0]))

weather_df["date"] = weather_df["date"].apply(convert_to_date_format)


# In[18]:


df = pd.merge(df, weather_df, on = "date", right_index=True)
# add the weather data
weather_df["date"] = pd.DatetimeIndex(weather_df["date"])


# ## Data after preprocessing

# In[19]:


df.head()


# ## Conducting Exploratory Data Analysis 

# ### Check for null values
# 
# Since the count of the given training data and the route data did not match (training data had one count more). So, we will remove this row.

# In[20]:


df.dropna(inplace = True, how = "any")
# drop rows with empty values


# ### Ride time trend from training data 

# In[21]:


trip_dur_desc = df["trip_duration"].describe()
# get the description about the data


# In[22]:


trip_dur_desc


# We see the minimum travel time to be one second and maximum value goes till 40 days (approx). Both of these data are weird in terms of commute time. 
# 
# So we will filter out data which are less than 15 seconds and more than 18,000 seconds. 

# In[23]:


df = df[(df["trip_duration"] > 15) & (df["trip_duration"] < 18000)]
# filter out data


# In[24]:


mini_log = np.log10(trip_dur_desc["min"])
maxi_log = np.log10(trip_dur_desc["max"])
n_bins = int(np.log2(trip_dur_desc["count"])) + 1
log_bins = np.logspace(mini_log, maxi_log, n_bins)


# For plotting the histogram, we will use the Sturge's formula to get the number of bins, which is given by
# k = log2(n) + 1

# In[25]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5), dpi = 96)
ax1.boxplot(x = df["trip_duration"], showfliers = True, showmeans = True)
ax1.set_yscale("log")
ax1.set_title("Box plot of trip duration (seconds) in log scale")
ax2.hist(x = df["trip_duration"], bins = log_bins)
ax2.set_xscale("log")
ax2.set_xlabel("trip duration")
ax2.set_ylabel("count")
ax2.set_title("Histogram of trip duration")


# In[26]:


trip_dur_desc


# From the above boxplot and histogram, we have a better idea about the trip duration distribution. Both the boxplot and histogram are plotted on log scale (base 10) to facilitate the plotting of huge number of data points.
# 
# We observe the average trip duration is 959 seconds (16 minutes) which gives us an idea that most of the trips have been short.
# 
# Going ahead we could see that the minimum value is 1 second, which is very weird and the maximum value is 3526282 seconds (40 days) which is so weird as well. As we go further we will filter out these types of values. Since, there has been such outliers we will plot the locations on map to have a better idea about the taxi commutes.

# ###  Ride time duration across both venders

# In[27]:


sns.set()
g = sns.FacetGrid(df, col = "vendor_id", size = 6, aspect = 1)
g.map(sns.boxplot, "trip_duration", orient = "v")
sns.set(font_scale = 1.5)
plt.yscale("log")
plt.suptitle("Box plot of trip duration (seconds) in log scale across vendors")
plt.subplots_adjust(top = 0.85)


# In[28]:


sns.set(font_scale = 1.5)
g = sns.FacetGrid(df, col = "vendor_id", size = 6, aspect = 1)
g.map(sns.distplot, "trip_duration", bins = log_bins, kde = False)
plt.suptitle("Histogram of trip duration (seconds) across vendors")
plt.xscale("log")
plt.subplots_adjust(top = 0.85)


#  ## Pickup locations 
# 
# Now lets try to visualize the pickup location on NYC map. Considering the huge points of data, we will plot 2000 random points on the map.

# In[29]:


sample_df = df.sample(2000)
# get 10% of data


# In[30]:


loc = (sample_df["pickup_latitude"][0], sample_df["pickup_longitude"][0])
# first location


# In[31]:


def add_pickup_loc_to_map(row):
    loc = (row["pickup_latitude"], row["pickup_longitude"])
    if row.vendor_id == 1:
        col = "orange"
    else:
        col = "blue"
    folium.CircleMarker(location = loc, radius = 2.5, color = col, fill_opacity = 0.5,
                       fill_color = col, weight = 1,
                       popup = "Trip ID:{}\nVendor ID:{}".format(row.name,
                       row.vendor_id)).add_to(pickup_cluster)
    
def add_dropoff_loc_to_map(row):
    loc = (row["dropoff_latitude"], row["dropoff_longitude"])
    if row.vendor_id == 1:
        col = "orange"
    else:
        col = "blue"
    folium.CircleMarker(location = loc, radius = 2.5, color = col, fill_opacity = 0.5,
                       fill_color = col, popup = "Trip ID:{}\nVendor ID:{}".format(row.name,
                                                                                  row.vendor_id),
                        weight = 1).add_to(dropoff_cluster)


# In[32]:


nyc_pickup_map = folium.Map(location = loc, tiles = "CartoDB positron", zoom_start = 11)
pickup_cluster = folium.MarkerCluster().add_to(nyc_pickup_map)


# In[33]:


sample_df.apply(add_pickup_loc_to_map, axis = 1)
nyc_pickup_map


# In[34]:


nyc_dropoff_map = folium.Map(location = loc, tiles = "CartoDB positron", zoom_start = 11)
dropoff_cluster = folium.MarkerCluster().add_to(nyc_dropoff_map)


# In[35]:


sample_df.apply(add_dropoff_loc_to_map, axis = 1)
nyc_dropoff_map


# #### Observations
# 
# From the pickup and dropoff location map, we can observe that the locations are heavily crowded around Manhattan.
# 
# 

# ### Passenger count analysis 

# In[36]:


sns.set()
g = sns.FacetGrid(df, col = "vendor_id", size = 6, aspect = 1)
g.map(sns.boxplot, "passenger_count", orient = "v")
sns.set(font_scale = 1.5)
plt.suptitle("Box plot of passenger count across vendors")
plt.subplots_adjust(top = 0.85)


# In[37]:


sns.set()
g = sns.FacetGrid(df, col = "vendor_id", size = 6, aspect = 1)
g.map(sns.distplot, "passenger_count", bins = 10, kde = False)
sns.set(font_scale = 1.5)
plt.suptitle("Histogram of passenger count across vendors")
plt.subplots_adjust(top = 0.85)


# From the above boxplot and histogram, we see that for more than one passenger, people prefer vendor 2 and vendor 1 has most of its rides with one passenger.
# 
# 1. Vendor 1 might does not have vehicles that supports 4+ passengers.
# 2. Its obvious the minimum number of seats available in a taxi would be 3 (excluding the driver), still vendor 2 wins for passenger count 2 and 3.

# In[38]:


sns.set_color_codes()
g = sns.FacetGrid(data = df, col = "vendor_id", size = 6)
g.map(sns.boxplot, "passenger_count", "trip_duration", palette = "Blues")
plt.yscale("log")
plt.suptitle("Boxplot of trip duration with various passenger counts for each vendor")
plt.subplots_adjust(top = 0.85)


# We obseve a weird passenger count from the above graph. Passenger count cannot be zero. It doesn't make any sense. Let's see the trip duration for this condition.

# In[39]:


pass_count_zero_df = df.query(expr = "passenger_count == 0")
# data with passenger count zero
pass_count_zero_df.sort_values(by = "trip_duration", inplace = True)


# In[40]:


print("Data points with passenger count zero is {}".format(len(pass_count_zero_df)))


# In[41]:


plt.scatter(x = range(0, len(pass_count_zero_df)), y = pass_count_zero_df["trip_duration"])
plt.yscale("log")
plt.title("trip duration when the passenger count is zero")
plt.ylabel("trip duration")
plt.xlabel("trip number")


# From the graph we can infer taxis have commuted for certain time with no passenger. It's obvious that these cases are close to impossible. So, we will remove the rows of that nature.

# In[42]:


df = df.query("passenger_count != 0")
# remove rows with passenger count is zero


# In[43]:


sns.set_color_codes()
g = sns.FacetGrid(data = df, col = "vendor_id", size = 6)
g.map(sns.boxplot, "passenger_count", "trip_duration", palette = "Blues")
plt.yscale("log")
plt.suptitle("Boxplot of trip duration with various passenger counts for each vendor")
plt.subplots_adjust(top = 0.85)


# Now we have removed passenger with count zero and we could still passenger counts such as 8, 9. Also, we see that passenger count does not seem to influence the trip duration.

# In[44]:


corr = np.corrcoef(x = df["trip_duration"], y = df["passenger_count"])
print("Correlation between trip duration and passenger count: {:.4f}".format(corr[0][1]))


# In[45]:


sns.set_color_codes()
g = sns.FacetGrid(data = df, col = "vendor_id", size = 6)
g.map(sns.boxplot, "passenger_count", "distance", palette = "Blues")
plt.yscale("log")
plt.suptitle("Boxplot of distance with various passenger counts for each vendor")
plt.subplots_adjust(top = 0.85)


# In[46]:


corr = np.corrcoef(x = df["distance"], y = df["passenger_count"])
print("Correlation between distance and passenger count: {:.4f}".format(corr[0][1]))


# From the above graph and correlation value we can conclude that passenger count does not influence the distance parameter as well.

# ## Analysis based on date 

# In[47]:


start_date = min(df["date"])
end_date = max(df["date"])


# In[48]:


print("The start and end date are as follows")
print(start_date.ctime())
print(end_date.ctime())


# In[49]:


date_df = df[["vendor_id", "date"]]
date_df = date_df.assign(trip_count_per_day = df.index)


# In[50]:


date_df_group = date_df.groupby(["vendor_id", "date"])


# In[51]:


date_count_df = date_df_group.count()


# In[52]:


mon_fri_dates = weather_df[(weather_df["date"].dt.dayofweek == 0) | (weather_df["date"].dt.dayofweek == 4)]["date"]


# In[53]:


ax = date_count_df.unstack(0).plot(figsize = (20, 5))
ax.legend(["vendor 1", "vendor 2"])
plt.title("Trip counts per day across vendors")
plt.ylabel("Count")
for i in range(len(mon_fri_dates)):
    if i % 2 == 0:
        color = "red"
    else:
        color = "violet"
    ax.axvline(x = mon_fri_dates.iloc[i], color = color, alpha = 0.9, linestyle = "--")


# From the above graph we could see a common pattern among both vendors. There is a constant rise and fall in the graph and two steep falls. Both these steep falls seems to come during Jan. end and May end.
# 
# Making a wild guess, the steep during Jan. might be because of winter and May might be due to some holiday.
# 
# The red stripped lines corresponds to Friday and violet color corresponds Monday. So, the block between red and violet represents weekend. We see the number of rides decreases during the weekend. This might be because people use taxis to commute for their work during weekdays.
# 
# The dip during May end might be because of the memorial day weekend. So, people don't commute to work on May 30th, 2016 (Memorial day) and since its a long weekend, many people might be on vacations as well.

# In[54]:


weather_df_train = weather_df[weather_df["date"] <= end_date]
# filter out for dates that are in training


# In[55]:


weather_df_train[["precipitation", "snow fall", "snow depth"]].plot(figsize = (12, 5))
plt.title("Weather across all days in training set")
plt.axvline()


# From the above graph we see that there has been a huge snow fall during Jan. end which has lead to high snow depth as well. This definitely should have contributed to the dip in the number of commutes per day. To be precise this dip has happened on Jan 23rd, 2016.

# #### **Grouping based on day** 

# In[56]:


day_df = df[["day", "vendor_id"]]
day_df = day_df.assign(trip_id = df.index)
day_df_group = day_df.groupby(["vendor_id", "day"])
day_count_df = day_df_group.count()


# #### **Grouping based on hours **

# In[57]:


hour_df = df[["hour", "vendor_id"]]
hour_df = hour_df.assign(trip_id = df.index)
# get the hour data


# In[58]:


hour_df_group = hour_df.groupby(["vendor_id", "hour"])
hour_count_df = hour_df_group.count()


# In[59]:


ax = hour_count_df.unstack(0).plot(figsize = (14, 5))
ax.legend(["vendor 1", "vendor 2"])
plt.title("Trip counts per hours across vendors")
plt.ylabel("Count")


# Based on the above graph we see that taxis are not being utilized much during early mornings, which is obvious. As the day starts we see an increase in the usage and it peaks during the evening time and gradually decreases during night time.
# 
# This is pattern one would expect for any city!

# In[60]:


hour_day_group_df = df[["hour", "day", "trip_duration"]].groupby(["hour", "day"]).mean()


# In[61]:


ax1 = hour_day_group_df.unstack(1).plot(figsize = (18, 8))
ax1.legend(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("Trip duration across hours for all days")
plt.ylabel("Trip duration (seconds)")


# From this line graph we have a key observation to make, the time taken duration morning hours is pretty low during the weekends and quite high during early morning compared to weekdays.
# 
# This might be because of people's work and night life culture!

# ## Influence of holidays on trip counts and duration
# 
# We saw above that because of Memorial day, there was a dip in the taxi counts. Let's see if other holidays have some effect as well.

# In[62]:


holiday_data_df = df.query("holiday_or_not == 1")
# trips done on holidays alone
holiday_not_df = df.query("holiday_or_not != 1")
# trips that are done on non-holiday days


# In[63]:


holiday_data_df_group = holiday_data_df[["date", "vendor_id"]].groupby("date").count()
holiday_not_df_group = holiday_not_df[["date", "vendor_id"]].groupby("date").count()


# In[64]:


ax1 = holiday_data_df_group.plot(figsize = (14, 5))
holiday_not_df_group.plot(ax = ax1)
ax1.legend(["holiday/event", "not holiday/event"])
plt.title("Number of trips done per day based on holiday or not a holiday")
plt.ylabel("count")


# 

# ## Machine learning - regression to predict the trip time

# In[67]:


df.columns


# In[68]:


columns_to_remove = ["dropoff_datetime", "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag",
                    "minute", "second", "date", "distance", "best_travel_time", "maximum temperature",
                    "minimum temperature", "average temperature"]


# 
