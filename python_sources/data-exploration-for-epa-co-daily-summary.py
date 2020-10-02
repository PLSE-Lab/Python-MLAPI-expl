#!/usr/bin/env python
# coding: utf-8

# ### Goal
# 
# We are going to have a cursory look at the EPA CO daily summary dataset. We will approach the dataset without any previous knowledge of it and see what we have to work with. We are also going to look at any cleaning the data could use, whether it's NaNs or duplicate/useless entries..

# Let's get started by importing our libraries and loading our data.

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("../input/epa_co_daily_summary.csv")


# I normally hate using generic variable names like "data", but here this is going to be the starting point for all our work, it'll save us on typing out something like "epa_co_daily_summary_original_data" every time. Let's have a cursory glance at our data to see what we're working with. I always start with looking at the column names (remember, DataFrame.columns isn't a function, so no () after columns).

# In[2]:


data.columns


# That's nice. Right off the bat, I am making the following assumptions just from the headers alone: 
#  * `state_code` corresponds to one of the 50 states of the US (plus potentially some optional ones like DC, Puerto Rico, Guam, etc).
#  * `county_code` is an identifier based on the county the measurement was taken in. Without having seen the data, I assume the county codes are unique (so they won't start counting at 1 for each state). 
#  * `site_num`probably corresponds to the location where the measurement was made. Like the above 2 columns, I expect I'll need to find some lookup table elsewhere.
#  * `parameter_code`is a very generic name, no clue what that is. Guess we'll have to hold off judgement until we get a bit further into the data exploration.
#  * `poc` could stand for a number of different things.
#  * `latitude` seems fairly self explanatory, the latitude of the measurement site. Only left to find out if it is done in degrees, minutes, seconds, or some decimal version of that.
#  * `longitude` similar to above, same question too.
#  * `datum` is likely to be a reference point for the `longitude` and `latitude` values (not uncommon when working with GIS information).
#  * `parameter_name`, delightfully vague again. Likely to be associated with `parameter_code`. 
#  * `sample_duration` sounds like how long the measurement was taken for. This could later be a possible source of variance in the data, so good to keep in mind.
#  * `pollutant_standard` could mean some form of legislation or a safe limit for the chemical.
#  * `date_local` I assume this is a date format. We will likely be sorting and filtering on this value, so we'll take a closer look at this in a bit.
#  * `units_of_measurement` likely to be the unit for the actual concentration measured at the site. I'm expecting something like parts per million, parts per billion, or heaven forbid something like mol/liter. Another key thing to note is whether they're all the same, or if we need to keep in mind some entries are done with a different unit.
#  * `event_type`, again no idea.
#  * `observation_count` sounds like our money variable. This is the actual concentration of CO measured.
#  * `observation_percent` looks to be the confidence they have in the measured value, could be related to time of measurement.
#  * `arithmetic_mean` is an interesting value. It could be related to the measured variable over the time period it was measured. We'll have to look this up.
#  * `first_max_value` sounds like a variable associated with the measurement method.
#  * `first_max_hour`, similar to above, it's starting to sound like we are going to need to read a bit about CO measurement methodologies at the EPA.
#  * `aqi` stands for air quality index (thank you Google), I wonder if it's related to urban area location or maybe `pollutant_standard`.
#  * `method_code` sounds like it relates to how the data was collected.
#  * `method_name` sounds like the name associated with `method_code`
#  * `local_site_name` sounds like a human readable identifier for `site_num`.
#  * `address` this speaks for itself I think, a street address for the location.
#  * `state_name` is the human readable version of `state_code`
#  * `county_name` is the human readable version of `county_code`
#  * `city_name` is the name of the city the measurement was taken.
#  * `cbsa_name` is the [core-based statistical area](https://en.wikipedia.org/wiki/Core-based_statistical_area)
#  * `date_of_last_change` is an interesting one. It means the data may have been corrected in some way after the fact.

# We all know the old adage about what it means to assume. So let's verify the above assumptions by looking at the data itself and using the sidebar/data description acompanying the data.
# 
# In this case we were largely right about a lot of things, but we made one crucial error. The `observation_count` is not our money value. It only shows the number of samples taken during the day. What we want to look at is the `arithmetic_mean`, `first_max_value` and `first_max_hour`, since those are the salient points.

# In[3]:


states = data.state_code.unique()
print(sorted(states))


# Interesting, let's notice the maximum value is 80 here, and we are missing quite a few in our 1-50 range. Let's see which State corresponds to which code and which ones we are missing:

# In[4]:


data.loc[:,['state_code', 'state_name']].drop_duplicates()


# All right, that makes a bit of sense. We can do the same procedure with the county codes and verify that that is the case there too.

# In[5]:


len(data.site_num.unique())


# All right, that means we have data from 273 sites around the North American continent. 

# ### Unique, duplicate and otherwise irrelevant data
# 
# I suspect, after some initial data exploration, that a lot of the data in this csv is duplicate or unique (as in the same value for all entries). This means we have a bunch of extra columns that aren't serving a useful purpose. After verifying that this is indeed the case, I'm going to create a new DataFrame with the filtered data.
# 
# **NOTE** I'm not changing any data or adding anything. This is a streamlined version of the original data, which means our computer doesn't have to load extra data into working memory and the output is cleaner since we don't have to go looking for the data we care about.

# In[6]:


data.parameter_code.unique()
data.poc.unique()
data.datum.unique()
data.parameter_name.unique()
data.units_of_measure.unique()
data.event_type.unique()


# We can deduce here that `parameter_code` 42101 refers to `parameter_name` Carbon Monoxide and it's the same for all elements. I suspect it's a holdover from a larger dataset where this data was taken from.
# 
# I'm going to make the executive decision to also drop the `datum` element. It's not unimportant, but the error between using different datums is measured in a few meters. That's well within the accuracy of any graphing program we will use([read here for more](https://blog.epa.gov/blog/2012/12/shifting-without-datum-documentation/)).
# 
# As suspected, all `units_of_measure` are measured in parts per million, so we can go ahead and write that down (or remember it) and drop that column from our data.
# 
# `event_type` is not a unique `None` everywhere as I'd hoped. So we will keep it in for now. 
# 
# `poc` is also not unique, so we will keep that in as well.
# 
# `method_code` and `method_name` are also duplicates, so we'll keep `method_code` and look it up if we ever need it.
# 
# I'm also going to drop all the geo location except latitude and longitude. We will have our original data for reference in case we want to look up a specific location, but I can't see that extra data being worth keeping 9 extra columns in our data.

# In[7]:


filtered_data = data.loc[:,['poc', 'latitude', 'longitude', 'sample_duration', 'pollutant_standard', 'date_local', 'event_type', 'observation_count', 'observation_percent', 'arithmetic_mean', 'first_max_value', 'first_max_hour', 'aqi', 'method_code', 'data_of_last_change']]


# We'll note we have effectively halved the amount of data we have to load each time we do an operation, without any loss of generality.

# Now, let's plot the locations we have to work with on a map. We want to see what the distribution is of the measuring stations. Remember, this is just the locations of all the locations we have, we don't yet know how complete each station's data is and its quality.
# 
# I've started the map's location in the middle of Kansas, which should provide a decent starting point. 

# In[8]:


location_lat_longs = filtered_data.loc[:,['latitude', 'longitude']].drop_duplicates()


# In[9]:


import folium

location_lat_longs = filtered_data.loc[:,['latitude', 'longitude']].drop_duplicates()
map_of_locations = folium.Map(location=[39, -98.5], zoom_start=3)

for _, location in location_lat_longs.iterrows():
    folium.Marker(location).add_to(map_of_locations)

map_of_locations


# Well that's unexpected. A quick look back at our data shows we made a misjudgement. We learned that there were 273 unique site names (we were also wrong in the North American continent part, one location is off the west coast of Africa), but that doesn't mean there would be only 273 unique latitude and longitude pairs. Apparently there are 1180.
# 
# A quick look at a number of these points shows that there are no obvious signs that they are the same point, but with a minute difference in location (maybe due to the datum). They're spaced out. If you zoom in on the Mexicali area, you can see they all belong to the same area, but are measurements taken all around town.
# 
# We can also see that the Eastern part of the USA fairly evenly covered in monitoring stations, as is the West coast, but the Midwest area shows large gaps (look at Idaho, South Dakota, North/West Texas).
# 
# So we can conclude that a site isn't a unique location. This is good to know.

# ### That's the spatial dimension, what about time?
# We've seen the distribution of the measurements across the continent, now let's take a few subsections and see how complete these measurements are in time.
# 
# Let's start small; Rhode Island.

# In[10]:


newly_filtered_data = data.loc[:,['state_name', 'poc', 'latitude', 'longitude', 'sample_duration', 'pollutant_standard', 'date_local', 'event_type', 'observation_count', 'observation_percent', 'arithmetic_mean', 'first_max_value', 'first_max_hour', 'aqi', 'method_code', 'data_of_last_change']]
rhode_island_data = newly_filtered_data.loc[lambda df: df.state_name == "Rhode Island", :]


# That's not small! 35385 rows is a lot of data for the littlest state in the union. Let's sort the rows by time.

# In[11]:


ri_time = rhode_island_data.sort_values("date_local", ascending=True)
ri_time


# Interesting. We seem to have four data points a day, two from 1 hour measurements and two from 8 hour runs. It seems we have two locations (roughly speaking), where we can see roughly the CO concentration in the city of Providence, and outside of it. 
# 
# What we can't see is the variation over the course of the day (we could guess based on the max value and the max hour, but we still wouldn't only be guessing at the distribution). So we can't make a pretty heatmap of the state of Rhode Island (well we could, but it would be dishonest and not an accurate representation of the data).
# 
# What we can do is make a few quick plots. Let's explore how the CO measures using the 8 hour runs varied in and outside the city. Let's also look at how both of those varied over the course of the year (ie, is there seasonality).

# In[12]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')

ri_8h_data = ri_time.loc[lambda df: df.sample_duration == "8-HR RUN AVG END HOUR", :]
ri_city_8h_data = ri_8h_data.loc[lambda df: df.latitude < 41.83, :]
ri_rural_8h_data = ri_8h_data.loc[lambda df: df.latitude >= 41.83, :]

ri_city_8h_data = ri_city_8h_data.sort_values("date_local", ascending=True)
ri_rural_8h_data = ri_rural_8h_data.sort_values("date_local", ascending=True)

plt.plot(ri_city_8h_data.date_local, ri_city_8h_data.arithmetic_mean, color='red')
plt.plot(ri_rural_8h_data.date_local, ri_rural_8h_data.arithmetic_mean, color='blue')
# ri_rural_8h_data.arithmetic_mean.plot()


# That is a mess. It does seems to show some kind of cyclical nature and it hints at a potential decreasing trend (it also does something weird, indicating somewhere the data isn't formatted quite right . So let's plot a year of the data.

# In[13]:


ri_city_1990 = ri_city_8h_data.loc[lambda df: df.date_local < "1991", :]
ri_rural_1990 = ri_rural_8h_data.loc[lambda df: df.date_local < "1991", :]
plt.plot(ri_city_1990.date_local, ri_city_1990.arithmetic_mean)
plt.plot(ri_rural_1990.date_local, ri_rural_1990.arithmetic_mean, color='blue')


# That looks decent. We can see that the concentrations of CO outside of the city are lower than in the city. We can see some randomness, and a very faint trend that suggests during the winter months more CO is present in the air in both locations.

# Now, let's look at that craziness that happened in 1995

# In[14]:


ri_city_1995 = ri_city_8h_data.loc[lambda df: df.date_local > "1994", :]
ri_city_1995 = ri_city_1995.loc[lambda df: df.date_local < "1996", :]
ri_rural_1995 = ri_rural_8h_data.loc[lambda df: "1994" < df.date_local, :]
ri_rural_1995 = ri_rural_1995.loc[lambda df: "1996" > df.date_local, :]
plt.plot(ri_city_1995.date_local, ri_city_1995.arithmetic_mean)
plt.plot(ri_rural_1995.date_local, ri_rural_1995.arithmetic_mean, color='blue')


# It looks like we're missing some data. Could this be because of the way we filtered the data or just some omissions in the data themselves?

# In[16]:


ri_rural_1995.arithmetic_mean.isnull().sum()


# So they're not NaN (Not a Number) values. What about just missing entries.

# In[17]:


len(ri_rural_1995.arithmetic_mean)


# In[19]:


len(ri_city_1995.arithmetic_mean)


# We found our answer. We are missing data. How we handle this data depends a lot on how much we're missing and where. We could ignore it, we could work around it, or we could fill it in somehow with an average value (between 2 neighboring points or the entire dataset). All of these bring their own trade-offs. 

# This is where I'm going to leave this kernel. I think we did some interesting exploration of the data, found some limitations to the data and hopefully learned a little about CO distributions in the US.
