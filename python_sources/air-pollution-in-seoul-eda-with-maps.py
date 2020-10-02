#!/usr/bin/env python
# coding: utf-8

# # Air Pollution in Seoul
# 
# ![](http://img.koreatimes.co.kr/upload/newsV2/images/201912/191819dec7dd4c1486a302ed4748f4fc.jpg/dims/resize/740/optimize "South Korea, March 5, 2019)
# *The Korea Times. Seoul, South Korea. Courtesy of Kim Joon-hee*
# 
# It is common knowledge that air pollution can cause several problems in the environment and in our health. The photo above was taken on December 11th, 2019 and shows how it can severaly impacted Seoul landscapes. In this occasion, a smog of ultrafine dust, coming all the way from China, lasted two days and made the local government issue emergency emission reduction measures. According to [The Korea Times](https://www.koreatimes.co.kr/www/nation/2020/02/281_280126.html), the Air Quality Forecasting Center, filliated with the Ministry of Environment, reported that on December 11th at 10 p.m the concentration of PM2.5 particles was about 118 micrograms per cubic meter in Seoul. 
# 
# Given that context, my intention in this study was to understand and visualize how air pollution is distributed geographically in the 25 districts that compose the country's capital. To do so,I analyzed a dataset containing hourly collected records of six pollutants (SO2, NO2, CO, O3, PM10, PM2.5) during a period of three years. In Seoul, each district has its own station measuring this set of pollutants, so I could analyze both pollution as whole and segregated by region.
# 
# This kernel provides maps to show the concentrations of gases and particles in Seoul as well as time series analysis to figure out cycles, sazonality and tendency. In the appendix I also analyze if there is a pattern in the occurence of errors in the stations. **This is my first kernel in Kaggle doing EDA and I'm still learning how to this properly. Feel free to comment below suggestions or errors, it really helps me a lot. And if like this notebook, please give it an upvote. Thank you!**
# 
# 
# ## Table of contents
# 
# - [Importing data and libs](#importing-data)
# - [Data preparation](#data-prep)
# - [Exploratory analysis](#eda)
#     * [Where are stations located?](#eda-map-station-loc)
#     * [Pollution levels](#eda-pol-level)
#     * [Does pollutants have correlation?](#eda-pol-corr)
#     * [How pollutant concentrations varies with location?](#eda-pol-local)
#     * [What are the most and least polluted regions of Seoul?](#eda-pol-most-least)
#     * [What really happened on that December 11th?](#eda-11-dec)
#     * [Does the concentration of pollutants have sazonality and tendency? (to be expanded)](#eda-pol-ts-daily)
# - [Appendix](#appendix)
#     * [What happens when an instrument is not working properly?](#eda-inst-not-work)
#     * [How many times did an instrument have status different of normal?](#eda-inst-dif-status)
#     * [Is there a pattern in when a instrument stop working?](#eda-inst-when)
#     * [Are there regions with more instrument problems?](#eda-inst-where)

# ## Importing data and libs <a id ='importing-data'></a>

# In[ ]:


# Standard packages
import json

# Libs to deal with tabular data
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import geopandas as gpd

# Plotting packages
import seaborn as sns
import matplotlib.pyplot as plt

# Lib to create maps
import folium 
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster

# To display stuff in notebook
from IPython.display import display, Markdown


# In[ ]:


# Reading Air Pollution in Seoul
stations = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')
measurements = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')
items = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')


# In order to create maps showing the 25 district borders, I utilized a GeoJSON file available at [this GitHub repository](https://github.com/southkorea/seoul-maps). This is simply a JSON file containing a specific standardized structure to store geographic data. The coordinate reference system used is the one used in GPS, called WGS 84.   

# In[ ]:


# Reading GeoJSON with Seoul administrative borders
with open('/kaggle/input/maps-of-seoul/juso/2015/json/seoul_municipalities_geo_simple.json', 'r') as file:
    district_borders = json.loads(file.read())


# ## Data preparation <a id ='data-prep'></a>
# 
# ### Pollutants <a id ='data-prep-pol'></a>
# 
# This table shows informations about each pollutant, such as unit of measurement and levels of quality.

# In[ ]:


print('Shape:', items.shape)
items


# #### Column data types

# In[ ]:


items.dtypes


# #### Parsing data and creating auxiliar structures

# In[ ]:


# Adding unit to the item name
items['Item name (unit)'] = items['Item name'] + ' (' + items['Unit of measurement'].str.lower() + ')'

# Creating a dict of item codes to item names.
items_dict = {row['Item code']: row['Item name (unit)'] for idx, row in items.iterrows()}


# In[ ]:


# This is a function generator that creates functions to say if a measurement is good, normal, bad or very bad.
def evaluation_generator(good, normal, bad, vbad):
    def measurement_evaluator(value):
        if(pd.isnull(value) or value < 0):
            return np.nan
        elif(value <= good):
            return 'Good'
        elif(value <= normal):
            return 'Normal'
        elif(value <= bad):
            return 'Bad'
        else:
            return 'Very bad'
        
    return measurement_evaluator

# A dictionary that maps pollutants to functions that evaluate the measurement level.
evaluators = {
    row['Item name (unit)']: evaluation_generator(row['Good(Blue)'], row['Normal(Green)'], row['Bad(Yellow)'], row['Very bad(Red)']) 
    for idx, row in items.iterrows()
}


# ### Stations<a id ='data-prep-sta'></a>
# 
# The following DataFrame shows the location of each station. There are 25 stations each one located in a different district.

# #### First 5 rows

# In[ ]:


print('Shape:', stations.shape)
stations.head()


# #### Column data types

# In[ ]:


stations.dtypes


# #### Parsing data and creating auxiliar structures

# In[ ]:


stations_dict = {row['Station code']: row['Station name(district)'] for idx, row in stations.iterrows()}


# ### Measurements <a id ='data-prep-mea'></a>
# 
# This table contains 150 (25 distrincts and 6 pollutants) time series stacked in a single DataFrame. Each row represents a measurement of a specific station and a specific pollutant. The value presented here in the column 'Average value' is the average value of the gas or particle concentration in the 1 hour frame of time before the date specified in the column 'Measurement date'.
# 
# Notice that instruments doesn't work 24/7 because they sometimes may need repair or calibration. So in this dataset we can have problematic measurements. Below you can see a Python dict containing a mapping from instrument status code to description.

# #### Five random rows

# In[ ]:


print('Shape:', measurements.shape)
measurements.sample(5, random_state=42)


# #### Column data types

# In[ ]:


measurements.dtypes


# #### Parsing data and creating auxiliar structures

# In[ ]:


# Pivoting table to reduce number of rows
measures = measurements.pivot_table(index=['Measurement date', 'Station code', 'Instrument status'], columns='Item code', values='Average value').reset_index()
measures.columns = measures.columns.rename('')


# In[ ]:


# Replacing meaningless numbers by labels 
intrument_status = {
    0: 'Normal',
    1: 'Need for calibration',
    2: 'Abnormal',
    4: 'Power cut off',
    8: 'Under repair',
    9: 'Abnormal data',
}
measures['Instrument status'] = measures['Instrument status'].replace(intrument_status)
measures['Station code'] = measures['Station code'].replace(stations_dict)
measures = measures.rename(columns=items_dict)

# Renaming columns
measures = measures.rename(columns={
    'Measurement date': 'Date',
    'Station code': 'Station',
    'Instrument status': 'Status'
})

# Adding levels 
for pol, func in evaluators.items():
    measures[pol.split()[0] + ' Level'] = measures[pol].map(func)
    
# Casting
measures['Date'] = pd.to_datetime(measures['Date'])

# Adding date related columns
weekday_dict = {
    0:'Monday',
    1:'Tuesday',
    2:'Wednesday',
    3:'Thursday',
    4:'Friday',
    5:'Saturday',
    6:'Sunday'
}
measures['Month'] = measures['Date'].dt.month
measures['Year'] = measures['Date'].dt.year
measures['Hour'] = measures['Date'].dt.hour
measures['Day'] = measures['Date'].dt.weekday.replace(weekday_dict)


# #### First five rows of the parsed measurements dataset

# In[ ]:


print('Shape:', measures.shape)
measures.head()


# #### First and last date

# In[ ]:


print('First date:', str(measures['Date'].min()))
print('Last date:', str(measures['Date'].max()))


# ## Exploratory analysis <a id ='eda'></a>
# 
# ### Where are stations located? <a id ='eda-map-station-loc'></a>
# 
# Reported concentrations of pollutants can vary depending on the address of the stations. In other words, stations near dense populated areas may yield higher values than in areas that have more parks or preserved nature. Below you can see a comparision of the station locations and district borders. Notice that station positions within each district vary and there are areas more well covered than others. For instance, in the central region of Seoul the stations of Dongdaemun-gu and Jongno-gu are very close to each other. However, there is a large portion of land uncovered in the south.
# 
# **Click on the marker to see the district name!**

# In[ ]:


stations_map = folium.Map(location=[37.562600,127.024612], tiles='cartodbpositron', zoom_start=11)

# Add points to the map
for idx, row in stations.iterrows():
    Marker([row['Latitude'], row['Longitude']], popup=row['Station name(district)']).add_to(stations_map)
    
# Adding borders
folium.GeoJson(
    district_borders,
    name='geojson'
).add_to(stations_map)

# Display the map
stations_map


# ### Pollution levels <a id='eda-pol-level'><a/>
# 
# In order to have a fair overview of the pollution in Seoul, I first remove instrument observations that are not considered normal. Then, I average the pollutant levels of each district so that there is a single value representing the overall air quality of the city for each pollutant and time stamp. In other words, I transformed 150 (25 x 6) time series in only 6 by averaging district concentrations.

# In[ ]:


bad_measures = measures.loc[measures['Status'] != 'Normal', :]
all_measures = measures.copy()
measures = measures.loc[measures['Status'] == 'Normal', :]
overview = measures.groupby('Date').mean().loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)']

# Adding levels 
for pol, func in evaluators.items():
    overview[pol.split()[0] + ' Level'] = overview[pol].map(func)


# #### Five random rows of Seoul pollution levels

# In[ ]:


print('Shape:', overview.shape)
overview.sample(5, random_state=42)


# #### Pollution levels along the three years time span 

# In[ ]:


fig, ax = plt.subplots(1, 6, figsize=(25, 6))
fig.suptitle('Distribution of pollutants', fontsize=16, fontweight='bold')
for n, pollutant in enumerate(evaluators.keys()):
    sns.boxplot(data = overview[pollutant], ax=ax[n])
    ax[n].set_title(pollutant)
plt.show()


# According to the box plots above, the shape of the distributions are almost like a bell (normal) but slightly right skewed. Also, notice that there  amount of outliers, specially in the PM10 and PM2.5 distributions.

# In[ ]:


general = overview.describe().loc[['min', 'max', 'mean', 'std', '25%', '50%', '75%'],:].T
general['level'] = None
for idx, row in general.iterrows():
    general.loc[idx, 'level'] = evaluators[idx](row['mean'])
    
general.T


# As we can see above, the pollutant levels in Seoul are good. Depite being one of the largest cities of the world and being densily urbanized, it has a very satisfactory control of pollutants. Two points that could be improved are the concentrations of PM10 and PM2.5, which are considered only normal.

# In[ ]:


level_counts = pd.concat([overview[col].value_counts() for col in overview.loc[:, 'SO2 Level':]], axis=1, join='outer', sort=True).fillna(0.0)
level_counts = level_counts.loc[['Very bad', 'Bad', 'Normal', 'Good'], :]

level_counts.T.plot(kind='bar', stacked=True, figsize=(8,6), rot=0,
                    colormap='coolwarm_r', legend='reverse')
plt.title('Levels of pollution in Seoul from 2017 to 2019', fontsize=16, fontweight='bold')
plt.show()


# This graph show the counts of each pollutant level along the three observed years. Besides having good overall poullutant levels, the plot shows that Seoul can sometimes still have bad or very bad concentrations of gases and particles.

# ### Does pollutants have correlation? <a id='eda-pol-corr'></a>
# 
# We know that cars and industries are agents responsible for large pollutant emissions. Also, it is very common that they release more than one type of gas at the same time. For example, motor vehicles typically release both CO and NO2. So we except some degree of correlation between the time series.  

# In[ ]:


measures_slice = measures.loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)']
measures_slice.columns = list(map(lambda x: x.split()[0], measures_slice.columns))
correlations = measures_slice.corr(method='spearman')
mask = np.zeros_like(correlations)
mask[np.tril_indices_from(mask)] = True

plt.figure(figsize=(8,6))
ax = sns.heatmap(data=correlations, annot=True, mask=mask, color=sns.color_palette("coolwarm", 7))
plt.title('Correlation between pollutants', fontsize=16, fontweight='bold')
plt.xticks(rotation=0) 
plt.yticks(rotation=0) 
plt.show()


# As we can see above, most combinations of variables have absolute correlation greater than 0.3. The top two correlations are between CO and NO2 and PM10 and PM2.5. So our hypothesis about vehicles emissions proved to be true.

# ### How pollutant concentrations varies with location? <a id='eda-pol-local'></a>
# 
# In order to analyze the data geographically, I first calculated the mean concentration of each pollutant per region. But here we face a problem that pollutants have diffent units of measurement. So I also standardized the 6 distributions of pollutants, each one having 25 points (districts).
# 
# In the following heatmap, the scale is the amount of standard deviations from the mean (Z-score). Negative values indicate below average pollution and positive values indicate the opposite.

# In[ ]:


district_pol = measures.groupby(['Station']).mean().loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)']
district_pol_norm = (district_pol - district_pol.mean()) / district_pol.std()
district_pol_norm.columns = list(map(lambda x: x.split(' ')[0],district_pol_norm.columns))

plt.figure(figsize=(10,10))
sns.heatmap(data=district_pol_norm, cmap="YlGnBu")
plt.title('Comparision of pollutant levels across districts', fontsize=16, fontweight='bold')
plt.xticks(rotation=0) 
plt.show()


# In the maps below you will see the geographical distribution of pollutants. Of course we cannot generalize the information collected in a specific point to the entire region, but it is a good indicator of the characteristics of the district.

# In[ ]:


for col in district_pol_norm.columns:
    pollutant_map = folium.Map(location=[37.562600,127.024612], tiles='cartodbpositron', zoom_start=11)

    # Add points to the map
    for idx, row in stations.iterrows():
        Marker([row['Latitude'], row['Longitude']], popup=row['Station name(district)']).add_to(pollutant_map)

    # Adding choropleth
    Choropleth(
        geo_data=district_borders,
        data=district_pol_norm[col], 
        key_on="feature.properties.SIG_ENG_NM", 
        fill_color='YlGnBu', 
        legend_name='Concentration of {} in Seoul (Z-score)'.format(col)
    ).add_to(pollutant_map)
    
    display(Markdown('<center><h3>{}</h3></center>'.format(col)))
    display(pollutant_map)


# ### What are the most and least polluted regions of Seoul? <a id='eda-pol-most-least'></a>
# 
# In order to answer this question, I calculated the average Z-score of the previous heatmap and produced a single value that represents how much each region is polluted. The map showing these values can be seen below.

# In[ ]:


pollution_map = folium.Map(location=[37.562600,127.024612], tiles='cartodbpositron', zoom_start=11)

# Add points to the map
for idx, row in stations.iterrows():
    Marker([row['Latitude'], row['Longitude']], popup=row['Station name(district)']).add_to(pollution_map)

# Adding choropleth
Choropleth(
    geo_data=district_borders,
    data=district_pol_norm.mean(axis=1), 
    key_on="feature.properties.SIG_ENG_NM", 
    fill_color='YlGnBu', 
    legend_name='Overall pollution in Seoul by region'
).add_to(pollution_map)

pollution_map


# ### What really happened on that December 11th? <a id='eda-11-dec'></a>
# 
# First I want to take a look at the average pollutant concentrations of the 25 districts. To do that I use the DataFrame `overview`.
# 
# #### City overview

# In[ ]:


reported_day_night = pd.Timestamp(year=2019, month=12, day=11, hour=22)

overview.loc[overview.index == reported_day_night, :]


# As you can see, there is nothing wrong with pollution on the reported date and hour said in the introduction! But if we take a look at this exact same day but at 10 a.m., instead, we obtein the following.

# In[ ]:


reported_day_morning = pd.Timestamp(year=2019, month=12, day=11, hour=10)

overview.loc[overview.index == reported_day_morning, :]


# So, probably there was a typing error in the article. Notice that, even though the PM10 and PM2.5 concentrations are really bad, the other pollutants weren't affected. 
# 
# But we still don't get the same value reported because we are looking at the city average.

# #### Records by stations

# In[ ]:


measures.loc[measures['Date'] == reported_day_morning, :'PM2.5 (mircrogram/m3)'].sort_values('PM2.5 (mircrogram/m3)', ascending=False).head(10)


# In the table above I showed the first 10 rows of the pollutant records in the correct hour and sorted by PM2.5 values. Again we couldn't see the reported value of 118 microgram/m3, but this may have happened because we are seeing the average value of an hour. 

# #### Comparison to the entire month 

# In[ ]:


first_day_dec = pd.Timestamp(year=2019, month=12, day=1)
last_day_dec = pd.Timestamp(year=2019, month=12, day=31)

december = overview.loc[(overview.index >= first_day_dec) & (overview.index <= last_day_dec),:]

fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True, constrained_layout=True)
fig.suptitle('Pollutant concentrations on December 2019', fontsize=16, fontweight='bold')
for n, pollutant in enumerate(evaluators.keys()):
    sns.lineplot(data = december[pollutant], ax=ax[n])
    ax[n].set_title(pollutant)
plt.xlabel('Day')
plt.show()


# We can see that all pollutants, except O3, had a spike in the middle of the month. But this spike was much more evident in the last two time series.

# ### Does the concentration of pollutants have sazonality and tendency? (to be expanded) <a id='eda-pol-ts-daily'></a>
# 
# To analyze this, we first take the mean of concentrations per hour and pollutant. 

# In[ ]:


concentration_hour = measures.groupby('Hour').mean()

fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True, constrained_layout=True)
fig.suptitle('Pollutant concentrations along the day', fontsize=16, fontweight='bold')
for n, pollutant in enumerate(evaluators.keys()):
    sns.lineplot(data = concentration_hour[pollutant], ax=ax[n])
    ax[n].set_title(pollutant)
plt.xlabel('Hour')
plt.show()


# Definitely, each pollutant has a different behavior. Some have two peaks per day and some have just one. But one thing in common is that all of them have a valley in the morning. Also, pollutants measured in ppm have smoother curves compared to the ones that are measured in ug/m3.
# 
# In the next graph we will see if there are sazonality and tendency in the pollutant time series.

# In[ ]:


measures_slice = measures.loc[:, ['Date'] + list(evaluators.keys())]
measures_slice['Date'] = measures['Date'].dt.date
concentrations_day = measures_slice.groupby('Date').mean()

fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True, constrained_layout=True)
fig.suptitle('Pollutant concentrations along the years', fontsize=16, fontweight='bold')
for n, pollutant in enumerate(evaluators.keys()):
    sns.lineplot(data = concentrations_day[pollutant], ax=ax[n])
    ax[n].set_title(pollutant)
plt.xlabel('Date')
plt.show()


# ## Appendix <a id='appendix'></a>
# 
# ### What happens when an instrument doesn't work properly? <a id ='eda-inst-not-work'></a>
# 
# #### Random sample of rows with problems

# In[ ]:


bad_measures.loc[:, 'Status':'PM2.5 (mircrogram/m3)'].sample(10, random_state=42)


# #### Rows where measurements should be discarded but were collected 

# In[ ]:


# Showing 
bad_measures.loc[~bad_measures.loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)'].isnull().any(1), 'Status':'PM2.5 (mircrogram/m3)']


# Notice that when the instrument status is different of normal, it can still provides measurements that seem to be right. But it can also give abnormal or negative numbers. Since we don't know how these instruments work and if they can deliver reliable measures even in need of calibration, it is safier to just discard these rows.
# 
# ### How many times did an instrument have status different of normal? <a id='eda-inst-dif-status'><a/>

# In[ ]:


print('Percentage of abnormal measurements:', bad_measures.shape[0] * 100 / all_measures.shape[0])

counts = all_measures['Status'].value_counts()
plt.figure(figsize=(9,7))
plt.title('Distribution of status values', fontsize=16, fontweight='bold')
sns.barplot(x = counts.values, y = counts.index)
plt.show()


# ### Is there a pattern in when a instrument stop working? <a id='eda-inst-when'><a/>
#     
# To answer this question, we group abnormal rows by hour of the day.

# In[ ]:


# Fails by time
#bad_measures.groupby('Year').apply(len)
#bad_measures.groupby('Month').apply(len)
bad_hourly = bad_measures.groupby('Hour').apply(len)

plt.figure(figsize=(9,7))
plt.ylabel('Quantity')
plt.xlabel('Hour')
plt.title('Quantity of abnormal measures by hour', fontsize=16, fontweight='bold')
sns.lineplot(data=bad_hourly)
plt.show()


# In[ ]:


bad_measures_hour = bad_measures.groupby(['Hour', 'Status']).apply(len).rename('Quantity').reset_index().pivot(index='Hour', columns='Status', values='Quantity').fillna(0).astype('int64')

plt.figure(figsize=(15,10))
plt.title('Distribution of the quantity of measurement fails along the day', fontsize=16, fontweight='bold')
sns.heatmap(data=bad_measures_hour, cmap="YlGnBu", annot=True, fmt='d')
plt.show()


# We can notice a few things according to the last two plots:
# 
# - An instrument can be under repair approximately any time of the day. However it is slightly more likely that the repair is being conducted during the morning or the evening.
# - Power problems can occur almost anytime.
# - Calibration is more likely to be required in the beggining of the evening.
# - Abnormal data is produced more frequently in the beggining of the afternoon, around 8 p.m.
# - Analysing the distribution of "need for calibration" and "abnormal data" jointly, I can infer that when an equipment starts to behave unexpectedly, it first change status to "need for calibration". After some time it change status again to "abnormal data", since the lack of calibration should make the instrument output weird numbers or even don't output them at all.
# 
# ### Are there regions with more instrument problems? <a id='eda-inst-where'><a/>
#     
# In this context, I grouped problems by district.

# In[ ]:


# Fails by station
bad_stations = bad_measures['Station'].value_counts()

plt.figure(figsize=(9,7))
plt.title('Amount of times that problems occured by district', fontsize=16, fontweight='bold')
sns.barplot(x = bad_stations.values, y = bad_stations.index)
plt.show()


# In[ ]:


bad_measures_station = bad_measures.groupby(['Station', 'Status']).apply(len).rename('Quantity').reset_index().pivot(index='Station', columns='Status', values='Quantity').fillna(0).astype('int64')

plt.figure(figsize=(15,10))
sns.heatmap(data=bad_measures_station, cmap="YlGnBu", annot=True, fmt='d')
plt.title('Number of records with problems by district', fontsize=16, fontweight='bold')
plt.show()


# We can observe that some districts have high number of bad status, for instance Gwangjin-gu, Gwanak-gu and Seodaemun-gu. Despite not knowing what really happened in these instruments, I suggest a substitution to see if they present less problems. The instrument in Gwanak-gu is probably old, since it has the highest number of status "under repair". 
