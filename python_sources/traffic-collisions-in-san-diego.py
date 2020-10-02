#!/usr/bin/env python
# coding: utf-8

# # Traffic Collisions in San Diego since 2017
# ![Typical Traffic in San Diego](https://www.spartnerships.com/wp-content/uploads/traffic-768x511.jpg)
# 
# In 2017, the Insurance company **QuoteWizard** ranked San Diego as the **5th worst city in the United States by Drivers**<sup>[1]</sup>. This was right behind right behind Richmond, CA, Riverside, CA, Salt Lake City, UT, and Sacramento, CA (which was ranked the worst). This came as a huge surprise to many readers.
# 
# In this module, we will do some basic analysis/visualizations to explore the public dataset<sup>[2]</sup> of Traffic Collisions in San Diego since 2017, which is maintained by the city of San Diego's Transportation Department. This was of particular interest to me since the dataset is of my local area and could provide some interesting insights given what I know about the dynamic changes in San Diego. While there is a slight risk of outcome bias, I believe having domain knowledge here would only help to either validate the results or challenge my current understanding of the local area.
# 
# ## Table of Contents
# * [Basic Exploratory Analysis and Questions](#section-one)
# * [Most Common Violations Related to Traffic Collisions](#section-two)
# * [Most Dangerous Violations](#section-three)
# * [Traffic Collision Frequency by Day of the Week, Month, and Season](#section-four)
# * [Traffic Collision Frequency by Time of Day](#section-five)
# * [Traffic Collision Frequency by Geography](#section-six)
# * [Conclusion & Looking Forward](#section-seven)
# 
# <sub><a name="myfootnote1">1</a>: QuoteWizard's Methodology used the weighted means calculated from these parameters of two million data points of its users: Accidents, Speeding tickets, DUIs, and Citations (https://quotewizard.com/news/posts/the-best-and-worst-drivers-by-city)</sub>
# 
# <sub><a name="myfootnote2">2</a>: The dataset can be obtained at https://data.sandiego.gov/datasets/police-collisions/</sub>
# 

# ## Packages

# In[ ]:


# Linear Algebra Packages
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime

# Visualization Packages
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')
import folium
from folium.plugins import HeatMap


# In[ ]:


df = pd.read_csv('../input/traffic-collisions-in-san-diego-since-2017/pd_collisions_datasd_v1.csv')


# ## Basic Exploratory Analysis <a id="section-one"></a>

# In[ ]:


print('Total Number of Accidents in Database: {}'.format(df.shape[0]))


# In[ ]:


df.sample(3)


# In[ ]:


print('Records are between {} and {}'.format(df.date_time.min(), df.date_time.max()))


# In[ ]:


# Remove blanks in dataframe with NAN
df = df.replace(r'^\s*$', np.nan, regex=True)


# In[ ]:


df.info()


# 
# ## Questions
# Some interesting questions and prior assumptions come to mind as something to draw insights from our dataset:
# 
# 1. What are the most common violations leading to traffic collisions? (One might assume DUI's or Speeding)
# 2. What violations are most commonly related to injuries/deaths? (One might again assume DUI's)
# 3. Does the data suggest that traffic collisions are more likely on any given day of the week, month, or season? (One might assume weekdays or Friday nights, during tourism season/summer months?)
# 4. Does the data suggest that certain areas of San Diego are more prone to traffic collisions? (One might assume downtown or metro areas)
# 
# Hopefully we are not begging the question too much and, indeed, as our analysis may surprise us with its results!

# ## 1. Most Common Violations related to Traffic Collisions <a id="section-two"></a>
# Let's check our result here using the general violation section which groups similar violations together under a violation code number. We will find out the top 5 codes, and then translate our top 5 violation codes with the documentation provided in the California Vehicle Code (http://leginfo.legislature.ca.gov):

# In[ ]:


charge_type_df = pd.DataFrame(df['violation_section'].value_counts())

# Create a Violation Type Counts Dataframe
viosec_df = pd.DataFrame(df['violation_section'].value_counts())
viosec_df_top = charge_type_df.iloc[:25] # Limit our output top 15 results

# Barplot
plt.figure(figsize=(12,6))
ax1 = sns.barplot(data=viosec_df_top, x=viosec_df_top.index, y='violation_section', palette=("Reds_d"))

# Formatting
sns.set_context("paper")
plt.title('SD Traffic Collision Counts since 2015 by Violation Section Code')
plt.xlabel('Violation Code')
plt.ylabel('Number of Collisions')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
viosec_df_top10 = viosec_df.iloc[:10] # Limit our output top 10 results
viosec_pie = plt.pie(data=viosec_df_top10, x='violation_section', autopct='%.0f%%', shadow=True)
plt.title('SD Traffic Collision Counts since 2015 by Violation (Top 10 Violation Codes)')
plt.legend(viosec_df.index, loc=(1,0.2))
plt.show()


# <b>Code Translations for top 10 Violation Codes</b> (from http://leginfo.legislature.ca.gov:)
# 1. 22107 - Turn a vehicle from a direct course or move right or left when unreasonable/not using turn signals
# 2. MISC-HAZ - Miscellaneous Driving Hazards
# 3. 22350 - Drive a vehicle upon a highway at a speed greater than is reasonable or prudent
# 4. 22106 - Start/backing a vehicle stopped, standing, or parked on a highway
# 5. 21453A - Not stopping red signal or at limit line
# 6. 21703 - Tailgaiting
# 7. 21801A - Not yielding right-of-way to incoming traffic during turn/u-turn
# 8. 21950A - Not yielding to pedestrian traffic
# 9. 21804 - Not yielding right-of-way when merging into traffic
# 10. 21802A - Not stopping at stop-signs/yielding right-of-way at stop signs
# 
# ### Insights
# We observe the top 5 violations related to traffic collissions are:
# 1. Turning movements without signaling
# 2. Misc. hazardous violations (Thanks for recording this so generically SDPD...)
# 3. Unsafe driving speeds
# 4. Starting parked vehicles/backing
# 5. Red or Stopped Vehicles at Limit Line (i.e. Running a Red Light)
# 
# Contrary to our prior assumption, DUI's (ranked 21st) don't even constitute the top 10 violations related to traffic collisions* (One should still refrain from driving under the influence). The top violation - turning without signaling - is far more frequent than all other violations. <b>Please people: use your turn signals! And stop driving so fast!</b>

# ## 2. Most Dangerous Violations <a id="section-three"></a>
# What violations are the most dangerous/deadly? Which ones lead to the most injuries and deaths?

# In[ ]:


# List the unique violations
unique_violations = df.violation_section.unique()
totalViolations = []

# Count each unique violation
for violation in unique_violations:
    deadly_vio_df = df[df.violation_section==violation]
    totalViolations.append((violation, sum(deadly_vio_df.injured), sum(deadly_vio_df.killed)))
    
# Sort and determine top 10 violations by injury and by fatalities
top10_vio_sort_injury = sorted(totalViolations, key=lambda tup: tup[1], reverse = True)[:10]
top10_vio_sort_killed = sorted(totalViolations, key=lambda tup: tup[2], reverse = True)[:10]
top10vio_injury_df = pd.DataFrame(data=top10_vio_sort_injury, columns = ['violation','injured','killed'])
top10vio_killed_df = pd.DataFrame(data=top10_vio_sort_killed, columns = ['violation','injured','killed'])


# In[ ]:


fig = plt.figure()

ax1 = fig.add_axes([0, 0, 1, 1], aspect=1)
plt.pie(x=top10vio_injury_df.injured, autopct='%.0f%%', shadow=True)
plt.legend(top10vio_injury_df.violation, loc=(1,0.3))

ax2 = fig.add_axes([1, 0, 1, 1], aspect=1)
plt.pie(x=top10vio_killed_df.killed, autopct='%.0f%%', shadow=True)
plt.legend(top10vio_killed_df.violation, loc=(1,0.3))

ax1.set_title('Worst Violations by Percentage of Total Injured')
ax2.set_title('Worst Violations by Percentage of Total Killed')
plt.show()


# ### Worst Injury Violations
# 1. 22350 - Speeding
# 2. 22107 - Unsafe turning or not using turn signals
# 3. 21453A - 21453A - Not stopping at stop signs
# 4. MISC-HAZ - Miscellaneous Hazards
# 5. 21801A - Turning vehicles (left or u-turn) not yielding right of way to incoming traffic
# 6. 21703 - Tailgaiting
# 7. 21950A - Not yielding right of way to pedestrians
# 8. 21804 - Not yielding right of way entering a highway or road
# 9. 21802A - Not stopping at intersections
# 10. 22106 - Starting or backing a vehicle stopped, standing, or parked on a highway
# 
# ### Worst Fatality Violations
# 1. 21954A - Pedestrians not yielding the right-of-way to all vehicles (e.g. jaywalking)
# 2. 22107 - Unsafe turning or not using turn signals
# 3. MISC-HAZ - Miscellaneous Hazards
# 4. 22350 - Speeding
# 5. 21453A - Not stopping at stop signs
# 6. 21950A - Not yielding right of way to pedestrians
# 7. 21801A - Turning vehicles (left or u-turn) not yielding right of way to incoming traffic
# 8. 21950B - Pedestrian leaving curb/crosswalk/place of safety into road
# 9. 21650 - Driving on the wrong side of the road
# 10. 21453C - Turning at a red light arrow
# 
# ### Insights
# What is of particular interest here is the difference between violations that lead the highest number of injuries vs. the highest number of fatalities. The top 3 violations that lead to injuries in San Diego is speeding, unsafe turning, and not stopping at stop signs, whereas the top 3 violations that lead to fatalities are jaywalking, unsafe turning, and speeding (if we disregard "miscellaneous hazards"). Many of the violations with high fatality counts are pedestrian related.

# ## 3. Traffic Collision Frequency by the Day of the Week, Month, & Season <a id="section-four"></a>

# In[ ]:


# Strip time from datetime string in our dataset
date_format = "%Y-%m-%d %H:%M:%S"
max_record = datetime.datetime.strptime(str(df.date_time.max()), date_format)
min_record = datetime.datetime.strptime(str(df.date_time.min()), date_format)

# Calculate number of days in our records to calculate mean
df.date_time = pd.to_datetime(df.date_time)
df["day_of_week"] = df.date_time.dt.dayofweek
delta = max_record - min_record
days_count = delta.days
days = list(range(7))
day_mean_df = pd.DataFrame(columns = ['day','mean'])
day_mean_df.day = ['Sun','Mon','Tues','Wed','Thurs','Fri','Sat']

def day_stats(day):
    day_mean = df[df['day_of_week']==day].shape[0]/days_count
    return(day_mean)
    
for day in days:
    day_mean_df.loc[day, 'mean'] = day_stats(day)

plt.figure(figsize=(12,6))
sns.barplot(data=day_mean_df, x='day', y='mean', palette=("Blues_d"))

plt.title('SD Traffic Collision Counts since 2015 by Day of the Week (mean)')
plt.xlabel('Day of the Week')
plt.ylabel('Mean Number of Collisions')
plt.show()


# In[ ]:


# Mean and Std. Dev by Day
df.index = pd.DatetimeIndex(df.date_time)
day_stats_df = pd.DataFrame(df.resample('D').size())
day_stats_df['day_mean'] = df.resample('D').size().mean()
day_stats_df['day_std'] = df.resample('D').size().std()

# Upper and Lower Control Limit
UCL = day_stats_df['day_mean'] + 3 * day_stats_df['day_std']
LCL = day_stats_df['day_mean'] - 3 * day_stats_df['day_std']

# Plot
plt.figure(figsize=(15,6))
df.resample('D').size().plot(label='Accidents per day', color='sandybrown')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
day_stats_df['day_mean'].plot(color='red', linewidth=2, label='Average')
plt.title('Traffic Collisions Timeline', fontsize=16)
plt.xlabel('')
plt.ylabel('Number of Traffic Collisions')
plt.tick_params(labelsize=14)


# In[ ]:


day_stats_df.drop(day_stats_df.tail(1).index, inplace=True) # drop last 1 row (since the records are incomplete for that day)
worst_day = day_stats_df[0].idxmax()
best_day = day_stats_df[0].idxmin()
print ('Worst Day was {}: {} accidents and Best Day was {}: {} accidents'.format(worst_day, day_stats_df[0].max(), best_day, day_stats_df[0].min()))


# ### Insights
# 
# We observe that the histogram of traffic collision frequency to the day the week seems uniformly distributed, meaning that we would expect the likelihood of a traffic collision to be consistent on any given day of the week*. 
# 
# <sub>*Although there might be an argument that there is a rise in traffic collisions during the week as opposed to the weekends. We would need to check the significance via a regression model, which is out of the scope of this particular module. 

# ### Traffic Collision Frequency by Day of the Month/Season
# Let's continue our analysis with some visualizations by month/season. Months would be interesting - are there particular months that are worse than others? And then grouping them by season would provide insights based on our domain knowledge of San Diego (i.e. Summer time is peak tourism season here).

# In[ ]:


# Count frequency of collisions by weekday / appending a month column
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
              'August', 'September', 'October', 'November', 'December']
df["accident_month"] = df.date_time.dt.month
month_df = pd.DataFrame(df['accident_month'].value_counts()).sort_index()
month_df['month_name'] = months
month_df.columns = ['accident_count', 'month']

# Re-order columns
columnsTitles=["month","accident_count"]
month_df=month_df.reindex(columns=columnsTitles)

sns.barplot(data=month_df, x='month', y='accident_count', palette=("Reds_d"))

plt.title('SD Traffic Collision Counts since 2015 by Month')
plt.xlabel('Month')
plt.ylabel('Mean Number of Collisions')
plt.xticks(rotation=90)
plt.show()


# One will quickly notice that the last three months have a disproportionately lower number of accidents. This is not because they are better months; our dataset does not contain 2019's October, November, and December months (they hadn't happened yet). We will instead need to come up with an average accident count by the number of years on record.

# In[ ]:


month1_df = month_df[0:9]
month1_avg_df = pd.DataFrame(month1_df['accident_count'].div(3))

month2_df = month_df[9:12]
month2_avg_df = pd.DataFrame(month2_df['accident_count'].div(2))

merged_df = pd.merge(month1_avg_df, month2_avg_df, how='outer')
merged_df['month'] = months
merged_df

columnsTitles=["month","accident_count"]
merged_df=merged_df.reindex(columns=columnsTitles)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
fig.suptitle("SD Traffic Collisions Frequency by Month", fontsize=16)

ax4a = sns.barplot(x='month', y='accident_count', color='Red', data=merged_df, ax = axes[0])
ax4a.set(xlabel='Month', ylabel='Number of Accidents')

ax4b = plt.pie(x=merged_df.accident_count, labels=months, colors=['red', 'darkorange', 'silver'], autopct='%.0f%%', shadow=True)
plt.show()


# ### Insights
# The data of traffic collisions is somewhat uniformly spread across the months, and our pie chart seems to support this. If we were to take the number of days into consideration for each month, it might cause our data to be even more uniformly distributed (an interesting consideration). This would contradict the notion that more tourism leads to more accidents - not necessarily so! But let's look at a linegraph of accidents over the three years:

# In[ ]:


date_before = datetime.date(2019, 10, 1)
df2 = df[df.date_time < date_before]

month_line_df = df2.resample('M').size()
plt.figure(figsize=(15,6))
month_line_df.plot(label='Total,  accidents per month', color='sandybrown')
month_line_df.rolling(window=12).mean().plot(color='red', linewidth=5)
plt.title('SD Total Traffic Collisions Per Month', fontsize=16)
plt.xlabel('')
plt.show()


# In[ ]:


print("Best Month {0}: {1} accidents".format(month_line_df.idxmin(), month_line_df[month_line_df.idxmin()]))
print("Worst Month {0}: {1} accidents".format(month_line_df.idxmax(), month_line_df[month_line_df.idxmax()]))


# In[ ]:


winter_df = pd.DataFrame(merged_df.iloc[[0,1,11],[1]].sum())
spring_df = pd.DataFrame(merged_df.iloc[[2,3,4],[1]].sum())
summer_df = pd.DataFrame(merged_df.iloc[[5,6,7],[1]].sum())
fall_df = pd.DataFrame(merged_df.iloc[[8,9,10],[1]].sum())

season1_df = pd.merge(winter_df, spring_df, how='outer')
season2_df = pd.merge(summer_df, fall_df, how='outer')
season_df = pd.merge(season1_df, season2_df, how='outer')
season_df['season'] = ['winter', 'spring', 'summer', 'fall']
season_df.rename(columns={0:'accident_count'}, inplace=True)
season_df = season_df.reindex(columns=['season','accident_count'])

plt.figure(figsize=(12,6))
ax5 = sns.barplot(x='season', y='accident_count', color='coral', data=season_df)
ax5.set(xlabel='Season of the Year', ylabel='Number of Accidents')
plt.show()


# ### Insights
# Our line graph provides a little bit more insight here (even though it is zoomed in quite a bit). We see a bit of a trend of a climb in accidents during the summer months, and then a decline during the winter. However, our other graphs seem to point to a uniform distribution, meaning there is not much variation due to the months or season. Hence, our analysis so far warrants further exploration using regression analysis - something to consider in the future!

# ## 4. Traffic Collision Frequency by Time of Day <a id="section-five"></a>

# In[ ]:


df["hour"] = df.date_time.dt.hour
print('The Mean Hour for Traffic Collisions: {number:.{digits}f}'.format(number=df.hour.mean(), digits=2))
print('The Median Hour for Traffic Collisions: {}'.format(df.hour.median()))
print('The Standard Deviation for Traffic Collisions: {number:.{digits}f}'.format(number=df.hour.std(), digits=2))
df.hour.describe()


# Our mean is rather low here - that may be due to the high number of outlier reports that have 00:00 reported in our datset. We should instead consider using the median as our indicator of central tendency. Let's move on to some visualizations of our data in regards to the time of day:

# In[ ]:


plt.figure(figsize=(10,4))
sns.catplot(x='hour', kind='count',height=8.27, aspect=3, color='black',data=df)
plt.show()


# ### Box and Violinplot

# In[ ]:


plt.figure(figsize=(15,6))
df["hour"] = df.date_time.dt.hour
day_dict = {0:'SUN', 1:'MON', 2:'TUES', 3:'WED', 4:'THURS', 5:'FRI', 6:'SAT'}
df['day_of_week_name'] = df['day_of_week'].map(day_dict)
day_time_bplot = sns.boxplot(y='hour', x='day_of_week_name', data=df, width=0.5, palette='colorblind')
plt.xlabel("Day of the Week")
plt.ylabel("Hour of the Day")
plt.title("Boxplot of Traffic Collisions (Day of Week to Hour of Day)")
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
day_time_vioplot = sns.violinplot(x='day_of_week_name', y='hour', data=df)
plt.xlabel("Day of the Week")
plt.ylabel("Hour of the Day")
plt.title("Violinplot of Traffic Collisions (Day of Week to Hour of Day)")
plt.show()


# ### Insights
# First of all, it looks like the time for 00:00:00 is used as a placeholder when times are not known, so we will throw out that data in our initial consideration. From our graph above, it looks like poor times to drive (i.e. higher risk of traffic collision) would be around our rush hour times in San Diego (around 7AM-930AM and 3PM-6PM). As evenings progress, the frequency starts to mellow down and spikes gently around 1AM. It would be interesting to find out more about SDPD's reporting process and find out what would constiute as a 00:00 automatically (perhaps an accident is not reported until the next day's business hours?).

# ## 5. Violations by Geography <a id="section-six"></a>
# What areas of San Diego are most prone to traffic collisions? This is perhaps the most difficult part of our analysis, as we will need to convert addresses in our dataset to latitude and longitude coordinates using geopy and geopandas. This is known as <i>geocoding</i>. The entire process took nearly 13 hours to complete, and a new .csv file was generated from a new dataframe that contains the geocoded dataset.
# 
# ### Geocoding
# We will not run the following code now, as it took aroudn 13 hours to geocode our dataset. We will simply read a new .csv that I made with the addresses converted to lattitude and longitude (many thanks to Abdishakur for his tutorial found at https://towardsdatascience.com/geocode-with-python-161ec1e62b89). The code below is used as reference:
# 
# ```import geopy```
# 
# ```import geopandas```
# 
# ```from geopy.extra.rate_limiter import RateLimiter1```
# 
# ```from geopy.geocoders import Nominatim```
# 
# `locator = Nominatim(user_agent='myGeocoder', timeout=10)`
# 
# `nom=Nominatim(domain='localhost:8080', scheme='http')`
# 
# `geocode = RateLimiter(locator.geocode, min_delay_seconds=1)`
# 
# `df1['location'] = df1['full_address'].apply(geocode)`
# 
# `df1['point'] = df1['location'].apply(lambda loc: tuple(loc.point) if loc else None)`
# 
# `df1[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df1['point'].tolist(), index=df1.index)`

# In[ ]:


# Replace NAN with empty string
df1 = df.replace(np.nan, '', regex=True)
df1["address_number_primary"]= df1["address_number_primary"].astype(str)
df1['full_address'] = pd.DataFrame(df1['address_number_primary']+' '+df1['address_pd_primary']+' '+df1['address_road_primary']+' '+df1['address_sfx_primary']+', SAN DIEGO, CALIFORNIA, USA')

geo_df = pd.read_csv('../input/traffic-collisions-in-san-diego-since-2017/geocoded_pd_collisions_dataset.csv')


# In[ ]:


# Create basic Folium collision map
collision_map = folium.Map(location=[32.7167, -117.1661],
                       tiles = "Stamen Toner",
                      zoom_start = 10.2)

# Add data for heatmap 
geo_heatmap = geo_df[['latitude','longitude']]
geo_heatmap = geo_df.dropna(axis=0, subset=['latitude','longitude'])
geo_heatmap = [[row['latitude'],row['longitude']] for index, row in geo_heatmap.iterrows()]
HeatMap(geo_heatmap, radius=10).add_to(collision_map)
collision_map


# ### Insights
# Our heatmap gives us a sense of what distrits of San Diego are most prone to traffic collisions<sup>[1](#myfootnote3)</sup>: Pacific Beach, Downtown, Uptown, and Greater North Park. One thing to note is that geopy was unable to geocode around 2000 rows due to inconsistent/improper address entry. Further preprocessing on the dataset would be helpful in completing our analysis.
# 
# <sub>[1](#myfootnote3) For a map of San Diego's Districts, see: https://www.sandiego.gov/sites/default/files/council_districts_2018.pdf</sub>

# ## Conclusion and Looking Forward <a id="section-seven"></a>
# We can summarize some of our results as follows
# 1. The worst violation related to high frequency of traffic collisions in San Diego: Turning improperly and not using turn signals.
# 2. The most dangerous violation related to high count of injuries/fatalities in San Diego: Improper pedestrian movements and Speeding
# 3. Our initial visualizations do not cause us to believe that there is a significant correlation between traffic collissions and the day of the week, the month, or the season.
# 4. The worst time of day in terms of traffic collisions is around 3-6PM.
# 5. The worst areas to drive: Pacific Beach, Downtown, Uptown, and Greater North Park.
# 
# Some things to consider for the future:
# 1. Performing some form of regression analysis. It would be interesting to note if this would support/contradict our intial thoughts concering day, month, and season with regards to the frequency of traffic collisions.
# 2. Waiting until the dataset is larger - perhaps five to six years worth of data would yield more concrete results.
# 3. Preproccessing our data more in terms of grouping similar violations codes, appending longitude and latitudes for messier records.

# # Some Further Analysis of My Local Area (Golden Triangle Area)
# Let's go a little further and explore the data with particular focus on the area right in my local area in San Diego.

# In[ ]:


goldentri = df[df['police_beat'].isin(['115', '931'])] # Includes 931, sorrento valley, since police officers often respond from this beat
goldentri_inj = goldentri[goldentri['injured']>1]


# In[ ]:


goldentri_charge = pd.DataFrame(goldentri_inj['violation_section'].value_counts())
golden_top = goldentri_charge.iloc[:25] # Limit our output top 15 results
                                   
plt.figure(figsize=(12,6))
ax1 = sns.barplot(data=golden_top, x=golden_top.index, y='violation_section', palette=("Reds_d"))

# Formatting
sns.set_context("paper")
plt.title('SD Traffic Collisions (with More than One Injury) since 2015 in the Golden Triangle Area')
plt.xlabel('Violation')
plt.ylabel('Number of Collisions')
plt.xticks(rotation=90)
plt.show()


# The top violation in the area right in my vicinity is 21453A - A driver facing a steady circular red signal alone shall stop at a marked limit line, but if none, before entering the crosswalk on the near side of the intersection or, if none, then before entering the intersection, and shall remain stopped until an indication to proceed is shown.
# 
# In other words, running a red light.
# 
# The second highest violation is 22350 is speeding.

# In[ ]:


plt.figure(figsize=(10,4))
sns.catplot(x='hour', kind='count',height=8.27, aspect=3, color='black',data=goldentri)
plt.show()


# Here we see the total number of accidents grouped by the hour over the past three years. This matches a lot of what we saw for San Diego in general: accidents spike near rush hour times (7-9AM and around 3-6PM).

# In[ ]:




