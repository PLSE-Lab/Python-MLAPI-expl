#!/usr/bin/env python
# coding: utf-8

# **<font size=5> SQL Scavenger Hunt </font>**
# 
# I have been aware of the power of BigQuery for a long time. Therefore, I can't be more excited to join this Scavenger Hunt. Hope myself can polish my SQL skills through this Scavenger Hunt. Cheers!

# **<font size=5>Tutorial : How should you start with BigQuery</font>**
# 
# - **First, load the dataset**

# In[ ]:


# import package with helper functions 
import bq_helper

accidents  = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# - **Take a look at the tables**

# In[ ]:


accidents .list_tables()


# - **Since you can browse the table on the right side of the kaggle UI, I only did it once to show that how to load the head of a table**

# In[ ]:


accidents .head("vehicle_2015")


# - **Choose the columns you want and manipulate them with SQL**

# In[ ]:


query = """SELECT COUNT(consecutive_number) as amount,day_of_crash
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY day_of_crash
            ORDER BY amount DESC
        """


# - **Run the Query**

# In[ ]:


accidents_by_day_in_month = accidents .query_to_pandas_safe(query)


# - **See what we got**

# In[ ]:


accidents_by_day_in_month.head()


# - **Visualize the outcome**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.barplot(x=accidents_by_day_in_month.day_of_crash,y=accidents_by_day_in_month.amount)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of Accidents by day")
plt.show()


# In[ ]:


query = """SELECT COUNT(consecutive_number) as accidents,
                  EXTRACT(Hour FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(Hour FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)


# In[ ]:


print(accidents_by_hour)


# In[ ]:


import seaborn as sns

plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.barplot(x=accidents_by_hour.f0_,y=accidents_by_hour.accidents)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of Accidents by Rank of Hour")
plt.show()


# In[ ]:


query = """SELECT COUNT(hit_and_run) AS Amount,registration_state_name AS STATE
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
                 
hit_n_run_by_state = accidents.query_to_pandas_safe(query)


# In[ ]:


hit_n_run_by_state.head()


# In[ ]:


plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.barplot(x=hit_n_run_by_state.Amount,y=hit_n_run_by_state.STATE)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of hit and run of each state")
plt.show()


# **Here I want to dig deeper. Some extra work to make more progress on my SQL skills.**
# 
# My goal is to create a dataframe that I can generate multivariable plot
# 
# I want to have the hit and run I have and join the drunk driver solumn from accident_2015.
# So that I can see the correlation between these two and maybe mark each point by state.
# Here, I assumed the correlation between the amount of hit and run and the amount of drunk driver would be strongly positive.

# In[ ]:


query = """With a AS(
                    SELECT consecutive_number,number_of_drunk_drivers
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                    WHERE number_of_drunk_drivers > 0)
            SELECT v.registration_state_name AS STATE,
                   COUNT(v.hit_and_run) AS Hit_and_run_Amount,
                   SUM(a.number_of_drunk_drivers) as Drunk_Drivers
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` v
            Join a
                ON v.consecutive_number = a.consecutive_number
            WHERE hit_and_run = "Yes"
            GROUP BY 1
            ORDER BY 3 DESC
        """
## You can use number (1,2,3) to stand for the columns you select

DrunkDriveEvent_by_state = accidents.query_to_pandas_safe(query)


# - **About Join, you can refer to [here](https://www.kaggle.com/justjun0321/sql-scavenger-hunt-day-5-done-by-wei-chun-chang)**

# In[ ]:


DrunkDriveEvent_by_state.head()


# In[ ]:


plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.regplot(x="Hit_and_run_Amount", y="Drunk_Drivers", data=DrunkDriveEvent_by_state)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of hit and run vs drunk drivers")
plt.show()


# As I assumed, the correlation is really strong.

# In[ ]:


Northeast_list = ['Connecticut','Maine','Massachusetts','New Hampshire','Rhode Island','Vermont','New Jersey','New York','Pennsylvania']
Midwest_list = ['Illinois','Indiana','Michigan','Ohio','Wisconsin','Iowa','Kansas','Minnesota','Missouri','Nebraska','North Dakota','South Dakota']
South_list = ['Delaware','Florida','Georgia','Maryland','North Carolina','South Carolina','Virginia','District of Columbia','West Virginia','Alabama','Kentucky','Mississippi','Tennesse','Arkansas','Louisiana','Oklahoma','Texas']
West_list = ['Arizona','Colorado','Idaho','Montana','Nevada','New Mexico','Utah','Wyoming','Alaska','California','Hawaii','Oregon','Washington']


# In[ ]:


'Arizona' in West_list


# In[ ]:


#DrunkDriveEvent_by_state['Region'] = 'Other'
#DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE in Northeast_list,'Region']='Northeast'
#DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE in Midwest_list,'Region']='Midwest'
#DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE in Northeast_list,'Region']='South'
#DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE in Northeast_list,'Region']='West'


# > The code keep show the error : ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
# 
# Hope anyone can help me out
# 
# Since I fail to use the smart way to change the value, I can only use the stupid way as input every state name.

# In[ ]:


DrunkDriveEvent_by_state['Region'] = 'Other'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Connecticut','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Maine','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Massachusetts','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'New Hampshire','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Rhode Island','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Vermont','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'New Jersey','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'New York','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Pennsylvania','Region']='Northeast'


# In[ ]:


DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Illinois','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Indiana','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Michigan','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Ohio','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Wisconsin','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Iowa','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Kansas','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Minnesota','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Missouri','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Nebraska','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'North Dakota','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'South Dakota','Region']='Midwest'


# In[ ]:


DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Delaware','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Florida','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Georgia','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Maryland','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'North Carolina','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'South Carolina','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Virginia','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'District of Columbia','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'West Virginia','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Alabama','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Kentucky','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Mississippi','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Tennesse','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Arkansas','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Louisiana','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Oklahoma','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Texas','Region']='South'


# In[ ]:


DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Arizona','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Colorado','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Idaho','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Montana','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Nevada','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'New Mexico','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Utah','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Wyoming','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Alaska','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'California','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Hawaii','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Oregon','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Washington','Region']='West'


# In[ ]:


plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.lmplot(x="Hit_and_run_Amount", y="Drunk_Drivers", data=DrunkDriveEvent_by_state,hue= "Region")

ax.set_xticklabels(rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of hit and run vs drunk drivers")
plt.show()


# I can see that the slope of West region is the highest and that of Northeast is the lowest.
# 
# Also, I can see that the point with the highest "Hit and run amount" and "Drunk drivers" is in Other region.
# 
# And that one is Unknown.

# In[ ]:


DrunkDriveEvent_by_state.head()


# I want to plot State map filled by color demonstrating the amount of Hit and run and that of drunk driver.
# 
# The way I plot refer to [here](https://stackoverflow.com/questions/7586384/color-states-with-pythons-matplotlib-basemap)

# In[ ]:


#from matplotlib.patches import Polygon

#Map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
#Map.readshapefile('st99_d00', name='states', drawbounds=True)

#state_names = []
#for shape_dict in map.states_info:
#    state_names.append(shape_dict['NAME'])
    
#ax = plt.gca()

#seg = Map.states[DrunkDriveEvent_by_state['STATE']]
#poly = Polygon(seg, facecolor='red',edgecolor='red')
#ax.add_patch(poly)

#plt.show()


# OK, now I want to see where did these drunk drivers have accidents.
# 
# So my goal is to plot a US map and have corresponded points on the map 

# In[ ]:


query = """With a AS(
                    SELECT consecutive_number,number_of_drunk_drivers,latitude,
                    longitude,number_of_motor_vehicles_in_transport_mvit
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                    WHERE number_of_drunk_drivers > 0)
            SELECT ROUND(latitude,0) as latitude,
                   ROUND(longitude,0) as longtitude,
                   COUNT(v.hit_and_run) AS Hit_and_run_Amount,
                   SUM(a.number_of_drunk_drivers) as Drunk_Drivers
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` v
            Join a
                ON v.consecutive_number = a.consecutive_number
            WHERE hit_and_run = "Yes"
            GROUP BY 1,2
            ORDER BY 4 DESC
        """

DrunkDriveEvent_map = accidents.query_to_pandas_safe(query)


# In[ ]:


DrunkDriveEvent_map


# In[ ]:


DrunkDriveEvent_map.info()


# Map plotting refer to [here](https://www.kaggle.com/camnugent/geographic-distribution-of-fatal-car-accidents)

# In[ ]:


DrunkDriveEvent_map.Drunk_Drivers = DrunkDriveEvent_map.Drunk_Drivers.astype(float)
DrunkDriveEvent_map.Hit_and_run_Amount = DrunkDriveEvent_map.Hit_and_run_Amount.astype(float)


# In[ ]:


import numpy as np
from mpl_toolkits.basemap import Basemap

Map = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130.,urcrnrlon=-60.,lat_ts=20,resolution='i')
Map.drawmapboundary(fill_color='paleturquoise')
Map.drawcoastlines()
Map.drawcountries()
Map.drawstates()
used = set()

min_marker_size = 0.5
for i in range(0,179):
    x,y = Map(DrunkDriveEvent_map.longtitude[i], DrunkDriveEvent_map.latitude[i])
    msize = min_marker_size * DrunkDriveEvent_map.Drunk_Drivers[i]
    Map.plot(x, y, markersize=msize)
    
plt.show()


# In[ ]:


import numpy as np
from mpl_toolkits.basemap import Basemap

Map = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130.,urcrnrlon=-60.,lat_ts=20,resolution='i')
Map.drawmapboundary(fill_color='paleturquoise')
Map.drawcoastlines()
Map.drawcountries()
Map.drawstates()
used = set()

x,y = Map(DrunkDriveEvent_map['longtitude'].values, DrunkDriveEvent_map['latitude'].values)
Map.plot(x, y, 'ro')
    
plt.show()


# I want to add on size by Drunk_Drivers and color by Hit_and_run_Amount.
# 
# I'm still trying to figure it out how to achieve this
# 
# Hope anyone can help me with this.

# In[ ]:


query = """SELECT COUNT(consecutive_number) as accidents,
                  EXTRACT(Hour FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(Hour FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)


# <font size=4>Feel free to upvote and leave comments</font>
