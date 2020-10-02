#!/usr/bin/env python
# coding: utf-8

# Hello and welcome to this notebook where we will be going over average temperature increase/decrease and plotting it so that we can visualize it clearly!

# In[ ]:


import pandas as pd
import geopandas as gpd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


temp_data=pd.read_csv("../input/daily-temperature-of-major-cities/city_temperature.csv", index_col="Country", usecols=['Country', 'City', 'Month','Day', 'Year', 'AvgTemperature'])
temp_data.index = temp_data.index.str.replace('US','United States')


# Since we have read our data in we need to make sure that no bad data is in here. We are only going to be dealing with the years 1995 and 2019 to see the variation of temperaure over time so we need to remove NAN values (which is why we dropped the state column) and temperatures like -99 which don't make sense.

# We are handling temperatures from 1995 and sorting that data by the Country, City, Month and Day. Also we are making sure that -99 is removed from the dataset.

# In[ ]:


past_temp_data =temp_data.loc[temp_data.Year.isin(['1995'])]
temp_1995=past_temp_data.loc[:,['City', 'Month', 'Day', 'AvgTemperature']]
temp_1995_filtered = temp_1995[temp_1995['AvgTemperature'] != -99.0] 
temp_1995_filtered.loc[temp_1995_filtered.City == 'Algiers']
temp_1995_filtered


# Now we are doing the same thing, but with 2019 so we can compare both data sets.

# In[ ]:


future_temp_data =temp_data.loc[temp_data.Year.isin(['2019'])]
temp_2019=future_temp_data.loc[:,['City', 'Month','Day', 'AvgTemperature']]
temp_2019_filtered = temp_2019[temp_2019['AvgTemperature'] != -99.0] 
temp_2019_filtered


# Now we will merge the data into one data set so it is easy to compare them and take the difference of the temperatures of 2019 and 1995 and sort it by the Month and Day so we can eventually find the average temperature difference from 1995 to 2019.

# In[ ]:


merged_temp_df=pd.merge(temp_2019_filtered, temp_1995_filtered, on=['Country', 'City', 'Month', 'Day'], how='inner')
merged_temp_df['Temp_Difference']=merged_temp_df['AvgTemperature_x']-merged_temp_df['AvgTemperature_y']
merged_temp_df.head()


# Now that we have the 'Temp_Difference' column we can take the mean of the differences by Month and Day so we have one solid average differnce for each World City.

# In[ ]:


temp_difference_df=merged_temp_df.groupby(['Country', 'City']).Temp_Difference.mean()
final_temp_df=temp_difference_df.sort_values(ascending=False)
final_temp_df=final_temp_df.to_frame()
print(final_temp_df.to_string())


# We can plot the data to see which cities had the highest average increase in temperature from 1995 to 2019. We see that Anchorage, Alaska has the highest average increase with about 5.28 degrees.

# In[ ]:


plt.figure(figsize=(40,10))
plt.title("Top 10 Cities With Highest Increase in Temperature 1995 to 2019", fontsize=60)
plt.tick_params(labelsize=15)
plt.xlabel("Country, City", fontsize=20)
sns.barplot(x=final_temp_df.head(10).index, y=final_temp_df.head(10)['Temp_Difference'],palette='autumn')
plt.ylabel("Increase in Temperature", fontsize=20)


# We can also do the opposite and see which contries had the highest average decrease in temperature from 1995 to 2019 and plot the data to make it clear. We can see that Monterry, Mexico has the highest average decrease with about -3.58 degrees.

# In[ ]:


opp_final_temp_df=temp_difference_df.sort_values(ascending=True)
plt.figure(figsize=(40,10))
plt.title("Top 10 Cities With Highest Decrease in Temperature from 1995 to 2019", fontsize=60)
plt.tick_params(labelsize=15)
plt.xlabel("Country, City", fontsize=20)
sns.barplot(x=opp_final_temp_df.head(10).index, y=opp_final_temp_df.head(10),palette='winter')
plt.ylabel("Decrease in Temperature", fontsize=20)


# Finally, it would be nice to plot this data on a world map in order to see the average temperature differences all over the world at once.

# Here we are renaming columns so we can merge the map with the data so that it will show up once we plot it and we are checking to see if it worked with the City Fairbanks in Alaska.

# In[ ]:


map_df = gpd.read_file("../input/worldm/World_Map.shp")
map_dff=gpd.read_file("../input/wcities/a4013257-88d6-4a68-b916-234180811b2d202034-1-1fw6kym.nqo.shp")
map_dff=map_dff.rename(columns = {'CNTRY_NAME':'Country'})
map_dff=map_dff.rename(columns = {'CITY_NAME':'City'}) 
map_dff.loc[map_dff.City =='Fairbanks']


# Now we are plotting the data on the map to see the all the cities with the average temperature differences represented as colored dots.

# In[ ]:


merged = pd.merge(map_dff,final_temp_df[['Temp_Difference']],on='City')
ax=map_df.plot(figsize=(20,12), color='none', edgecolor='gainsboro', zorder=3)
vmin, vmax = -4, 6
merged.plot(column='Temp_Difference',cmap='coolwarm', ax=ax)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmax=vmax, vmin=vmin))
sm._A = []
cbar = plt.colorbar(sm)
ax.set_title('Average Temperature Increase from 1995 to 2020', fontdict={'fontsize': '40', 'fontweight' : '3'})

