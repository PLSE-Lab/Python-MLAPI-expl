#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots
import datetime # manipulating date formats

#Maps
import folium
from folium.plugins import HeatMap

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# settings
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[ ]:


#Load Data
crimes = pd.read_csv("../input/crime.csv", encoding='Windows-1252')


# In[ ]:


#Building DF Districts
data = {'DISTRICT':['A1','A15','A7', 'B2','B3', 'C11', 'C6', 'D14', 'D4', 'E13', 'E18', 'E5',''], 'NAMES':['Downtwon','Charlestown','East Boston','Roxbury', 'Mattapan', 'Dorchester', 'South Boston', 'Brighton', 'South End', 
                           'Jamaica Plain', 'Hyde Park', 'West Roxbury', 'Unknown Location']} 
  
# Create DataFrame 
districts = pd.DataFrame(data) 
  
# Print new DF 
districts


# In[ ]:


#Simple Preview 
crimes.head(5)


# In[ ]:


districts.head(5)


# In[ ]:


#Create dataframe with Day Period
hour_day = ['Daybreak']*6 + ['Morning']*6 + ['Evening']*6 + ['Night']*6

hour = [list(range(0,24))]
for cont in hour:
    df = [(cont * 1),hour_day]
    
df2 = pd.DataFrame.from_dict(df)

period = df2.transpose()
period.columns = ["HOUR", "HDAY_NAME"]
period.head()


# In[ ]:


#Let's add the Column 'District Name'("vlookup" between Crimes and District)
crimes = crimes.merge(districts, on = 'DISTRICT')


# In[ ]:


period['HOUR']=period['HOUR'].astype(int)


# In[ ]:


#Let's add the Column 'District Name'("vlookup" between Crimes and Period)
crimes = crimes.merge(period, on = 'HOUR')
crimes.head()


# After load data, it's possible that some informations dont' be relevant. So, in this case we'll delete 2 columns.

# In[ ]:


#Some columns not be use in this example. So let's drop two: 

crimes = crimes.drop(["INCIDENT_NUMBER", "OFFENSE_DESCRIPTION"], axis=1)

#other way to delete some columns, could be 
#del crimes['INCIDENT_NUMBER']
#del crimes['OFFENSE_DESCRIPTION']


# In[ ]:


#Reorder Columns to better visualization.
crimes = crimes[['OCCURRED_ON_DATE','OFFENSE_CODE','OFFENSE_CODE_GROUP','DISTRICT','NAMES','REPORTING_AREA',
                 'SHOOTING', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR','HDAY_NAME', 'UCR_PART','STREET', 'Lat', 'Long', 'Location']]


# In[ ]:


#Fill NaN values on Shooting
crimes[['SHOOTING']] = crimes[['SHOOTING']].fillna(value='Not Informed')


# In[ ]:


crimes.head(5)


# In[ ]:


#View type data
crimes.dtypes


# How many data we have?

# In[ ]:


print("\nData size (line x column): {} ".format(crimes.shape)) 


# Now, let's view the completeness:

# In[ ]:


#Applying a crosstab to see general data by Year
view_by_Month = pd.crosstab(crimes["MONTH"], crimes["YEAR"], margins = True)
view_by_Month


# Depeding your analysis, will be necessary delete some informations. For example, we can select the period between 2017-01-01 until 2019-12-31. Let's create another dataframe to maitain the principal df.
# 
# Before some manipulations on datetime, we need convert column date to datetime index.

# In[ ]:


crimes['datetime'] = pd.to_datetime(crimes['OCCURRED_ON_DATE'])
crimes = crimes.set_index('datetime')
crimes.drop(['OCCURRED_ON_DATE'], axis=1, inplace=True)
crimes.head()


# In[ ]:


crimes_2017 = crimes['2017-01-01':'2017-12-31']


# In[ ]:


#Applying a crosstab to see general data by Month
view_by_Month = pd.crosstab(crimes_2017["MONTH"], crimes_2017["YEAR"], margins = True)
view_by_Month


# In[ ]:


#Cross by District Name x Month
view_by_District = pd.crosstab(crimes_2017["NAMES"], crimes_2017["MONTH"], margins = True)
view_by_District


# In[ ]:


#This command drop the column and line sum(do this if necessary)
view_by_District = view_by_District.drop('All',axis=1)
view_by_District = view_by_District.drop('All',axis=0)

view_by_District


# In[ ]:


#View barplots by District x Month
view_by_District.plot(kind="bar", figsize=(17,6), stacked=True)


# In[ ]:


#Applying a crosstab to see general data - 12 Months
view_by_Year = pd.crosstab(crimes_2017["YEAR"], crimes_2017["NAMES"], margins = True)
view_by_Year


# In[ ]:


# NaN Info: replace -1 values in Lat/Long
crimes_2017.Lat.replace(-1, None, inplace=True)
crimes_2017.Long.replace(-1, None, inplace=True)


# In[ ]:


# Plot districts "Segmentations"
sns.scatterplot(x='Lat', y='Long', hue='NAMES', alpha=0.01,data=crimes_2017)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# In[ ]:


grid = sns.FacetGrid(crimes_2017, col='HDAY_NAME', row='YEAR', size=8.7, aspect=1.3,  col_order=["Daybreak", "Morning", "Evening", "Night"])
grid.map(plt.hist, 'MONTH', alpha=.7, bins=12)
grid.add_legend();


# In[ ]:


#Unified Info by Hour.
plt.subplots(figsize=(27,5))
sns.countplot(x='HOUR', data=crimes_2017)
plt.title('CRIME AND REQUESTS HOURS')


# In[ ]:


#Applying a crosstab to see general data by Period
view_by_HDAY = pd.crosstab(crimes_2017["HDAY_NAME"], crimes_2017["MONTH"], margins = True)
view_by_HDAY.sort_values(by=['All'], ascending=True)


# In[ ]:


#Applying a crosstab to see general data by Period
view_by_CrimeGroup = pd.crosstab(crimes_2017["OFFENSE_CODE_GROUP"], crimes_2017["HDAY_NAME"], margins = True)
view_by_CrimeGroup.sort_values(by=['All'], ascending=True)
view_by_CrimeGroup.head(10)


# In[ ]:


# 10 Main Crimes and Requests/Day Period
view_by_CrimeGroup.nlargest(11, ['All']) 


# In[ ]:


#Folium crime map
#To other view options, explore documentation tiles.
map_Crime = folium.Map(location=[42.3125,-71.0875], tiles = "OpenStreetMap", zoom_start = 11)

# Add data for heatmp 
heatmap_Info = crimes_2017[crimes_2017.YEAR == 2017]
heatmap_Info = crimes_2017[['Lat','Long']]
heatmap_Info = crimes_2017.dropna(axis=0, subset=['Lat','Long'])
heatmap_Info = [[row['Lat'],row['Long']] for index, row in heatmap_Info.iterrows()]
HeatMap(heatmap_Info, radius=10).add_to(map_Crime)

# Plot Map
map_Crime


# ******Start with Timeseries**

# If you would to start an exploration with Timeseries, you can do this for example...

# In[ ]:


#Select a period...
crimes_ts = crimes['2017-01-01':'2017-06-30']


# In[ ]:


#Delete not relevant columns...
crimes_ts = crimes_ts.drop(["OFFENSE_CODE", "DISTRICT","NAMES","REPORTING_AREA", "SHOOTING","YEAR","MONTH","DAY_OF_WEEK","HOUR","HDAY_NAME",
                            "UCR_PART",	"STREET","Lat","Long","Location"], axis=1)


# In[ ]:


#"Group by"Dat(D)
ts = crimes_ts.resample('D').count()
ts.head()


# In[ ]:


ts.plot(figsize=(30,4), grid=True)
plt.title('TIME SERIES: JAN-JUN')


# ...And after you need validate if this timeseries it's a Stationary or not. This issue needs more time, because envolve more details explanation.
# 
# Thanks for read!

# In[ ]:




