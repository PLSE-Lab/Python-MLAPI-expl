#!/usr/bin/env python
# coding: utf-8

# This report aims to study the arrests statistics  in the city of Baltimore through Data Visualization using Python Library

# **Import the standard Libraries** 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from pylab import rcParams
from mpl_toolkits.basemap import Basemap
from matplotlib import cm


# **Loading the Data Sets from CSV files**

# In[ ]:


arrests_data = pd.read_csv("../input/BPD_Arrests.csv")


# **Descriptive Statistics**

# In[ ]:


arrests_data.describe()


# In[ ]:


arrests_data.shape


# In[ ]:


arrests_data.info()


# **Analyzing the Arrest Data**</br>

# In[ ]:


arrests_data.head(5)


# **Data Cleaning and Feature extraction**

# In[ ]:


# Seperating the Arrest Date into Days,Month and Year
arrests_data['ArrestDate']      = pd.to_datetime(arrests_data['ArrestDate'])
# Extracting the Timestamp Details
arrests_data['DayofWeeknumber'] = arrests_data['ArrestDate'].dt.dayofweek
arrests_data['DayOfWeek']       = arrests_data['ArrestDate'].dt.weekday_name
arrests_data['MonthDayNum']     = arrests_data['ArrestDate'].dt.day
arrests_data['Month']           = arrests_data['ArrestDate'].dt.month
arrests_data['Year']            = arrests_data['ArrestDate'].dt.year

# Cleanse the Location to extract the Coordniates

# Fill the coordinates value which are not null
coordinates = arrests_data['Location 1'][arrests_data['Location 1'].notnull()] 
locs_lon=[]
locs_lat=[]

# For loop to append latitudes and longitudes to these lists
for loc in coordinates:
    lat, lon = loc[1:-1].split(', ')
    locs_lon.append(float(lon))
    locs_lat.append(float(lat))

# Cleanse the Location to remove unwanted opening and closing brackets
arrests_data.rename(columns={'Location 1': 'Coordinates'}, inplace=True)
# Convert the type to string
arrests_data.Coordinates = arrests_data.Coordinates.astype(str)

# Remove the curly brackets
arrests_data['Coordinates'] = arrests_data['Coordinates'].str.replace('(','')
arrests_data['Coordinates'] = arrests_data['Coordinates'].str.replace(')','')
# Converting The coordinates to Latitude and Longitude 
arrests_data['lat'] = arrests_data.Coordinates.str.split(',').str.get(0)
arrests_data['lon'] = arrests_data.Coordinates.str.split(',').str.get(1)
arrests_data = arrests_data[(arrests_data.lat != 'nan')]
arrests_data = arrests_data[(arrests_data.lon != 'nan')]
arrests_data.lat = arrests_data.lat.astype(float)
arrests_data.lon = arrests_data.lon.astype(float)
# Drop the Column 'Coordinates'
arrests_data.drop({'Coordinates'}, axis=1, inplace=True)
# Dropping the unneccessary columns
arrests_data.drop({'Arrest','Charge'}, axis=1, inplace=True)


# In[ ]:


# Some further Cleansing
arrests_data.Post = arrests_data.Post.astype(str)
# Delete Entries in Arrest Data for which there is no Crime Location
arrests_data = arrests_data[(arrests_data.Post != ' ')]
arrests_data = arrests_data[(arrests_data.Post != 'nan')]
arrests_data.Post = arrests_data.Post.astype(float)


# In[ ]:


arrests_data.head(3)


# **Line Plot to visualize the Arrests/day over the years**

# In[ ]:


# Grouping the data by Year,Month and Day
arrests_day = arrests_data.groupby(['Year','Month','MonthDayNum'])['MonthDayNum'].count()
# Putting the count to Frame
Count = arrests_day.to_frame(name='count').reset_index()
# Figure 
plt.figure(figsize=(9, 5))
plt.plot_date(arrests_data.ArrestDate.unique(), Count['count'], '-',color ='darkred')
plt.ylabel('Number of Arrests')
plt.xlabel('Year')
plt.title('Average Arrests over the year')


# **An increasing trend in arrests is seen** 

# Now we will Plot the Arrest Locations in the area of Baltimore<br>
# I used the Basemap function of mpl_toolkit with the Gall projection, <br>
# Refer http://matplotlib.org/basemap/users/gall.html for understanding the parameters<br>
# CMAP afmhot style is used which is suitable for heatmaps

# In[ ]:


fig = plt.figure(figsize=(7,5))
m = Basemap(projection='gall', llcrnrlat=min(locs_lat), urcrnrlat=max(locs_lat),
            llcrnrlon=min(locs_lon), urcrnrlon=max(locs_lon), lat_ts=min(locs_lat), resolution='c')
x, y = m(pd.Series(locs_lon).values, pd.Series(locs_lat).values)
m.hexbin(x, y, gridsize=100, bins='log', cmap=cm.afmhot);
plt.title('HeatMap of Arrests')


# In[ ]:


arrest_map  = arrests_data.groupby(['DayOfWeek'],as_index=False).count()


# **Arrests by day through pie charts**

# In[ ]:


# Setting predefined colors as, the same needs to be used in Arrests Data also    
colors = ['blue','green', 'orange', 'purple', 'coral','magenta','red']

# Initializating the arrays for size and label
pie_chunk =[]
pie_name=[]

# Running a for loop to map the arrests count per District 
for i in range(len(arrest_map)):
    pie_chunk.insert(i,arrest_map["ArrestLocation"].iloc[i])
    pie_name.insert(i,str(arrest_map["DayOfWeek"].iloc[i]))
     
explode = (0, 0, 0, 0.3, 0, 0, 0)
# Setting the figure size
plt.figure(figsize=(6,6))
# Title
plt.title('Arrests by Day')
# Plotting the figure
plt.pie(pie_chunk, explode=explode, labels=pie_name, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()     


# **Heatmap of crimes over Baltimore**

# In[ ]:


# Creating the Heatmap using Seaborn Library
# Grouping the data based on District and Year and doing an index reset
arrest_dist = arrests_data.groupby(['District', 'Year']).count().reset_index()
# Creating pivot table for the grouped data and set the index,columns,values and the aggregate function
arrest_pivot = arrest_dist.pivot_table(index = 'District', columns = 'Year', values = 'IncidentLocation', aggfunc = np.mean)
# Set the figure size
plt.figure(figsize = (8, 8))
# Create the cmap rules of seaborn parameter cubehelix_palette, start and rot can be used to adjust the intensity of colors and variation
cmap = sns.cubehelix_palette(start = 0, rot = 2, as_cmap = True)
# Create the Heatmap by setting the pivot table, labels, rules and the width 
sns.heatmap(arrest_pivot, xticklabels = 3, cmap = cmap, linewidths = 0.2)
# Set X label
plt.xlabel('Year')
# Set Y Label
plt.ylabel('Districts')
# Set the Title
plt.title('Heat Map of Arrests in various Districts over the years')

