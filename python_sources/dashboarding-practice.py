#!/usr/bin/env python
# coding: utf-8

# This dashboard displays the total number of red camera violations in the city of Chicago, Illinois. It includes a count of total violations, a chart with the number of violations per day, and a map with the locations of the red light cameras and number of violations. This work was completed after viewing the December 2018 Kaggle Dashboard Training. It is hosted online by the Google Cloud Platform service. The page is updated daily at midnight Central Standard Time (CST). The interactive graphs were created through Plotly. 
# 
# Normally, I would hide my raw code, as well as the work I did to clean and manipulate the data. However, I kept it visible for now, to showcase those skill sets.

# In[ ]:


#import packages
import numpy as np 
import pandas as pd

#import data
red_cam_viol_org = pd.read_csv('../input/chicago-red-light-and-speed-camera-data/red-light-camera-violations.csv')
red_cam_loc_org = pd.read_csv('../input/chicago-red-light-and-speed-camera-data/red-light-camera-locations.csv')


# # Identifying Relevant Information for a Dashboard. 
# 
# After looking at the dataset, I think that the following information would be interesting to track in a dashboard: 
# 
# * Number of red camera violations, 
# * Distribution of violations by day of week, 
# * Distrbution of violations by location, and 
# * Distribution of violations by camera ID

# # Data Validation
# 
# Because I will be automatically updating this Dashboard, I want to validate my source dataset to check that the format and content remains as it is now (and to flag the dataset if something significant changes that would affect my Dashboard). 

# In[ ]:


# install csvvalidator 
import sys
get_ipython().system('{sys.executable} -m pip install csvvalidator')


# In[ ]:


# import packages 
from csvvalidator import *

# fields for first dataframe
field_names_1 = ('INTERSECTION', 'VIOLATION DATE', 'VIOLATIONS')

# create validator object
validator_1 = CSVValidator(field_names_1)

# write checks
validator_1.add_value_check('INTERSECTION', str, 'EX1.1', 'Intersection must be a string')
validator_1.add_value_check('VIOLATION DATE', datetime_string('%Y-%m-%d'), 'EX1.2', 'Invalid date')
validator_1.add_value_check('VIOLATIONS', int, 'EX1.3', 'Number of violations not an integer')

# fields for second dataframe
field_names_2 = ('INTERSECTION', 'LONGITUDE', 'LATITUDE')

# create validator object
validator_2 = CSVValidator(field_names_2)

# write checks
validator_2.add_value_check = ('INTERSECTION', str, 'EX2.1', 'Intersection must be a string')
validator_2.add_value_check = ('LONGITUDE', float, 'EX2.2', 'Longitude not a float')
validator_2.add_value_check = ('LATITUDE', float, 'EX2.3', 'Latitude not a float')


# In[ ]:


# import libraries 
import csv
from io import StringIO

# first sample csv
good_data_1 = StringIO("""INTERSECTION,VIOLATION DATE,VIOLATIONS
Test1-Test2,2014-08-05, 5
Test3-Test4,2014-07-11,12
Test5-Test6,2014-07-04,30
""")

# read text in as a csv
test_csv_1 = csv.reader(good_data_1)

# validate first good csv
validator_1.validate(test_csv_1)


# In[ ]:


# second sample csv
good_data_2 = StringIO("""INTERSECTION,LONGITUDE,LATITUDE
Test1-Test2, 41.931791, -87.726979
Test3-Test4, 41.924237, -87.746302
Test5-Test6, 41.923676, -87.785441
""")

# read text in as a csv
test_csv_2 = csv.reader(good_data_2)

# validate first good csv
validator_2.validate(test_csv_2)


# # Data Cleanup and Munging

# In[ ]:


#examine dataframe
red_cam_viol_org.head()


# Uh-oh! I already see that a lot of information regarding the location of the violations is missing. I will quickly remove these columns from my first dataset.

# In[ ]:


#remove unnecessary columns

red_cam_viol = red_cam_viol_org[["INTERSECTION", "CAMERA ID", "ADDRESS", "VIOLATION DATE", "VIOLATIONS"]].copy()
red_cam_viol.head()


# The first piece of information I want is the total number of red camera violations. This should be pretty easy, as the "Violations" column records the number of violations captured by each camera ID on each day. 

# In[ ]:


# number of red light camera violations 
red_cam_viol["VIOLATIONS"].sum()


# Next, I want to track how many violations, on average, occur on each day of the week. It would also be interesting to track when these violations occur, but unfortunately it looks like there is no time series data available. 

# In[ ]:


# number of red light camera violations per day

# convert column to "date time"
red_cam_viol["VIOLATION DATE"] = pd.to_datetime(red_cam_viol["VIOLATION DATE"])

# add new column with day of week 
red_cam_viol["Day of Week"] = red_cam_viol["VIOLATION DATE"].dt.day_name()

# create two dictionarys to sort by day of week 
days = {'Monday' : 1, 'Tuesday' : 2, 'Wednesday' : 3, 'Thursday' : 4, 'Friday' : 5, 'Saturday' : 6, 'Sunday' : 7}
days2 = {1: 'Monday', 2: 'Tuesday', 3 : 'Wednesday', 4: 'Thursday', 5 : 'Friday', 6 : 'Saturday', 7 : 'Sunday'}

# group by day of week, sum number of violations, and sort by day of week
viol_per_day = red_cam_viol.groupby(["Day of Week"])["VIOLATIONS"].count()
viol_per_day = viol_per_day.reset_index()
viol_per_day["Day of Week"] = viol_per_day["Day of Week"].map(days)
viol_per_day = viol_per_day.sort_values(by = "Day of Week")
viol_per_day["Day of Week"] = viol_per_day["Day of Week"].map(days2)
viol_per_day.set_index("Day of Week", drop = True, inplace = True)

# plot data
ax = viol_per_day.plot.bar(color = 'b', ylim=[59000, 73000])
ax.set_ylabel("Total Number of Violations")


# Now I want to visualize the geographical distribution of red light camera violations by projecting the number of violations onto a map of Chicago. To do this, I will need the location of each camera. This information wasn't in my first dataset, but let's see if it's in my second dataset. 

# In[ ]:


red_cam_loc_org.head()


# In[ ]:


#remove unneeded columns
red_cam_loc = red_cam_loc_org[["INTERSECTION", "LATITUDE", "LONGITUDE"]].copy()
red_cam_loc.head()


# Luckily, the second dataset contains information about the Latitude and Longitude of each camera.  Unfortunately, the Camera IDs aren't included in this second dataset. But I noticed that information about the intersection is present in each dataset, so I can merge the datasets on this column and hopefully retrieve information about where the Camera IDs are located. To do this, I will first need to modify the "Intersection" column of the first dataset. 

# In[ ]:


#remove all caps
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.title()

#replace "and" with "-"
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace(" And ","-")

red_cam_viol.head()


# In[ ]:


#Before merging, I want to learn a bit more about my datasets. 

null_counts_viol = red_cam_viol.isnull().sum()
print(null_counts_viol)


# In[ ]:


red_cam_viol.info()


# In[ ]:


null_counts_loc = red_cam_loc.isnull().sum()
print(null_counts_loc)


# In[ ]:


red_cam_loc.info()


# The first time I merged the two datasets, 241,983 rows contained no "Latitude" and Longitude" data and the Camera ID column went from 552 null values to 630. Because I merged on the "Intersections" columns, I wanted to further explore the data in those columns to see if I can figure out why some data disappeared. My assumption is that the Intersections will be the same for both datasets, but that may not be true.

# In[ ]:


#create unique lists of intersections within each dataset

loca = np.sort(red_cam_loc['INTERSECTION'].unique())
viol = np.sort(red_cam_viol['INTERSECTION'].unique())


# In[ ]:


#find values that do not appear in viol lists
def missing(loca, viol): 
    return (list(set(loca) - set(viol)))

missing = missing(loca, viol)
missing.sort()
print(missing)


# In[ ]:


# find missing values that do not appear in loc list

def missing2(viol, loca): 
    return (list(set(viol) - set(loca)))

missing2 = missing2(viol, loca)
missing2.sort()
print(missing2)


# I notice right away that there are some problems with capitalization - for example, the intersections in list viol are written with "st" or "th", while the intersections in list loc are written with "St" and "Th". To avoid these kinds of capitalization errors, I will make the columns in each dataframe uppercase.

# In[ ]:


red_cam_loc['INTERSECTION'] = red_cam_loc['INTERSECTION'].str.upper()
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.upper()


# I gained a bit more Latitude and Longitude data (from 241,983 missing to 230,165), but there is still more munging to do! Two other problems I noticed: in the loc list, sometimes "?" appear instead of spaces (like in the street "Stony Island", which is also misspelled in the viol list). A second problem is the order of the Intersection streets. In the viol list, one intersection is listed as "California-Irving Park" but in the loc list, the same intersection appears as "Irving Park-California." 

# In[ ]:


# replace errors found in spot check of lists

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("?", " ")
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("STONEY", "STONY")
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("31ST ST-MARTIN LUTHER KING DRIVE", "DR MARTIN LUTHER KING DRIVE-31ST")
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("4700 WESTERN", "47TH-WESTERN")


# In[ ]:


# resave first list with capitalized letters
loca = np.sort(red_cam_loc['INTERSECTION'].unique())


# In[ ]:


# create new column
red_cam_viol["Corrected Intersection"] = 'Unchecked'

# divides intersections by hyphen and insert in new column
red_cam_viol["First Street"], red_cam_viol["Second Street"] = red_cam_viol.INTERSECTION.str.split("-", 1).str
red_cam_viol['New Intersection'] = red_cam_viol["Second Street"] + "-" + red_cam_viol["First Street"]

def match(df, loca): 
    df.loc[df['INTERSECTION'].isin(loca), "Corrected Intersection"] = df['INTERSECTION']
    df.loc[(~df['INTERSECTION'].isin(loca)) & (df["New Intersection"].isin(loca)), "Corrected Intersection"] = df['New Intersection'] 
    errors = df.loc[df['Corrected Intersection'] == 'Unchecked']
    errors_list = np.sort(errors['INTERSECTION'].unique())
    return df, errors_list

# call function 
red_cam_viol, errors_list = match(red_cam_viol, loca)
print(errors_list)


# My function hasn't removed all the errors, so I will have to munge manually. But it's much easier than before to spot these smaller discrepencies! For example, I now notice that a few intersections list three streets instead of two (like "Stony Island/Cornell-67th"), but uses a "/" instead of "-" to divide the streets. 

# In[ ]:


red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("/", "-")
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("HIGHWAY", "HWY")
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("83RD-STONY ISLAND", "STONY ISLAND-83RD")
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("95TH-STONY ISLAND", "STONY ISLAND-95TH")
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("ARCHER-NARRAGANSETT-55TH", "ARCHER-NARRAGANSETT")
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("LAKE-UPPER WACKER", "LAKE-WACKER")
red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("DR MARTIN LUTHER KING-31ST", "DR MARTIN LUTHER KING DRIVE-31ST")
red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("LAKE SHORE-BELMONT", "LAKE SHORE DR-BELMONT")
red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("PULASKI-ARCHER-50TH", "PULASKI-ARCHER")
red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("KOSTNER-GRAND-NORTH", "KOSTNER-GRAND")
red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("HOMAN-KIMBALL-NORTH", "HOMAN-KIMBALL")
red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("WESTERN-DIVERSEY-ELSTON", "WESTERN-DIVERSEY")
red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("KEDZIE-79TH-COLUMBUS", "KEDZIE-79TH")
red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("HALSTED-FULLERTON-LINCOLN", "HALSTED-FULLERTON")


# In[ ]:


#resave list with corrected intersection names
loca = np.sort(red_cam_loc['INTERSECTION'].unique())

#rewrite columns with corrected values
red_cam_viol["First Street"], red_cam_viol["Second Street"] = red_cam_viol.INTERSECTION.str.split("-", 1).str
red_cam_viol['New Intersection'] = red_cam_viol["Second Street"] + "-" + red_cam_viol["First Street"]

# call function again 
red_cam_viol, errors_list = match(red_cam_viol, loca)
print(errors_list)


# In[ ]:


red_cam_viol.info()


# It doesn't look like there's much else left to munge. I am a little confused how these street intersections reported red camera violations if there is no camera located there, but for now I will move on with my merge and mapping EDA. 

# In[ ]:


red_cam_final_org = pd.merge(red_cam_viol, red_cam_loc, left_on = "Corrected Intersection",right_on= "INTERSECTION", how = "left")
red_cam_final_org.head()


# In[ ]:


null_counts_merged = red_cam_final_org.isnull().sum()
print(null_counts_merged)


# In[ ]:


red_cam_final_org.info()


# I now only have 45,637 rows without Latitude and Longitude data, as opposed to over 240,000 before munging. Next, I'm going to clean up my final dataset a little.

# In[ ]:


red_cam_final_org["INTERSECTION_y"] = np.where(red_cam_final_org["INTERSECTION_y"].isnull(), red_cam_final_org["INTERSECTION_x"], red_cam_final_org["INTERSECTION_x"])


# In[ ]:


red_cam_final = red_cam_final_org[["INTERSECTION_y", "CAMERA ID", "LATITUDE", "LONGITUDE", "VIOLATION DATE", "VIOLATIONS"]].copy()
red_cam_final.rename(columns = {"INTERSECTION_y" : "INTERSECTION"}, inplace = True)
red_cam_final.head()


# For my mapping, I've decided to concentrate on the total number of violations per intersection. This means I will have to manipulate my red_cam_final dataframe to create a new dataframe, which I can then re-merge with the original dataframe to retrieve the location data. I used ideas from: https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972 and https://towardsdatascience.com/exploring-and-visualizing-chicago-transit-data-using-pandas-and-bokeh-part-ii-intro-to-bokeh-5dca6c5ced10 for help plotting. 

# In[ ]:


intersection_grouped = red_cam_final.groupby("INTERSECTION")
intersection_summed = pd.DataFrame(intersection_grouped["VIOLATIONS"].sum())
intersection_summed.head()


# In[ ]:


red_cam_totals = pd.merge(intersection_summed,red_cam_final,left_on = "VIOLATIONS", right_index = True)
red_cam_totals = red_cam_totals[["VIOLATIONS", "LATITUDE", "LONGITUDE"]].copy()

# set 100 as scale factor
red_cam_totals["SCALE"] = red_cam_totals["VIOLATIONS"] / 100
red_cam_totals.head()


# Now I need to convert the Latitude and Longitude into x and y tuples. 

# In[ ]:


# import libraries
from shapely.geometry import Point, Polygon 
import matplotlib.pyplot as plt
import geopandas as gpd 
import descartes


# In[ ]:


#convert to a geo-dataframe
geometry = [Point(xy) for xy in zip(red_cam_totals['LONGITUDE'], red_cam_totals['LATITUDE'])]
crs = {'init','epsg:4326'}
gdf = gpd.GeoDataFrame(red_cam_totals, crs=crs, geometry=geometry)


# In[ ]:


#plot red light cameras onto city map
street_map = gpd.read_file('../input/chicago-streets-shapefiles/geo_export_75808441-05b9-4a51-a665-cf23dcf0a285.shx')
fig,ax = plt.subplots(figsize = (15,15))
street_map.plot(ax = ax, alpha = 0.4, color = "grey")
gdf.plot(ax=ax, markersize=red_cam_totals["SCALE"], marker="o", color="red")


# For an actual dashboard, my charts, graphs and maps would be more interactive, to allow other users to modify the data as they see fit. Other tips (from Kaggle's Dasbhoarding tutorial) include: 
# 
# * Hiding code, 
# * Removing text, and 
# * Consolidating charts and tables on the same line when possible

# In[ ]:


viol_per_day = viol_per_day.reset_index()
viol_per_day.head()


# In[ ]:


#install package 
get_ipython().system('pip install chart-studio')

#import plotly 
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()


# In[ ]:


data = [
    go.Scatter(x=viol_per_day['Day of Week'], y=viol_per_day['VIOLATIONS'], marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5,)),
    opacity=0.6)]

layout = go.Layout(autosize = True, title="Red Light Camera Violations per Day", xaxis={'title':'Days of Week'}, yaxis={'title':'Total Violations'})

fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


# reset index to convert dataframe from geo to plotly
gdf_2 = gdf.reset_index()
gdf_2['TEXT'] = gdf_2['INTERSECTION'] + ": " + gdf_2['VIOLATIONS'].astype(str) + ' total violations'


# In[ ]:


#Instead of using scaled data, I will normalize the "VIOLATIONS" column.

# normalize data using min-max 
gdf_2["NORMALIZED"] = (gdf_2["VIOLATIONS"] - gdf_2["VIOLATIONS"].min()) / (gdf_2["VIOLATIONS"].max() - gdf_2["VIOLATIONS"].min())


# In[ ]:


mapbox_access_token = 'pk.eyJ1IjoidGdhc2luc2tpIiwiYSI6ImNqcXc3MjhpNzEzMnYzeG9ieDNkb2M5ZmQifQ.m3MsgcBIXdwOT6hxvi007g'

data = [
    go.Scattermapbox(
        lat= gdf_2['LATITUDE'],
        lon= gdf_2['LONGITUDE'],
        mode='markers',
        text = gdf_2['TEXT'],
        hoverinfo = 'text',
        marker=dict(
            size= 8,
            color = gdf_2['NORMALIZED'],
            colorscale= 'Jet', 
            showscale=True,
            cmax=1,
            cmin=0),),]

layout = go.Layout(
    title = "Number of Total Red Light Violations by Intersection in Chicago", 
    autosize=True,  
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=41.881832,
            lon=-87.623177),
        pitch=0,
        zoom=10),)

fig = dict(data=data, layout=layout)
iplot(fig)


# # Final Dashboard

# In[ ]:


print("Total Number of Red Camera Violations in Chicago:")
red_cam_viol["VIOLATIONS"].sum()


# In[ ]:


mapbox_access_token = 'pk.eyJ1IjoidGdhc2luc2tpIiwiYSI6ImNqcXc3MjhpNzEzMnYzeG9ieDNkb2M5ZmQifQ.m3MsgcBIXdwOT6hxvi007g'

trace1 = go.Scatter(x=viol_per_day['Day of Week'], y=viol_per_day['VIOLATIONS'], mode='lines+markers+text', xaxis='x1', yaxis='y1')#, subplot = 'plot1')

trace2 = go.Scattermapbox(
        lat= gdf_2['LATITUDE'],
        lon= gdf_2['LONGITUDE'],
        mode='markers',
        text = gdf_2['TEXT'],
        hoverinfo = 'text',
        marker=dict( 
            size= 8,
            color = gdf_2['NORMALIZED'],
            colorscale= 'Jet',
            showscale=True,
            colorbar=dict(title = dict(text="Violations (Scaled)", side="right"), x = 1), 
            cmax=1,
            cmin=0), 
        subplot = 'mapbox')

data = [trace1, trace2]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 1]
    ),
    mapbox=dict(
        accesstoken=mapbox_access_token,
        domain = {'x' : [.5, 1], 'y' : [0,1]},
        bearing=0,
        center=dict(
            lat=41.8746,
            lon=-87.6687),
        pitch=0,
        zoom=10),)

fig = go.Figure(data=data, layout=layout)

fig['layout']['xaxis1'].update(title='Days per Week')
fig['layout']['yaxis1'].update(title='Total Number of Violations')
fig['layout'].update(showlegend=False)
fig['layout'].update(height=600, width=800, title='Total Red Light Camera Violations in Chicago: By Day and Intersection')

iplot(fig) 

