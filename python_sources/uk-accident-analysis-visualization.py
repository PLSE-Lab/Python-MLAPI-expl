#!/usr/bin/env python
# coding: utf-8

# ## UK Accident Analysis 2005 - 2014

# 
# - This project gives detailed insights into **United Kingdom (UK) long-term road accident trends between 2005 - 2014,** which includes but not limited to potential casualties due to road accidents, areas most affected by accidents, highway authorities notorious for road accidents and those that are the "safest", conditions likely to cause accidents and road types and geographical regions well known for acidents.
# 
# 
# 
# - The aforementioned insights led us to ask questions that were answered through interactive data visualization **(with the help of Plotly).**
# 
# 
# 
# ### Questions:
# 
# 
# 1. What is the trend of road accidents locations in UK from 2005 - 2014?
# 
# 
# 
# 2. 
# 
#    a.  What is the rate of road accidents (i.e. the number of casualties) in UK between 2005 - 2014?
#    
#    b.  What is the rate of the road accidents based on different UK Regions, has it all been the same in every region since 2005?
# 
# 
# 
# 3. Which ***highway authorities are the most dangerous or safest*** in UK based on accident records between 2005 - 2014?  ***We will look at only the top 20 of them.***
# 
# 
# 
# 4. What are the accident occurence rate in UK ***based on time of the day, weekdays, and months of the year,*** between 2005 - 2014?
# 
# 
# 
# 5. 
# 
#    a. Which particular road network group (based on network density range) is the most dangerous (i.e. has high numbers of  casualties) between 2005 - 2014?
#    
#    b. Which road type has the highest rate of road accidents between 2005 - 2014?
# 
# 
# 6. What condition **(taking all conditions into account i.e. road, weather and light conditions)** caused the most road accidents in UK between 2005 - 2014?
# 
# 
# 
# 7. Is pedestrian crossing a cause of road accident, or does it influence the road casualties in UK?
# 
# 
# 
# 8. 
# 
#      a. How many numbers of casualties occur per accidents in UK and what is their distribution in terms of the total amount of road accident that occured between 2005 - 2014?
# 
#      b. How many vehicles are invloved in each road accident, and what is their distribution in terms of the total amount of road accident that occured between 2005 - 2014?
# 
# 
# 
# 
# 9. Which speed limit is closely associated with road accidents in UK, from 2005 - 2014?
# 
# 
# 
# 10. 
# 
#     a. Which areas (urban / rural) in UK were road accidents the most frequent in, from 2005 - 2014?
# 
#     b. How is the distribution of accident severity in UK, from 2005 - 2014?
# 

# In[ ]:





# In[ ]:


import math 
import calendar
import pandas as pd
import datetime


import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.express as px


# In[ ]:





# In[ ]:


#load in datasets
uk_2007 = pd.read_csv('../input/uk-acc-2005-2007/accidents_2005_to_2007.csv', low_memory=False)
uk_2011 = pd.read_csv('../input/uk-accident-datasets/accidents_2009_to_2011.csv', low_memory=False)
uk_2014 = pd.read_csv('../input/uk-accident-datasets/accidents_2012_to_2014.csv', low_memory=False)


# In[ ]:


uk_accidents = pd.concat([uk_2007, uk_2011, uk_2014], ignore_index=True)


# In[ ]:


uk_accidents.shape


# In[ ]:


#Number of nulls by each columns
uk_accidents.isnull().sum()


# In[ ]:


#drop columns with greater than 10,000 null values
uk_accidents = uk_accidents.drop(['Junction_Detail', 'Junction_Control', 'LSOA_of_Accident_Location'], axis=1)


# In[ ]:


uk_accidents.dropna(inplace = True)


# In[ ]:


uk_accidents.shape


# In[ ]:


uk_accidents.head(5)


# In[ ]:


#load additional datasets
uk_LAD = pd.read_csv('../input/uk-accident-datasets/LAD.csv')
uk_LAD3 = pd.read_csv('../input/uk-accident-datasets/LAD3.csv')
uk_LAD4 = pd.read_csv('../input/uk-accident-datasets/LAD4.csv')


# In[ ]:


uk_LAD.head()


# In[ ]:


uk_LAD3.head(5)


# In[ ]:


#Road Network Classification
uk_LAD4.head(10)


# In[ ]:


#lets merge the two dataframe based on the common column that both dataframe have which is 'Sub-Group'

uk_LAD2 = pd.merge(uk_LAD3, uk_LAD4, on='Sub-Group')


# In[ ]:


#columns with more than 100,000 None Values
none_columns = []
for x in uk_accidents.columns:
    none_ct = uk_accidents[x].loc[uk_accidents[x] == 'None'].count()
    if none_ct >= 100000:
        none_columns.append(x)


# In[ ]:


print(none_columns)


# In[ ]:


uk_accidents = uk_accidents.drop(none_columns, axis=1)


# In[ ]:


uk_accidents.shape


# In[ ]:


#rename ONS to LAD_Code
uk_accidents.rename(columns={'Local_Authority_(Highway)': 'LAD_Code'}, inplace = True)
uk_LAD2.rename(columns={'ONS Code': 'LAD_Code'}, inplace = True)
uk_LAD.rename(columns={'Code': 'LAD_Code'}, inplace = True)


# In[ ]:


uk_LAD2 = pd.merge(uk_LAD, uk_LAD2, on='LAD_Code')


# In[ ]:


uk_accidents = pd.merge(uk_accidents, uk_LAD2, on='LAD_Code')


# In[ ]:


df_uk_gpd = uk_accidents.copy()


# In[ ]:


#due to the large size of the accident dataset (~1.5 million rows)
#i will only convert the accident df for year 2014 to "GeoDataFrame"
df_uk_2014 = df_uk_gpd[df_uk_gpd['Year'] == 2014]


# In[ ]:


df_uk_gpd_2014 = df_uk_2014.copy()


# In[ ]:


#convert accident df
points = df_uk_gpd_2014.apply(lambda row: Point(row.Location_Easting_OSGR, row.Location_Northing_OSGR), axis=1)
df_uk_gpd_2014 = gpd.GeoDataFrame(df_uk_gpd_2014, geometry=points)


# In[ ]:


#uk 2018 road network map
gb_shape = gpd.read_file('../input/shape-file/2018-MRDB-minimal.shp')


# In[ ]:


gb_shape.shape


# ### Visualize locations of road accidents based on different categories: Regions in UK, Settlement Areas and Road Networks groups

# In[ ]:


#lets plot points where road accidents occured based on various columns/aspect of the data
def map_plot(df1, df2, column, column_title, color):
    
    ax = df1.plot(figsize=(30,15), color='black', linewidth=0.6)
    df2.plot(column= column, ax=ax, markersize=60, legend = True, cmap=color, edgecolor='white')
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((1.1,0.5))
    leg.set_title(column_title)
    leg.set_frame_on(False)
    ax.set_axis_off()
    ax.set_title('Locations of Road Accidents in UK based on the ' + column_title + ', 2014', fontsize=16, pad=10)


# In[ ]:


map_plot(gb_shape, df_uk_gpd_2014, 'Region/Country', 'Regions', "gist_rainbow_r")


# In[ ]:


map_plot(gb_shape, df_uk_gpd_2014, 'Supergroup name', 'Settlement Areas', "plasma_r")


# In[ ]:


map_plot(gb_shape, df_uk_gpd_2014, 'Sub-Group Description', "Road Networks", "rainbow")


# ### 1. What is the trend of road accidents locations in UK from 2005 - 2014?

# In[ ]:


sns.set_context('talk')
sns.set_style("darkgrid")
g = sns.FacetGrid(df_uk_gpd, col="Year", hue="Region/Country", palette="gist_rainbow_r", col_wrap=3, height=6)
g.map(plt.scatter, "Longitude", "Latitude")
g.add_legend()
g.fig.subplots_adjust(top=0.93)
plt.suptitle('Yearwise trend of road accidents in UK between 2005 - 2014 for different regions', fontsize=24)
plt.show()


# ***Observation:*** As seen from the above plots, the rate of accidents in terms of geographical distribution, as pretty much remained the same i.e. locations where accidents occured in the past is likely to be the same location where accidents will occur tomorrow/next year. 
# 
# Due to the high accident rates in UK between 2005 - 2014, the yearwise accident trends will be visualized better either with  line plots or bar charts, this is done in the next sctions.

# In[ ]:


#convert all datetime columns to datetime formats
df_uk = uk_accidents.copy()
df_uk['Date'] = pd.to_datetime(df_uk['Date'])
df_uk['Year'] = df_uk['Date'].dt.year


# In[ ]:


df_uk['Day'] = df_uk.Date.dt.day
df_uk['week_in_month'] = pd.to_numeric(df_uk.Day/7)
df_uk['week_in_month'] = df_uk['week_in_month'].apply(lambda x: math.ceil(x))
df_uk['month'] = df_uk.Date.dt.month


# In[ ]:


#datetime.time(df_uk['Time'])
df_uk['Time'] = pd.to_timedelta(df_uk['Time'] +':00')


# In[ ]:


df_uk['Time']


# In[ ]:


df_uk['Hour'] = df_uk['Time'].dt.components['hours']


# In[ ]:


df_uk.Hour.unique()


# ### 2a. What is the rate of road accidents (i.e. the number of casualties) in UK between 2005 - 2014?
# 
# ### 2b. What is the rate of the road accidents based on different UK Regions, has it all been the same in every region since 2005?

# In[ ]:


def groupby_accidents(df, column):
    col_agg = df.groupby(column).Number_of_Casualties.agg(['sum', 'count', 'mean'])
    col_agg.reset_index(inplace = True)
    col_agg.sort_values(by = column, inplace = True)
    return col_agg


# In[ ]:


year_agg = groupby_accidents(df_uk, 'Year')


# In[ ]:


data = go.Scatter(x = year_agg['Year'], y=year_agg['sum'], mode="lines+markers", name='Number of Casualties', 
                  line= dict(color = ('rgb(255,165,0)'), width=4), showlegend = True)
layout = go.Layout(title='<b> Records of Road Accidents (Casualties per year) in UK between 2005 - 2014 <b>',
                   xaxis=dict(title='<b> Years <b>',titlefont=dict(size=16, color='#7f7f7f')),
                   yaxis=dict(title='<b> Number of Casualties <b>',titlefont=dict(size=16,color='#7f7f7f'))
                 )
fig = go.Figure(data=data, layout = layout)
fig.update_xaxes(dtick=1)

iplot(fig)


# **Observations**
# 
# - From the above plot, the rates of road accidents are declining, but in 2012 there was a huge spike, following this year road accidents continue to decline but year 2014 seems not to be declining relative to year 2013.

# In[ ]:


#records of road accidents based on different regions
def plot_regions_agg(df, regions, content):
    
    def groupby_region(df, column, name):
        
        col_agg = df[df['Region/Country'] == name].groupby(column).Number_of_Casualties.agg(['count', 'sum'])
        col_agg.reset_index(inplace = True)
        col_agg.sort_values(by = column, inplace = True)
        return col_agg
    
    region_traces = []
    colors = ['lightslategray', 'crimson', 'darkcyan', 'darkgoldenrod', 'cornsilk', 'turquoise', 'limegreen',               'darkorchid', 'palevioletred', 'forestgreen', 'silver', 'lightsteelblue']
    for name in range(len(regions)):
        name_agg = groupby_region(df, 'Year', regions[name])
        data_agg = go.Scatter(x = name_agg['Year'], y= name_agg['sum'], mode="lines+markers", name=regions[name], 
                              line= dict(color = colors[name], width=2.5))
        region_traces.append(data_agg)
    layout = go.Layout(title='<b> Rates of Road Accidents in different ' + content + ' regions between 2005 - 2014<b>', width=1100, 
                       height=600, xaxis=dict(title='<b> Year <b>',titlefont=dict(size=16, color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')), 
                       yaxis=dict(title='<b> Number of Casualties <b>',titlefont=dict(size=16,color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')))
        
    fig = go.Figure(data=region_traces, layout = layout)
    fig.update_xaxes(dtick=1)
    fig.show()


# In[ ]:


regions = df_uk['Region/Country'].unique().tolist()
plot_regions_agg(df_uk, regions, 'UK')


# ***Lets check regions where there were spikes in accident rates in year 2012***

# In[ ]:


areas = ['London, England', 'North East, England', 'Yorkshire and The Humber, England', 'North West, England']
plot_regions_agg(df_uk, areas, 'England')


# **Observations**
# 
# - Regions in UK ***except London, North East, Yorkshire and the Humber and North West*** have road accidents declining between 2005 - 2014.
# 

# In[ ]:


region_agg = groupby_accidents(df_uk, 'Region/Country').sort_values(by = 'sum', ascending = False)
fig = px.bar(region_agg, x= 'Region/Country', y= 'sum', color='sum', 
             labels={'sum':'<b> Number of Casualties <b>'}, width=1000, height=700,
             color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(title='<b> Road Accidents in UK (based on different regions) between 2005 - 2014 <b>', xaxis_title='<b> Year <b>',
                  yaxis_title='<b> Number of Casualties <b>')
fig.show()


# **Observations**
# 
# - South East, England have the highest occurence of road accident while North East, England have the least occurence compared to other regions.

# ### 3a. Which ***highway authorities are the most dangerous or safest*** in UK based on accident records between 2005 - 2014?
# 
# ***We will look at only 20 of them***

# In[ ]:


top_20 = groupby_accidents(df_uk, 'Highway Authority').sort_values(by = 'sum', ascending = False)[:20]
bottom_20 = groupby_accidents(df_uk, 'Highway Authority').sort_values(by = 'sum', ascending = False)[-20:]


# In[ ]:


#Data Visualization
def plot_highway(df, name, color):
    fig = px.bar(df, x= 'Highway Authority', y= 'sum', color='sum', 
                 labels={'sum':'<b> Casualties <b>'}, width=1000, height=700,
                 color_continuous_scale=color)
    
    if name == 'Safest':
        content = 'lowest'
    else:
        content = 'highest'
    
    fig.update_layout(title='<b> 20 ' + name + ' Highway Authorities with the ' + content + ' record of road accidents in UK <b>', xaxis_title='<b> Highway Authority <b>', 
                      yaxis_title='<b> Number of Casualties <b>')
    fig.show()


# In[ ]:


plot_highway(top_20, 'Dangerous', px.colors.sequential.Cividis)


# In[ ]:


plot_highway(bottom_20, 'Safest', px.colors.sequential.Inferno)


# **Observations**
# 
# - Kent County is said to be the **'most notorious highway'** for road accidents while East Ayrshire is siad to be the **'safest'** among other highway authorities between 2005 - 2014.

# ### 4. What are the accident occurence rate in UK based on time of the day, weekdays, and months of the year, between 2005 - 2014?

# In[ ]:


def plot_time_agg(df, column):
    
    def groupby_col(df, column, year):
        
        col_agg = df[df['Year'] == year].groupby(column).Number_of_Casualties.agg(['count', 'sum'])
        if year == None:
            col_agg = df.groupby(column).Number_of_Casualties.agg(['count', 'sum'])
            col_agg['average'] = col_agg['sum'] / 9
        col_agg.reset_index(inplace = True)
        col_agg.sort_values(by = column, inplace = True)
        if column == 'month':
            col_agg['month'] = col_agg['month'].apply(lambda x: calendar.month_abbr[x])
        if column == 'Day_of_Week':
            col_agg['Day_of_Week'] = col_agg['Day_of_Week'] - 1
            col_agg['Day_of_Week'] = col_agg['Day_of_Week'].apply(lambda x: calendar.day_abbr[x])
            
        return col_agg
    
    year_list = df_uk['Year'].unique().tolist() + [None]
    list_of_traces = []
    colors = ['darkmagenta', 'deeppink', 'lavender', 'lightsteelblue', 'orchid', 'navy', 'forestgreen',               'greenyellow', 'silver', 'darkslategrey']
    
    for year in range(len(year_list)):
        if year_list[year] == None:
            name_agg = groupby_col(df, column, year_list[year])
            data_agg = go.Scatter(x = name_agg[column], y= name_agg['average'], mode="lines+markers", name='Overall Average', 
                                 line= dict(color = 'darkslategrey', width=4))
        else:
            name_agg = groupby_col(df, column, year_list[year])
            data_agg = go.Scatter(x = name_agg[column], y= name_agg['sum'], mode="lines+markers", name=regions[year], 
                                  line= dict(color = colors[year], width=2, dash = 'dashdot'))
        list_of_traces.append(data_agg)
        
        
    #Data visualization
    if column == 'month':
        content = 'by Months of the year'
        tk = ''
    elif column == 'Hour':
        content = 'during Day and Night'
        tk = 1
    elif column == 'Day_of_Week':
        content = 'by weekdays'
        tk = ''
    elif column == 'week_in_month':
        content = 'by weeks in a month'
        tk = ''
    
    layout = go.Layout(title='<b> Occurences of Road Accidents ' + content + ' in UK between 2005 - 2014<b>', 
                       xaxis=dict(title='<b> ' + column + ' <b>',titlefont=dict(size=16, color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')), 
                       yaxis=dict(title='<b> Number of Casualties <b>',titlefont=dict(size=16,color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')))
    
    fig = go.Figure(data=list_of_traces, layout = layout)
    fig.update_xaxes(dtick=tk)
    iplot(fig)


# In[ ]:


plot_time_agg(df_uk, 'Day_of_Week')


# In[ ]:


plot_time_agg(df_uk, 'Hour')


# In[ ]:


plot_time_agg(df_uk, 'week_in_month')


# In[ ]:


plot_time_agg(df_uk, 'month')


# **Observations**
# 
# - Based on the week days, **Saturday** has the most occurence of road accidents in UK while **Mondays** has the least occurence of road accidents in UK, 2005 - 2014.
# 
# - Based on the hours of the day, **8am and 5pm** have the highest occurence of road accidents, while **12am - 4am** have the least occurence of accident, with 4am having the lowest occurence.
# 
# - Based on the weeks in a month, the first 4 weeks in a month are relatively the same in regards to the rate of accidents, while the 5th week which is mostly not frequent has the least occurence.
# 
# - Based on the months of the year, **February** has the least records of accident while **October and November** has the most records of accident. Nevertheless, **May - November** has relatively the same average records of accident in a year, between 2005 - 2014.

# ### 5a. Which particular road network group (based on network density range) is the most dangerous (i.e. has high numbers of casualties) between 2005 - 2014?
# 
# ### 5b. Which road type has the highest rate of road accidents between 2005 - 2014?

# In[ ]:


network_agg = groupby_accidents(df_uk, 'Sub-Group Description').sort_values(by = 'sum', ascending = False)


# In[ ]:


network_agg


# In[ ]:


#Data Visualization

def visualize_aggregates(df, column, color):
    
    fig = px.bar(df, x= column, y= 'sum', color='sum', 
                 labels={'sum':'<b> Casualties <b>'}, width=1000, height=700, 
                 color_continuous_scale=color)
    
    if column == 'Sub-Group Description':
        content = 'Road Networks in UK with the highest record of road accidents, 2005 - 2014 '
    elif column == 'Road_Type':
        content = 'Road Types in UK with the highest record of road accidents, 2005 - 2014'
    elif column == 'Conditions':
        content = 'Weather, Road & Light Conditions at the time of Road accidents in UK between 2005 - 2014'
    elif column == 'Pedestrian_Crossing':
        content = 'Pedestrian Crossings at the locations of Road accidents in UK between 2005 - 2014'
    elif column == 'Speed_limit':
        content = 'Speed Limits associated with Road accidents in UK between 2005 - 2014 '
    
    fig.update_layout(title='<b> ' + content + ' <b>', 
                      xaxis=dict(title='<b> ' + column + ' <b>',titlefont=dict(size=16), tickfont=dict(size=13, color='darkslateblue')), 
                      yaxis=dict(title='<b> Number of Casualties <b>',titlefont=dict(size=16), tickfont=dict(size=13, color='darkslateblue'))) 
    
    fig.show()


# In[ ]:


visualize_aggregates(network_agg, 'Sub-Group Description', px.colors.diverging.Spectral)


# In[ ]:


road_agg = groupby_accidents(df_uk, 'Road_Type').sort_values(by = 'sum', ascending = False)


# In[ ]:


visualize_aggregates(road_agg, 'Road_Type', px.colors.diverging.RdBu)


# **Observations**
# 
# - **Sparsely network rural authorities** road network has the highest occurence of road accidents in UK while **very sparsely network rural scottish island** road network has the least accurence.
# 
# - **Single carriageway** records the highest occurence of accident among all other road types.

# ### 6. What condition (taking all conditions into account i.e. road, weather and light conditions) caused the most road accidents in UK between 2005 - 2014?

# In[ ]:


#lets look at the distinct condition underwhich road accidents occured

conditions_columns = ['Road_Surface_Conditions', 'Weather_Conditions', 'Light_Conditions']
print('Conditions under which road accidents occured: \n')
for column in conditions_columns:
    print(column + ': ')
    print(df_uk[column].unique().tolist())
    print('\n')


# In[ ]:


roadsurface_agg = groupby_accidents(df_uk, 'Road_Surface_Conditions')
weather_agg = groupby_accidents(df_uk, 'Weather_Conditions')
light_agg = groupby_accidents(df_uk, 'Light_Conditions')


# In[ ]:


#convert 'other' value in weather_agg df to 'Unknown'
weather_agg['Weather_Conditions'] = weather_agg['Weather_Conditions'].replace({'Other': "Unknown"})


# In[ ]:


#rename all the conditions to 'Conditions'
roadsurface_agg.rename(columns={'Road_Surface_Conditions': 'Conditions'}, inplace = True)
weather_agg.rename(columns={'Weather_Conditions': 'Conditions'}, inplace = True)
light_agg.rename(columns={'Light_Conditions': 'Conditions'}, inplace = True)


# In[ ]:


condition_agg = pd.concat([roadsurface_agg, weather_agg, light_agg], ignore_index=True).sort_values(by = 'sum', ascending = False)


# In[ ]:


condition_agg


# In[ ]:


visualize_aggregates(condition_agg, 'Conditions', px.colors.diverging.RdYlGn)


# - From above plot, Most road accidents happen under conditions that **do not affect the road, vehicle nor the driver of the vehicle**, this is simply because most road accidents are said to occur under **Fine weather condition without High winds, Daylight with street light present and under dry road condition**. 
# 
# - On the other hand, Most road accidents do not occur when there is flood, darkness or snow fall with high winds.

# ### 7. Is pedestrian crossing a cause of road accident, or does it influence the road casualties in UK?

# In[ ]:


#we have two pedestrian crossing columns, lets check them out
#lets look at the distinct condition underwhich road accidents occured

pedestrian_columns = ['Pedestrian_Crossing-Physical_Facilities', 'Pedestrian_Crossing-Human_Control']
print('Pedestrian crossing at the time of road accidents: \n')
for column in pedestrian_columns:
    print(column + ': ')
    print(df_uk[column].unique().tolist())
    print('\n')


# In[ ]:


pedcrs1_agg = groupby_accidents(df_uk, 'Pedestrian_Crossing-Physical_Facilities')
pedcrs2_agg = groupby_accidents(df_uk, 'Pedestrian_Crossing-Human_Control')


# In[ ]:


pedcrs1_agg.rename(columns={'Pedestrian_Crossing-Physical_Facilities': 'Pedestrian_Crossing'}, inplace = True)
pedcrs2_agg.rename(columns={'Pedestrian_Crossing-Human_Control': 'Pedestrian_Crossing'}, inplace = True)


# In[ ]:


pedestrian_agg = pd.concat([pedcrs1_agg, pedcrs2_agg], ignore_index=True).sort_values(by = 'sum', ascending = False)


# In[ ]:


pedestrian_agg


# In[ ]:


visualize_aggregates(pedestrian_agg, 'Pedestrian_Crossing', px.colors.diverging.Spectral)


# - From the above plot, road accidents commonly do not occur at pedestrian crossing and pedestrian crossing do not in any way affect the cause of road accidents in UK, between 2005 - 2014.

# ### 8a. How many numbers of casualties occur per accidents in UK and what is their distribution in terms of the total amount of road accident that occured between 2005 - 2014?
# 
# ### 8b. How many vehicles are invloved in each road accident, and what is their distribution in terms of the total amount of road accident that occured between 2005 - 2014?

# In[ ]:


#visualize the distribution
def visualize_distribution(df, column, color):
    
    fig = px.histogram(df, x=column, color_discrete_sequence = color, 
                   width = 1000, height = 700)
    
    if column == 'Number_of_Casualties':
        content = 'Frequency of Casualties due to Road Accidents in UK between 2005 - 2014'
    elif column == 'Number_of_Vehicles':
        content = 'Distribution of Vehicles involved in Road Accidents in UK between 2005 - 2014'
    
    fig.update_layout(title='<b>' + content + '<b>', 
                      xaxis=dict(range=[0, 20], title='<b> ' + column + ' <b>',titlefont=dict(size=16, color='#7f7f7f'), 
                                 tickfont=dict(size=15, color='darkslateblue')), 
                      yaxis=dict(title='<b> Frequency <b>',titlefont=dict(size=16,color='#7f7f7f'), 
                                 tickfont=dict(size=15, color='darkslateblue')))
    
    fig.update_xaxes(dtick=1)
    fig.show()


# In[ ]:


visualize_distribution(df_uk, 'Number_of_Casualties', px.colors.diverging.balance)


# In[ ]:


visualize_distribution(df_uk, 'Number_of_Vehicles', px.colors.diverging.RdYlBu)


# **Observations**
# 
# - Casualties caused by road accidents are mostly to a number of one, as the distribution are rightly skewed.
# 
# - This is similar to the number of vehicles involved in an accident, as the number of vehicles is likely to be two, as the distribution is rightly skewed.

# ### 9. Which speed limit is closely associated with road accidents in UK, from 2005 - 2014?

# In[ ]:


speed_agg = groupby_accidents(df_uk, 'Speed_limit').sort_values(by = 'sum', ascending = False)


# In[ ]:


visualize_aggregates(speed_agg, 'Speed_limit', px.colors.diverging.Portland)


# **Observations**
# 
# - Most road accidents significantly occur when the speed limits are 30 mph, ***this may mean that most drivers violate the speed limit.***

# ### 10a. Which areas (urban / rural) in UK were road accidents the most frequent in, from 2005 - 2014?
# 
# ### 10b. How is the distribution of accident severity in UK, from 2005 - 2014?

# In[ ]:


severity_agg = groupby_accidents(df_uk, 'Accident_Severity').sort_values(by = 'sum')


# In[ ]:


severity_agg["Accident_Severity"].replace({1: "Fatal", 2: "Serious", 3: "Slight"}, inplace=True)


# In[ ]:


area_agg = groupby_accidents(df_uk, 'Urban_or_Rural_Area').sort_values(by = 'sum')


# In[ ]:


area_agg["Urban_or_Rural_Area"].replace({1: "Urban", 2: "Rural", 3: "Unallocated"}, inplace=True)


# In[ ]:


### GET THE LABELS AND VALUES FOR THE PIE CHART ###
labels1 = severity_agg["Accident_Severity"].values.tolist()
labels2 = area_agg["Urban_or_Rural_Area"].values.tolist()
values_acc = severity_agg["sum"].values.tolist()
values_area = area_agg['sum'].values.tolist()


# In[ ]:


# Create subplots, using 'domain' type for pie charts

night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)', 'rgb(6, 4, 4)']
cafe_colors =  ['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)', 'rgb(35, 36, 21)']

specs = [[{'type':'domain'}, {'type':'domain'}]]
fig = make_subplots(rows=1, cols=2, specs=specs, subplot_titles=['<b> Percent of Road Accidents Severity <b>', 
                                                                 '<b> Percent of Areas affected by Road Accidents <b>'])

# Define pie charts
fig.add_trace(go.Pie(labels=labels1, values=values_acc, name='Accident Severity',
                     marker_colors= night_colors, textinfo='label+percent', insidetextorientation='tangential'), 1, 1)
fig.add_trace(go.Pie(labels=labels2, values=values_area, name='Urban or Rural Area',
                     marker_colors= cafe_colors, textinfo='label+percent', insidetextorientation='radial'), 1, 2)


fig = go.Figure(fig)
fig.show()


# **Observations**
# 
# - Occurence of road accidents in urban areas is higher than its occurence in rural areas.
# 
# - Most road accidents have a slight severity on the victim, only a minute amount of road accidents is fatal.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




