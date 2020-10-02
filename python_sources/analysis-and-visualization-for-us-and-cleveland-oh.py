#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Part 1 : Accidents in the United States

# In[ ]:


# Import data analysis libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')

from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go


# In[ ]:


# Install geospatial data analysis libraries

get_ipython().system('pip install geopandas')
  # Necessary packages that go with geopandas for the spatial analysis
get_ipython().system('apt install libspatialindex-dev')
get_ipython().system('pip install rtree')


# In[ ]:


import geopandas as gpd
import math

import folium
from folium import Marker
from folium.plugins import HeatMap, MarkerCluster


# ## What are the top 10 States in term of accidents

# In[ ]:


# Load the data.
df_accidents = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
df_accidents.head(5)


# In[ ]:


# A serie with state abbr as labels and numbers of accidents as values.
accidents_per_state = df_accidents['State'].value_counts().sort_values(ascending=False)
accidents_per_state.head()


# In[ ]:


# Dictionary of US states and territories copied from internet to map the States' abbreviations with to get the state full names.
states_terrirories = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


# In[ ]:


# Create the DataFrame accidents per state.
df_accidents_per_state = pd.DataFrame({'State': accidents_per_state.index.map(states_terrirories), 
                                       'Abbr': accidents_per_state.index, 'Number Of Accidents': accidents_per_state.values})
# Add the column Percentage to the DataFrame.
df_accidents_per_state['Percentage'] = df_accidents_per_state['Number Of Accidents'].                                        apply(lambda x: '{:05.2f}'.format(x/len(df_accidents['ID'])*100))


# In[ ]:


# Display the top 10 States in term of accidents.
rank = np.arange(1,11)
df_top10_states = df_accidents_per_state.head(10).set_index(rank)
df_top10_states


# In[ ]:


rate_5_states = df_top10_states.head(5)['Number Of Accidents'].sum()/len(df_accidents['ID'])*100
print('Comment: Almost 50 % ({:05.2f} %) of all the US accidents occur in the top 5 states.'.format(rate_5_states))


# ## What are the top 10 cities in term of accidents?

# In[ ]:


# We can't use value_counts() for city like we did in the case of states.
# Cities name are not unique over the United States but seems to be unique by state.
# Let's use groupby state and city then create the DataFrame accidents per city.
df_city = df_accidents[['ID','State','City']]


# In[ ]:


# The groupby DataFrame
df_city = df_city.groupby(['State','City']).count()
df_city


# In[ ]:


# The method index.get_level_values(level) allows to get a serie from a multi-index according to the given level.
# Create the DataFrame accidents per city
df_accidents_per_city = pd.DataFrame({'City': df_city.index.get_level_values(1),'State': df_city.index.get_level_values(0).map(states_terrirories),
                                       'Number Of Accidents': df_city['ID']}).reset_index(drop=True)
df_accidents_per_city['Percentage'] = df_accidents_per_city['Number Of Accidents'].                                        apply(lambda x: '{:05.2f}'.format(x/len(df_accidents['ID'])*100))


# In[ ]:


df_top10_cities = df_accidents_per_city.sort_values(by='Number Of Accidents',ascending=False).head(10).set_index(rank)
df_top10_cities


# In[ ]:


houston = df_top10_cities.iloc[0,2] # first row: index = 0 , 3rd column: column = 2
pennsylvania = df_top10_states.loc[7,'Number Of Accidents'] # index label = 7, column label= 'Number of Accidents'

print('Comment: The top 1 city, Houston, has more accidents ({:,}) than the entire Pennsylvania state ({:,}) which is ranked 7th in the top 10 states.'.format(houston, pennsylvania))


# ## Visualiztion of the top 10 States and Cities

# In[ ]:


# Visualization of the 10 states and cities where accidents happen the most using pandas bult-in visualization

df_state = df_top10_states.set_index('State')
df_city = df_top10_cities.set_index('City')

fig,ax = plt.subplots(1,2,figsize=(18,6))
color = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

df_state['Number Of Accidents'].plot.bar(ax=ax[0],color=color)
ax[0].set_title("Top 10 States",size=20)
ax[0].set_xlabel('States',size=18)
ax[0].set_ylabel('Number of Accidents',size=18)

# Add percentages
total = len(df_accidents)
for p in ax[0].patches:
    height = p.get_height()
    # ax[0].text(x,y,string,ha,fontsize). x,y is the coordinates of the percentage text on the ax[0].
    ax[0].text(p.get_x()+p.get_width()/2,
            height + 5000,
            '{:1.2f}%'.format(height/total*100),
            ha="center",
            fontsize=12) 

df_city['Number Of Accidents'].plot(kind='bar',ax=ax[1],color=color)
ax[1].set_title("Top 10 Cities",size=20)
ax[1].set_xlabel('Cities',size=18)
ax[1].set_ylabel('Number of Accidents',size=18)

# Add percentages
total = len(df_accidents)
for p in ax[1].patches:
    height = p.get_height()
    # ax[1].text(x,y,string,ha,fontsize). x,y is the coordinates of the percentage text on the ax[1].
    ax[1].text(p.get_x()+p.get_width()/2,
            height + 1000,
            '{:1.2f}%'.format(height/total*100),
            ha="center",
            fontsize=12) 


# In[ ]:


print('Comment: Although Texas is not the top 1 state in term of accidents, it has 3 cities (Houston, Austin and Dallas) in the top 5 cities.')


# > ## Interactive Visualization of Accidents Distribution over the entire United States

# In[ ]:


# Defining data variable
text = df_accidents_per_state.apply(lambda x: x['State'] + ': ' + str(x['Percentage']) + ' %',axis=1)
data = dict(type = 'choropleth',
           locations = df_accidents_per_state['Abbr'],
           locationmode = 'USA-states',
            reversescale = True,
           text = text,
           z = df_accidents_per_state['Number Of Accidents'], # the color vary according to z value.
           colorbar = {'title':'Number of Accidents'}
           )

# Defining layout variable
layout = dict(geo = dict(scope='usa', showlakes=False, lakecolor='rgb(85,173,240)'),
             title = 'Accidents from February 2016 to December 2019 Distribution in the United States.'
             )

# Displaying the interactive choropleth
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# ## How accidents are distributed in term of severity and their occurrence moment (Day or Night)

# In[ ]:


# Distribution of accidents according to their severity and when (Day/Night) they occur.
fig,ax = plt.subplots(1,2,figsize=(18,6))

# Pie chart, that shows the distribution of accidents by severity from 1 to 4.
sizes = df_accidents.groupby('Severity').size() # serie with severity as index and the number of row of each categorie as values.
sizes = sizes[[2,1,3,4]] # the serie is reordered so that the values are clearly visible in the pie.
labels = 'Severity 2','Severity 1', 'Severity 3', 'severity 4' 
explode = (0.1,0.1,0.1,0.1)
colors = ['red','purple','green','yellow']
ax[0].pie(sizes, explode=explode, labels= labels,colors=colors,autopct='%1.2f%%',shadow=True, startangle=0)
ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax[0].set_title('Accidents by Severity', size=20)

# Seaborn countplot of accident distributions.
sns.countplot(data=df_accidents,x='Severity', hue='Sunrise_Sunset',ax=ax[1])
ax[1].set_title('Accidents by Severity and Moment', size= 20)
ax[1].set_xlabel('Severity', size=18)
ax[1].set_ylabel('Number of Accidents', size=18)


# In[ ]:


print('Comment:The large majority of accidents has from the category 2 on a scale of 4 in term of severity \nand no matter its severity, an accident likely occurs during the day than at night.')


# # Part 2: Accidents in Cleveland Ohio

# ## How Cleveland Accidents are distributed per year, month, day and hour

# In[ ]:


# Let's do some type conversions and filters to get only  accidents that occurred in Cleveland Ohio.
from datetime import datetime

years = ['2016','2017','2018','2019']
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

df_accidents['Start_Time'] = pd.to_datetime(df_accidents['Start_Time'])
df_accidents['End_Time'] = pd.to_datetime(df_accidents['End_Time'])

df_accidents_cleveland = df_accidents[(df_accidents['State']=='OH') & (df_accidents['City']=='Cleveland')]
df_accidents_cleveland.head()


# In[ ]:


print('Comment: There are {:,} accidents that occurred in Cleveland Ohio between February 2016 and December 2019'.format(len(df_accidents_cleveland)))


# In[ ]:


# Let's add some new columns that will help us in hours, week days, months, and years analysis.

  # function to create dates on format 'Fri Mar 20th, 2020'
def myDate(d):
  if d.day in [1,21,31]:
    return d.strftime('%a %b %dst, %Y')
  elif d.day in [2,22]:
    return d.strftime('%a %b %dnd, %Y')
  elif d.day==3:
    return d.strftime('%a %b %drd, %Y')
  else:
    return d.strftime('%a %b %dth, %Y')

df_accidents_cleveland['Hour'] = df_accidents_cleveland['Start_Time'].apply(lambda t: t.hour)
df_accidents_cleveland['Weekday'] = df_accidents_cleveland['Start_Time'].apply(lambda t: t.dayofweek)
df_accidents_cleveland['Monthday'] = df_accidents_cleveland['Start_Time'].apply(lambda t: t.day)
df_accidents_cleveland['Month'] = df_accidents_cleveland['Start_Time'].apply(lambda t: t.month)
df_accidents_cleveland['Year'] = df_accidents_cleveland['Start_Time'].apply(lambda t: t.year)
df_accidents_cleveland['Date'] = df_accidents_cleveland['Start_Time'].apply(myDate)


# In[ ]:


df_accidents_cleveland.head()


# ### Accidents per Year

# In[ ]:


fig = plt.figure(figsize=(16,6))

ax = sns.countplot(data=df_accidents_cleveland,x='Year')
ax.set_title('Distribution of Cleveland Accidents per Year', size= 20)
ax.set_xlabel('Years', size=18)
ax.set_ylabel('Number of Accidents', size=18)

# Adding percentage for each year
total = len(df_accidents_cleveland)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
            height + 20,
            '{:1.0f}%'.format(height/total*100),
            ha="center",
            fontsize=14) 


# In[ ]:


print('Comment:The number of accidents has doubled from 2016 to 2017, then almost doubled from 2017 to 2018 before decreasing from 2018 to 2019.')


# ### Accidents per Month

# In[ ]:


fig = plt.figure(figsize=(16,6))

ax = sns.countplot(data=df_accidents_cleveland,x='Month',palette='bright')
ax.set_title('Distribution of Cleveland Accidents per Month', size= 20)
ax.set_xlabel('Months', size=18)
ax.set_xticklabels(months)
ax.set_ylabel('Number of Accidents', size=18)

# Adding percentage for each month
total = len(df_accidents_cleveland)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
            height + 5,
            '{:1.2f}%'.format(height/total*100),
            ha="center",
            fontsize=14) 


# In[ ]:


print('Comment: Most accidents happen in October, followed by August. July is the month with less accidents.')


# In[ ]:


# Matrix
matrix_month = pd.pivot_table(data=df_accidents_cleveland,values='ID',index=['Year'],columns='Month',aggfunc='count')
  # Replace the column with month number by the whole month name.
matrix_month.columns = months

matrix_month


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(20,6))

# Evolution of Accidents over Months each year.
  # when a serie is plotted, the values are the y and indexes are the x
ax[0].plot(matrix_month.loc[2016],color='m',label='2016',linewidth=3,
           linestyle='solid',marker='*',markersize=18, markerfacecolor='w',markeredgecolor='m',markeredgewidth='2')
ax[0].plot(matrix_month.loc[2017],color='g',label='2017',linewidth=3,
           linestyle='solid',marker='*',markersize=18, markerfacecolor='w',markeredgecolor='g',markeredgewidth='2')
ax[0].plot(matrix_month.loc[2018],color='r',label='2018',linewidth=3,
           linestyle='solid',marker='*',markersize=18, markerfacecolor='w',markeredgecolor='r',markeredgewidth='2')
ax[0].plot(matrix_month.loc[2019],color='b',label='2019',linewidth=3,
           linestyle='solid',marker='*',markersize=18, markerfacecolor='w',markeredgecolor='b',markeredgewidth='2')

ax[0].legend(loc=0)
ax[0].set_title('Evolution of Cleveland Accidents over Month each Year', size= 20)
ax[0].set_xlabel('Months', size=18)
ax[0].set_ylabel('Number of Accidents', size=18)

# Heatmap
sns.heatmap(matrix_month,cmap='coolwarm',linewidth=1,linecolor='black',ax=ax[1],xticklabels=months)
ax[1].set_title('Cleveland Accidents Month_Year Heatmap', size= 20)
ax[1].set_xlabel('Months', size=18)
ax[1].set_ylabel('Year', size=18)


# In[ ]:


print('Comment: October 2018 ({} accidents) and January 2019 ({} accidents) registered the most accidents\nwhile October 2016 ({} accidents) and July 2016 ({} accidents) registered the less'.format(int(matrix_month.loc[2018,'October']),
                                                                                              int(matrix_month.loc[2019,'January']),
                                                                                              int(matrix_month.loc[2016,'October']),
                                                                                              int(matrix_month.loc[2016,'July']))) 


# In[ ]:


# Evolution over months for the 4 years

  # a df is created for each year with the number of accidents per month and then those df are concatenated on years
df_evolution = pd.concat([matrix_month.loc[[2016]].rename(index={2016:'Number of Accidents'}),
                         matrix_month.loc[[2017]].rename(index={2017:'Number of Accidents'}),
                        matrix_month.loc[[2018]].rename(index={2018:'Number of Accidents'}),
                        matrix_month.loc[[2019]].rename(index={2019:'Number of Accidents'})],
                         keys=years,axis=1)
df_evolution.index.name = 'Accidents'

df_evolution


# In[ ]:


# Returns a list in the format J16, F16, ..., N19,D19 for January 2016, February 2016,...,November 2019, December 2019.
temp = []
for y in years:
  for m in months:
    temp.append(m[0]+y[2:])
    x_values= np.array(temp)

y_values = df_evolution.loc['Number of Accidents'].values # only the values of the serie that has a multi-level index.


# In[ ]:


plt.figure(figsize=(18,6))

plt.plot(y_values,color='b',linewidth=3,linestyle='solid',marker='*',markersize=18, markerfacecolor='w',markeredgecolor='r',markeredgewidth='2')
  # The 48 month names replace the value 0 - 48 which are the original ticks.
plt.xticks(ticks=np.arange(0,49),labels=x_values); # the semi colon (;) at the end allows to hide the description output.
plt.title('Evolution of Accidents per month from February 2016 to December 2019 in Cleveland',size=18)
plt.xlabel('Months', size=18)
plt.ylabel('Number of Accidents', size=18)


# In[ ]:


print('Comment: Since August 2017 when the number of accidents passed the threshold of 100 accidents per month, \nit has always been over 100 except for the month of November 2017')


# ### Accidents per Day

# In[ ]:


fig = plt.figure(figsize=(16,6))

ax = sns.countplot(data=df_accidents_cleveland,x='Weekday',palette='bright')
ax.set_title('Distribution of Cleveland Accidents per Day', size= 20)
ax.set_xlabel('Months', size=18)
ax.set_xticklabels(days)
ax.set_ylabel('Number of Accidents', size=18)

# Adding percentage for each day
total = len(df_accidents_cleveland)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
            height + 5,
            '{:1.0f}%'.format(height/total*100),
            ha="center",
            fontsize=14) 


# In[ ]:


print('Comment: Accidents occur less in weekend (Saturdays and sundays). They likely happen on Tuesdays.')


# In[ ]:


# Heatmap of accidents per day
matrix_day = pd.pivot_table(data=df_accidents_cleveland,values='ID',index=['Year'],columns='Weekday',aggfunc='count')
matrix_day.columns = days

matrix_day


# In[ ]:


# Distribution of accidents over years and days using Seaborn visualization.
fig,ax = plt.subplots(1,2,figsize=(20,6))

sns.countplot(data=df_accidents_cleveland,x='Weekday',hue='Year',ax=ax[0])
ax[0].set_title('Distribution of Cleveland Accidents per Day and Year', size= 20)
ax[0].set_xlabel('Days', size=18)
ax[0].set_ylabel('Number of Accidents', size=18)
ax[0].set_xticklabels(labels=days)

sns.heatmap(matrix_day,cmap='coolwarm',linewidth=1,linecolor='white',ax=ax[1])
ax[1].set_title('Cleveland Accidents Day_Year Heatmap', size= 20)
ax[1].set_xlabel('Days', size=18)
ax[1].set_ylabel('Years', size=18)


# In[ ]:


print('Comment: Except for the year 2016 when it was Thursday, Tuesday is the day when accidents likely happen.')


# In[ ]:


# A serie giving the number of accidents that happen each day.
accidents_per_day = df_accidents_cleveland['Date'].value_counts()
accidents_per_day.head(5)


# In[ ]:


# Create the DataFrame accidents per day.
df_accidents_per_day = pd.DataFrame({'Date': accidents_per_day.index,'Number of Accidents': accidents_per_day.values})

# Add the column Percentage to the DataFrame.
df_accidents_per_day['Percentage'] = df_accidents_per_day['Number of Accidents'].                                        apply(lambda x: '{:05.2f}'.format(x/len(accidents_per_day)*100))
                                        
# Display the top 10 Days that registered the most accidents.
rank = np.arange(1,11)
df_top10_days = df_accidents_per_day.head(10).set_index(rank)
df_top10_days


# In[ ]:


# Visualization of the 10 days when accidents happen the most using pandas bult-in visualization
fig = plt.figure(figsize=(18,6))
df_days = df_top10_days.set_index('Date')

color = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

ax = df_days['Number of Accidents'].plot.bar(color=color)
plt.title("Top 10 Days with the most Accidents",size=20)
plt.xlabel('Days',size=18)
plt.ylabel('Number of Accidents',size=18)

# Adding the number of accident for each day
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
            height + 0.2,
            '{}'.format(height),
            ha="center",
            fontsize=14) 


# In[ ]:


total_days = len(accidents_per_day)
total_accidents = accidents_per_day.values.sum()
daily_average = total_accidents/total_days

print('While the daily average for accidents is about {}, each day of the top ten days registers about 5 times the average.'.format(int(daily_average)))


# ### Accidents per Hour

# In[ ]:


fig = plt.figure(figsize=(18,6))

ax = sns.countplot(data=df_accidents_cleveland,x='Hour',palette='bright')
ax.set_title('Distribution of Cleveland Accidents per Hour', size= 20)
ax.set_xlabel('Hours', size=18)
ax.set_ylabel('Number of Accidents', size=18)

# adding percentage for each hour
total = len(df_accidents_cleveland)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
            height + 5,
            '{:1.0f}%'.format(height/total*100),
            ha="center",
            fontsize=14) 


# In[ ]:


print('Comment: The large majority of accidents happens around 7 and 8 am.')


# In[ ]:


# Using seaborn catplot to display two charts side by side like the facetgrid
g= sns.catplot(data=df_accidents_cleveland,x='Hour',col='Year',kind='count',height=4, aspect=1,palette='bright')
g.set_titles(size=18)
g.set_xlabels(size=18)
g.set_ylabels('Number of Accidents',size=18)


# In[ ]:


print('Except 2016, the 3 other years show the same trends: Accidents mostly happen in the morning and occur around 7 and 8 am.\nSince we already saw that accidents mostly occur on week days,we can conclude that accident happen during morning rush hours\n when people are on their way to work.')


# In[ ]:


# Heatmap of accidents per Hour
matrix_hour = pd.pivot_table(data=df_accidents_cleveland,values='ID',index=['Year'],columns='Hour',aggfunc='count')

matrix_hour


# In[ ]:


# Distribution of accidents per Hour using Seaborn visualization.
fig,ax = plt.subplots(1,2,figsize=(20,6))

sns.countplot(data=df_accidents_cleveland,x='Hour',hue='Year',ax=ax[0])
ax[0].set_title('Distribution of Cleveland Accidents per Hour and Year', size= 20)
ax[0].set_xlabel('Hours', size=18)
ax[0].set_ylabel('Number of Accidents', size=18)

sns.heatmap(matrix_hour,cmap='coolwarm',linewidth=1,linecolor='black',ax=ax[1])
ax[1].set_title('Cleveland Accidents Hour_Year Heatmap', size= 20)
ax[1].set_xlabel('Hours', size=18)
ax[1].set_ylabel('Years', size=18)


# In[ ]:


print('Comment: The heatmap shows clearly that the accidents happen mostly around 7 and 8 am for 2017, 2018 and 2019.')


# ### How accident features are correlated

# In[ ]:


# Let's convert the zip code feature into float by adding a new column
df_accidents['ZipcodeFloat'] = df_accidents['Zipcode'].apply(lambda z: z if pd.isna(z) else float(z[:5]))

matrix_corr = df_accidents[['Severity','Distance(mi)','ZipcodeFloat','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)',
                            'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)']].corr()

plt.figure(figsize=(18,6))
sns.heatmap(matrix_corr,cmap='coolwarm',linewidth=1,linecolor='black',annot=True,annot_kws={'size':14})
plt.title('Cleveland Accidents features Heatmap', size= 20)


# In[ ]:


print('Comment: From this heatmap, it can be said that there is some realationship between the severity of the accidednt and\nthe distance in mile representing the length of the road extent affected by the accident ')


# ## Spatial Analysis of Cleveland Accidents
# In this spatial Analysis section, we will plot Cleveland accidents, EMS, and Hospitals locations to analyze the spatial relationships that exist between those entities.

# ### Getting GeoDataFrames from DataFrames and Shapefiles

# #### Cleveland Boundary GeoDataFrame (Shapefile [source](https://opendata.arcgis.com/datasets/4441f1fc778a48748489a6534482c96e_0.zip))

# In[ ]:


# Use attribute queries to select only Cleveland boundary from the shapefile of all the Cuyahoga county municipalities (GCS = WGS84 that is epsg=4326)
gdf_municipalities = gpd.read_file('/kaggle/input/cleveland-accident-analysis/Municipalities_WGS84__Tiled/Municipalities_WGS84__Tiled.shp')
gdf_cleveland = gdf_municipalities[gdf_municipalities['MUNI_NAME']=='Cleveland'].reset_index(drop=True)

gdf_cleveland = gdf_cleveland[['OBJECTID', 'MUNI_NAME', 'CENSUS_ID', 'geometry']]
gdf_cleveland


# #### Cleveland Accidents GeodataFrame

# In[ ]:


df_spatatial = df_accidents_cleveland[['ID','Date', 'Severity', 'Start_Lat', 'Start_Lng', 'Street', 'City', 'County',
       'State', 'Zipcode']] 

# Create the GeoDataFrame and set its Geographic Coordinate System to WGS 84 (EPSG:4326)
gdf_accidents_cleveland = gpd.GeoDataFrame(df_spatatial,
                                           geometry=gpd.points_from_xy(df_spatatial['Start_Lng'],df_spatatial['Start_Lat']))
gdf_accidents_cleveland.crs = {'init': 'epsg:4326'}                                        

# Get all accidents that actually fall into Cleveland boundary using spatial join
gdf_accidents_cleveland_spatial = gpd.sjoin(gdf_accidents_cleveland,gdf_cleveland,op='within',lsuffix='ACC',rsuffix='CLE').reset_index(drop=True)
gdf_accidents_cleveland_spatial.head(3)


# In[ ]:


print('Comment: While the attribute query performed on all United States accidents gives {:,} accidents for Cleveland, the spatial join tells us that\n there are actually {:,} accidents that fall within cleveland municipality boundary.'.format(len(df_accidents_cleveland),len(gdf_accidents_cleveland_spatial)))


# #### Cleveland Emergency Medical Service (EMS) Points GeoDataFrame (Shapefile [source](https://opendata.arcgis.com/datasets/362c9480f12e4587b6a502f9ceedccde_0.zip?outSR=%7B%22latestWkid%22%3A3857%2C%22wkid%22%3A102100%7D))

# In[ ]:


# Use spatial join to select only cleveland EMS from the shapefile of all United States EMS (GCS = WGS84 that is epsg=4326)
gdf_ems = gpd.read_file('/kaggle/input/cleveland-accident-analysis/Emergency_Medical_Service_EMS_Stations/Emergency_Medical_Service_EMS_Stations.shp')
gdf_ems = gdf_ems[['OBJECTID','ADDRESS','CITY','STATE','ZIP','COUNTY','NAICSDESCR','TELEPHONE','DIRECTIONS','NAME','geometry']]

# Spatial join: it will return all EMS that fall within Cleveland boundary
gdf_ems_cleveland = gpd.sjoin(gdf_ems,gdf_cleveland,op='within',lsuffix='EMS',rsuffix='CLE').reset_index(drop=True)
gdf_ems_cleveland.head(3)


# In[ ]:


print('Comment: There are {} EMS locations in Cleveland'.format(len(gdf_ems_cleveland)))


# #### Cleveland Hospitals Points GeoDataFrame (Shapefile [source](https://opendata.arcgis.com/datasets/6ac5e325468c4cb9b905f1728d6fbf0f_0.zip?outSR=%7B%22latestWkid%22%3A3857%2C%22wkid%22%3A102100%7D))

# In[ ]:


# Use spatial join to select only cleveland Hospitals from the shapefile of all United States EMS (GCS = WGS84 that is epsg=4326)
gdf_hospitals = gpd.read_file('/kaggle/input/cleveland-accident-analysis/Hospitals/Hospitals.shp')
gdf_hospitals = gdf_hospitals[['OBJECTID','NAME','ADDRESS','CITY','STATE','ZIP','TYPE','TRAUMA','STATUS','NAICS_DESC','geometry']]
gdf_hospitals = gdf_hospitals[gdf_hospitals['STATUS']=='OPEN']

# Spatial join: it will return all EMS that fall within Cleveland boundary
gdf_hospitals_cleveland = gpd.sjoin(gdf_hospitals,gdf_cleveland,op='within',lsuffix='HOS',rsuffix='CLE').reset_index(drop=True)
gdf_hospitals_cleveland.head(3)


# In[ ]:


print('Comment: There are {} Hospitals locations in Cleveland'.format(len(gdf_hospitals_cleveland)))


# #### Plotting GeoDataFrames

# In[ ]:


ax = gdf_cleveland.plot(figsize=(20,6), color='none',edgecolor='black')
gdf_accidents_cleveland_spatial.plot(color='blue',label='Accidents',ax=ax)
gdf_ems_cleveland.plot(color='red',markersize=80,label='EMS',ax=ax)
gdf_hospitals_cleveland.plot(color='Yellow',markersize=80,label='Hospitals',ax=ax)
ax.set_title('Accidents, Hospitals and EMS in Cleveland',size=20)
ax.legend()


# In[ ]:


print("Comment: The South-West part of Cleveland that is the Cleveland Hopkins Airport area doesn't register any accident.")


# ### Interactive Visualizations

# #### Interactive Heatmap of Accidents in Cleveland

# In[ ]:


# Create a base map.
base_map1 = folium.Map(location=[41.49,-81.7059], tiles='cartodbpositron', zoom_start=11,width='90%',height='75%')

# Add a heatmap data (a list of Latitude and Longitude) created using list comprehension, to the base map
heat_data = [[row['Start_Lat'],row['Start_Lng']] for index, row in gdf_accidents_cleveland_spatial.iterrows()]
HeatMap(heat_data,radius=17,).add_to(base_map1)

# Display the map
base_map1


# In[ ]:


print('Comment: Downtown Cleveland area seems to be the area where accidents happen the most.')


# #### Interactive Map showing Markers for EMS and Hospitals on the Heatmap

# In[ ]:


base_map2 = folium.Map(location=[41.49,-81.7059], tiles='OpenStreetMap', zoom_start=11,width='90%',height='100%')

# Add a heatmap data (a list of Latitude and Longitude) created using list comprehension, to the base map
heat_data = [[row['Start_Lat'],row['Start_Lng']] for index, row in gdf_accidents_cleveland_spatial.iterrows()]
HeatMap(heat_data,radius=17,).add_to(base_map2)

# Adding EMS markers
mc = MarkerCluster()
for idx, row in gdf_ems_cleveland.iterrows():
  if not math.isnan(row['geometry'].y) and not math.isnan(row['geometry'].x):
    mc.add_child(Marker([row['geometry'].y, row['geometry'].x],tooltip=row['NAME'],icon= folium.Icon(color='red',icon='ambulance',prefix='fa')))
base_map2.add_child(mc)

# Adding Hospital Markers
mc = MarkerCluster()
for idx, row in gdf_hospitals_cleveland.iterrows():
  if not math.isnan(row['geometry'].y) and not math.isnan(row['geometry'].x):
    mc.add_child(Marker([row['geometry'].y, row['geometry'].x],tooltip=row['NAME'],icon= folium.Icon(color='blue',icon='h-square',prefix='fa')))
base_map2.add_child(mc)

# Display the map
base_map2


# ### Proximity Analysis
# The previous section has been done using the Geographic Coordinate Sytem WGS84 (EPSG:4326). However, for the proximity analysis that uses distance and area calculation, we will use a projected coordinate system that keeps areas and distances accuracy in cleveland area. We chose WGS 84 / UTM zone 17N (EPSG:32617).

# In[ ]:


# Project gdf into WGS 84 / UTM zone 17N (EPSG:32617)
gdf_cleveland = gdf_cleveland.to_crs(epsg=32617)
gdf_accidents_cleveland_spatial = gdf_accidents_cleveland_spatial.to_crs(epsg=32617)
gdf_ems_cleveland = gdf_ems_cleveland.to_crs(epsg=32617)
gdf_hospitals_cleveland = gdf_hospitals_cleveland.to_crs(epsg=32617)


# #### EMS Coverage within 5 Miles Radius

# In[ ]:


# The projected coordinate system map unit is meter. 5 miles = 8046.72 meters
ems_5m_coverage = gpd.GeoDataFrame(geometry=gdf_ems_cleveland.geometry).buffer(8046.72)

# Plot
ax = gdf_cleveland.plot(figsize=(20,6), color='none',edgecolor='black')
gdf_accidents_cleveland_spatial.plot(color='blue',label='Accidents',ax=ax)
ems_5m_coverage.plot(color='bisque',edgecolor='red',markersize=80,alpha=0.6,ax=ax,)
gdf_ems_cleveland.plot(color='red',markersize=80,label='EMS',ax=ax)
ax.set_title('EMS 5 Miles Buffer Coverage ',size=20)
ax.legend()


# In[ ]:


ems_5m_coverage_union = ems_5m_coverage.geometry.unary_union
range_5m_ems = gdf_accidents_cleveland_spatial[gdf_accidents_cleveland_spatial['geometry'].apply(lambda x: ems_5m_coverage_union.contains(x))]
print('Comment: Accidents located in North-East and South-East areas of Cleveland are out of the range.That is {:.2%} of accidents'.format(1-(len(range_5m_ems)/len(gdf_accidents_cleveland_spatial))))


# #### Hospitals Coverage within 5 Miles Radius

# In[ ]:


hospitals_5m_coverage = gpd.GeoDataFrame(geometry=gdf_hospitals_cleveland.geometry).buffer(8046.72)

# Plot 
ax = gdf_cleveland.plot(figsize=(20,6), color='none',edgecolor='black')
gdf_accidents_cleveland_spatial.plot(color='blue',label='Accidents',ax=ax)
hospitals_5m_coverage.plot(color='bisque',edgecolor='red',markersize=80,alpha=0.6,ax=ax,)
gdf_hospitals_cleveland.plot(color='red',markersize=80,label='Hospitals',ax=ax)
ax.set_title('Hospitals 5 Miles Buffer Coverage ',size=20)
ax.legend()


# In[ ]:


hospitals_5m_coverage_union = hospitals_5m_coverage.geometry.unary_union
range_5m_hospitals = gdf_accidents_cleveland_spatial[gdf_accidents_cleveland_spatial['geometry'].apply(lambda x: hospitals_5m_coverage_union.contains(x))]
print('Comment: Accidents located in North-East area of Cleveland are out of the range.That is {:.2%} of accidents'.format(1-(len(range_5m_hospitals)/len(gdf_accidents_cleveland_spatial))))


# In[ ]:




