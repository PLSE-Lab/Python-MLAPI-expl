#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# This is a demonstration of __ChoroPie__, a Basemap/Matplotlib toolkit I created which allows the simplified creation of choropleth maps with colorbars using shapefiles, and the combined plotting of pie charts within the centroid coordinates of the shapefile's polygons.
# 
# The library can be installed with pip using _pip install choropie_
# 
# #### The final output:
# <img src="https://user-images.githubusercontent.com/30331170/33050018-b200156e-ce30-11e7-9ffa-b58885df2062.png" width="75%"/>
# 
# ##### Url to Github repo:
# https://github.com/vinceniko/choropie
# 
# ###### *Disclaimer: The colors used to present the racially focused data is not reflective of any kind of idealogy. I realize that some may find the use of these colors to be offensive, but no offense was implied or intended. The chosen colors are merely used to better explain the concepts being introduced below.

# # imports

# In[ ]:


import numpy as np
import pandas as pd
import os
import datetime as dt

from choropie import ChoroPie as cp


# # state names and abbreviations

# In[ ]:


dict_states = {
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


# # import general state population dataset and remove first three rows

# In[ ]:


df_pop = pd.read_excel('../input/us-state-population-estimates-racial-breakdowns/state_population_estimates.xlsx', skiprows=3)  # taken from census data


# In[ ]:


df_pop.head()


# # to remove '.' in front of state in df_pop

# In[ ]:


def remove_period(x):
    if isinstance(x, str) and x[0] == '.':
        x = x.replace('.', '')
        return x


df_pop.iloc[:, 0] = df_pop.iloc[:, 0].apply(remove_period)  # perform remove_period


# In[ ]:


df_pop.head()


# # select necessary state rows and correct year

# In[ ]:


series_pop = df_pop.set_index(df_pop.columns[0]).loc['Alabama':'Wyoming', 2016]
series_pop.name = 'population'


# In[ ]:


series_pop.head()


# # import police killings dataset

# In[ ]:


df_killings = pd.read_csv('../input/police-shootings-us-20152017-concated/PoliceKillingsUS.csv', encoding="latin1") # set proper encoding or get error. i combined the sheets into one file here


# In[ ]:


df_killings.head()


# # replace race abbreviations

# In[ ]:


def abr(x):
    try:
        if x[0] == 'A':
            return "Asian"
        if x[0] == 'B':
            return "Black"
        if x[0] == 'H':
            return "Hispanic"
        if x[0] == 'N':
            return "Native American"
        if x[0] == 'O':
            return "Ocean Pacific"
        if x[0] == 'W':
            return "White"
    except Exception:
        return None


df_killings['race'] = df_killings['race'].apply(abr)


# In[ ]:


df_killings.head()


# # use datetime module to extract min and max dates of dataset

# In[ ]:


# format dates to Jan. 01, 06 format
# used for title of plot
max_date = df_killings['date'].apply(
    lambda x: dt.datetime.strptime(x, '%d/%m/%y')).max().strftime('%b. %d, %y')
min_date = df_killings['date'].apply(
    lambda x: dt.datetime.strptime(x, '%d/%m/%y')).min().strftime('%b. %d, %y')


# # series breaking down count of killings by state

# In[ ]:


series_state = df_killings.groupby('state').count()['id']
series_state.rename('counts', inplace=True)


# # series breaking down count of killings by state and race (MultiIndex)

# In[ ]:


series_race = df_killings.groupby(['state', 'race']).count()['id']


# In[ ]:


series_race.head()


# # percentage of each race killed by state

# In[ ]:


series_state_crime_race_percs = series_race /     series_race.groupby('state').sum() * 100


# In[ ]:


def set_index_states(df):
    if isinstance(df.index, pd.MultiIndex):
        list_abb = [dict_states[abb] for abb in df.index.levels[0]]
        df.index.set_levels(list_abb, level=0, inplace=True)
    elif isinstance(df.index, pd.Index):
        list_abb = [dict_states[abb] for abb in df.index]
        df.index = list_abb


# fix indexes (replace state abbreviations with state name)
# series_race and series_state
set_index_states(series_race)
set_index_states(series_state)


# In[ ]:


series_race.head()


# In[ ]:


series_state.head()


# # df_state is the first df we will use for plotting

# In[ ]:


df_state = pd.concat([series_state, series_pop], axis=1)
# per capita percentage
df_state['per_capita'] = df_state['counts'] / df_state['population']


# In[ ]:


df_state.head()


# # intermediary step

# In[ ]:


# population by race for each state
df_state_race = pd.read_excel('../input/us-state-population-estimates-racial-breakdowns/state_race.xlsx', index_col=0)

df_state_race = df_state_race.iloc[1:, :]
df_state_race.columns.name = 'race'

# transform columns into multiindex
df_massaged = pd.melt(df_state_race.reset_index(),
                      id_vars='Geography', value_vars=df_state_race.columns)
df_massaged = df_massaged.groupby(
    ['Geography', 'race']).agg(lambda x: x.iloc[0])


# In[ ]:


df_massaged.head()


# # df_race is the second df we will use for plotting

# In[ ]:


df_race = pd.concat([series_race, series_state_crime_race_percs,
                     df_massaged], axis=1).dropna()
df_race.columns = ['count', 'percs', 'pop']  # count, percent, population
df_race['per_capita'] = df_race['count'] / df_race['pop']

###################


# In[ ]:


df_race.head()


# # shp file processing

# In[ ]:


shp_file = 'Data/cb_2016_us_state_500k/cb_2016_us_state_500k'

shp_lst = cp.get_shp_attributes(shp_file)
shp_key = cp.find_shp_key(df_state.index, shp_lst)  # which shp attribute holds our index values
###

basemap = dict(
    basemap_kwargs=dict(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64,
                        urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45, lon_0=-95),
    shp_file=shp_file,
    shp_key=shp_key,
    figsize=(22, 12),
)

choro = dict(
    num_colors=8,
    cmap='hot_r',
    color_data=df_state['counts'],
)

pie = dict(
    size_data=df_state['per_capita'],
    size_ratios=df_race['per_capita'],
    pie_data=df_race['percs'],
    pie_dict={'Asian': 'yellow', 'Black': 'black', 'Hispanic': 'brown',
              'Native American': 'red', 'Ocean Pacific': 'purple', 'White': 'white'},
    scale_factor_size=1,
    scale_factor_ratios=1 / 2
)


# # Arguments Explained:
# Where color_data and size data are Pandas single-index series with the area_names used in the shp file as the index.  
# Ie.  
# 
# area_name | per capita rate
# --- | ---
# alabama | .000010
# alaska | .000020
# arizona | .000017
# 
# Where pie_data and size_ratios are Pandas multi-index series with the area_names used in the shp file as the first index, and the pie chart slices (the ones passed into the pie_dict parameter), as the second index. 
# Ie.
# 
# area_name | race | per-race rate
# --- | --- | ---
# alabama | black | 0.000919
# alabama | white | 0.000188
# alaska | black | 0.000338
# alaska | native american | 0.001135
# alaska | white | 0.000105
# 
# ##### Notes-   
# * The ChoroPie class inherits directly from Basemap.
# * Pie plotting is optional. If pies are plotted, both size_data and size_ratios are optional. Not all pies have to be plotted as well (if it gets too cluttered...though in that case you can call the zoom_to_area method).  
# * Choropleth plotting is optional.  
# * The pie_dict parameter selects the colors for each pie slice.  
# 

# # create ChoroPie object

# In[ ]:


m = cp.ChoroPie(**basemap)


# # plot choropleths

# In[ ]:


m.choro_plot(**choro)


# # plot pies

# In[ ]:


m.pie_plot(**pie)


# # insert colorbar

# In[ ]:


m.insert_colorbar(colorbar_title='Map: Count of Killings',
                     colorbar_loc_kwargs=dict(location='right'))


# # insert legend for pie charts

# In[ ]:


m.insert_pie_legend(legend_loc='lower right',
                       pie_legend_kwargs=dict(title='Pies: Racial Breakdown'))


# # set title

# In[ ]:


m.ax.set_title('Police Killings: {} - {}\nTotal: {:,d}'.format(min_date, max_date,
    df_killings.iloc[:, 0].count()), fontsize=35, fontweight='bold', x=0.61, y=0.90)


# # ticks of the colorbar

# In[ ]:


m.ax_colorbar.set_yticklabels(['{:.0f}'.format(
    float(i.get_text())) for i in m.ax_colorbar.get_ymajorticklabels()])


# # display the map 

# In[ ]:


m.fig


# # Conclussions:

# ### Results:
# By examining the results we can see that:
# 1. California has had the most police killings.  
# 2. California has not had the highest per capita rate of police killings, with states such as New Mexico edging out ahead.  
# 3. In most states, the race with the most deaths were whites.  
# 4. Despite that, in states such as Oklahoma and Missiori, more blacks were killed proportionally when adjusted for the population differences of each race.  

# # Other examples:
# 
# Without size_data:  
# <img src="https://user-images.githubusercontent.com/30331170/33050049-ebfc0cd2-ce30-11e7-92df-84269f423ea8.png" width="60%" />
# 
# With size_data:  
# <img src="https://user-images.githubusercontent.com/30331170/33052907-04c44316-ce3f-11e7-9bb0-d3c426502de4.png" width="60%" />
# 
