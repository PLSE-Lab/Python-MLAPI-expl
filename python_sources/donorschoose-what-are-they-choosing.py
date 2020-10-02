#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np # linear algebra
from numpy import log10, ceil, ones
from numpy.linalg import inv 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # prettier graphs
import matplotlib.pyplot as plt # need dis too
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import HTML # for da youtube memes
import itertools # let's me iterate stuff
from datetime import datetime # to work with dates
import geopandas as gpd
from fuzzywuzzy import process
from shapely.geometry import Point, Polygon
import shapely.speedups
shapely.speedups.enable()
import fiona 
from time import gmtime, strftime
from shapely.ops import cascaded_union
import gc
import folium # leaflet.js py map
from folium import plugins

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

sns.set_style('darkgrid') # looks cool, man
import os

df_donations = pd.read_csv("../input/io/Donations.csv")
df_donors = pd.read_csv("../input/io/Donors.csv", low_memory=False)
df_projects = pd.read_csv("../input/io/Projects.csv", error_bad_lines=False)
df_resources = pd.read_csv("../input/io/Resources.csv", error_bad_lines=False)
df_schools = pd.read_csv("../input/io/Schools.csv", error_bad_lines=False)
df_teachers = pd.read_csv("../input/io/Teachers.csv", error_bad_lines=False)

df_population = pd.read_csv("../input/population-by-state/population.csv")
df_st_abbrev = pd.read_csv("../input/state-abbreviations/state_abbrev.csv")

df_schools.rename(columns={'School State': 'state', 'School City': 'city'}, inplace=True)
df_donors.rename(columns={'Donor State': 'state', 'Donor City': 'city'}, inplace=True)


# # DonorsChoose
# Kaggle has another great data and modeling challenge with the crowdsourced donation platform [DonorsChoose](https://www.donorschoose.org), seeking assistance with help in ingesting, evaluating, prioritizing and approving the many applicants they receive for worthy projects to be posted on the platform.  I do not have a time to take a stab at that right now, but am taking a break from some other things I am working on to put some effort into a draft Exploratory Data Analysis (EDA) kernel to see what kind of neat things we may suss out, and perhaps give some others ideas to help with their exploration and this cause.  **Consider an upvote if you like my work and you feel you learned something interesting from it.**  Thanks!

# # Dirty Data
# There's some pretty dirty stuff in here.  As an example, here's the count of Seattle donors.  I am unclear on how the state would be set so incorrectly for them, as I would assume this is something the user would specify in their donor account.  Note there is 1. no Seattle in these other places (I assume, but did not check, all) and also note that 2. I shortened the list here for display purposes.

# In[122]:


df_donors[df_donors['city'] == 'Seattle'][['city', 'state']].groupby(['city', 'state']).size().reset_index().rename(columns={0: 'count'}).sort_values('count', ascending=False).head(15)


# 107,809 donors don't resolve to states *at all* of 2,122,640, or 5.08%.  As we can see this includes 120 Seattle (other) - but doesn't include a whole bunch of Seattle in states that don't actually even have a Seattle.  Certainly a city can be in multiple states, although we can probably help clean up this data a little bit.  We can make a unique set of city-state combos, get a count of donors, and for all those assigned with the state of 'other' we can assign them the state of the most numerous city match.  Ie. Seattle *in Washington* will always win for the above; although we'll only set the value for those records in which the state is currently defined as 'other'.  Let's try it.

# In[59]:


df1 = df_schools[~df_schools['city'].isnull()][['city', 'state']]
df2 = df_donors[(df_donors['state'] != 'other') & (~df_donors['city'].isnull())][['city', 'state']]
# all existing seemingly valid city state pairs
frames = [df1, df2]
df_city_state_map = pd.concat(frames)
df_city_state_map.rename(columns={'state': 'map_state', 'city': 'map_city'}, inplace=True)

# count, keep the city/state pair with largest count as map
df_city_state_map = df_city_state_map.groupby(['map_city', 'map_state']).size().reset_index().rename(columns={0: 'count'})
df_city_state_map['rank'] = df_city_state_map.groupby(['map_city'])['count'].rank(ascending=False)
df_city_state_map = df_city_state_map[df_city_state_map['rank'] == 1]

# fix the data
df_donors = df_donors.merge(df_city_state_map[['map_city', 'map_state']], how='left', left_on='city', right_on='map_city')
df_donors['state'] = np.where((df_donors['state'] == 'other') & (~df_donors['city'].isnull()), df_donors['map_state'], df_donors['state'])
df_donors.drop(columns=['map_city', 'map_state'], inplace=True)

# show count now
df_donors[df_donors['city'] == 'Seattle'][['city', 'state']].groupby(['city', 'state']).size().reset_index().rename(columns={0: 'count'}).sort_values('count', ascending=False).head(5)


# Well, we've got some good and bad here...  the good first, Seattle 'other' got assigned to Seattle, Washington.  Yay!  So did other cities.  The bad...  95,082 are null anyway.  So 5.08% unresolved moved down to 4.48% that can't be resolved to US states...  I mean, a small win, but not that great either.  Oh well, they can't all be winning ideas, am I right?  At least now I can add state abbreviations for some maps and such.

# In[60]:


df_schools = df_schools.merge(df_st_abbrev[['State', 'Abbreviation']], how='left', left_on='state', right_on='State').drop(columns=['State'])
df_schools.rename(columns={'Abbreviation': 'st'}, inplace=True)
df_donors = df_donors.merge(df_st_abbrev[['State', 'Abbreviation']], how='left', left_on='state', right_on='State').drop(columns=['State'])
df_donors.rename(columns={'Abbreviation': 'st'}, inplace=True)


# # Registered Donor Locations
# Where our are donors located?

# In[61]:


df_state = df_donors.groupby(['st']).size().reset_index().rename(columns={0: 'count'})

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['st'],
        z = df_state['count'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'count of donors')
        ) ]

layout = dict(
        title = 'Donors by State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# Of course, we also may simply be measuring population above; let's take a look at the population adjusted rate; the picture changes quite a bit in fact!

# In[62]:


df_state = df_state.merge(df_population, how='left', left_on='st', right_on='State').drop(columns='State')
df_state['donors_per_pop'] = df_state['count'] / df_state['Population']

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['st'],
        z = df_state['donors_per_pop'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'registered donors per population')
        ) ]

layout = dict(
        title = 'Donors by State Per Capita',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# # Money Flows
# What's the median donation?  $25 is very common.

# In[63]:


df_donations = df_donations.merge(df_donors[['Donor ID', 'state', 'st', 'Donor Is Teacher']], how='left', on='Donor ID').drop(columns=['Donor ID'])
df_donations.rename(columns={'state': 'Donor State', 'st': 'Donor ST'}, inplace=True)


# In[64]:


df_state = df_donations.groupby(['Donor ST'])['Donation Amount'].median().reset_index()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['Donor ST'],
        z = df_state['Donation Amount'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Median Donations')
        ) ]

layout = dict(
        title = 'median donation FROM state',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# What about the average donation?  As we saw in my [Kiva Lenders kernel](https://www.kaggle.com/doyouevendata/an-exploratory-look-at-kiva-lenders), this can strongly be influenced by crowdfunding 'whales', although I did not look for them in this dataset as of yet.  They are sure to exist however and can skew a statistic like average significantly while having little effect on the median above.

# In[65]:


df_state = df_donations.groupby(['Donor ST'])['Donation Amount'].mean().reset_index()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['Donor ST'],
        z = df_state['Donation Amount'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Mean Donations')
        ) ]

layout = dict(
        title = 'mean donation FROM state',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# How about the median donation *to* a state?  Note, the code methodology here skews a slight bit (enough I'd say for a don't care) as 4 projects appear to span multiple schools; I have 4 extra records after my initial joins below.  I did not dig into why but this is my speculation at my point.  It's a very small number compared to the overall amount of donations so it is unlikely to change anything meaningful in the presentation of this data.

# In[66]:


df_state = df_donations.merge(df_projects[['Project ID', 'School ID']], how='left', on='Project ID')
df_state = df_state.merge(df_schools[['School ID', 'state', 'st']], how='left', on='School ID')
df_state.rename(columns={'state': 'School State', 'st': 'School ST'}, inplace=True)


# In[67]:


df_display = df_state.groupby(['School ST'])['Donation Amount'].median().reset_index()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_display['School ST'],
        z = df_display['Donation Amount'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Median Donations')
        ) ]

layout = dict(
        title = 'median donation TO state',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# What percent of money stays within the state, when we actually know the donor state?

# In[68]:


df_state['donation_within'] = np.where(df_state['Donor ST'] == df_state['School ST'], 'Yes', 'No')
df_display = df_state[~df_state['Donor ST'].isnull()]['donation_within'].value_counts().to_frame().reset_index()
df_display.rename(columns={'donation_within': 'count'}, inplace=True)
df_display.rename(columns={'index': 'donation_within'}, inplace=True)

trace = go.Pie(labels=df_display['donation_within'], values=df_display['count'], marker=dict(colors=['#75e575', '#ea7c96']))

py.iplot([trace], filename='basic_pie_chart')


# 68.2% is a very significant amount.  People are absolutely not equally weighting all loans and showing a heavy preference to providing aid to schools in the state in which they live.  This could even additionally be explored at a region or city level, although that will take a lot more effort.  An algorithm to accelerate the applicant approval process should take into account how many registered users there are in a state vs. the location of the school applicant, and maybe even the amount of other competing openly funding loans at the time.  I am not surprised at these results being a pretty strong signal.
# 
# Of course, this also varies by state; so an even more detailed model could take that into account.

# In[69]:


df_display = df_state[~df_state['Donor ST'].isnull()].groupby(['Donor ST', 'donation_within']).size().reset_index().rename(columns={0: 'count'})
df_display = df_display.pivot(index='Donor ST', columns='donation_within', values='count').reset_index()
df_display['percent_within_state'] = df_display['Yes'] / (df_display['No'] + df_display['Yes'])

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_display['Donor ST'],
        z = df_display['percent_within_state'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Same State Donations')
        ) ]

layout = dict(
        title = 'Percent of Donations Kept Within State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# A whopping 85% of Oklahoma donors contributed their donations to in state schools.  I do not think it is a stretch to say this is because the public funding in this state has been significantly cut.
# 
# ![](https://i1.wp.com/okpolicy.org/wp-content/uploads/education-cuts-465x1024.jpg)
# 
# Utah is not far behind.  It is also [over 60% Mormon](http://archive.sltrib.com/article.php?id=1842825&itype=CMSID), the strongest area for representation by this religious community.  Most Kaggle competitions are more straight mathy than this so these kinds of things matter not; but it should be noted in those areas where people are more creatively exploring US data, from donations to firearms, that the US is a very large and diverse nation.  It has a very heterogeneous mix of ethnicities, religions, culture, history, and policies.

# # Other Donor Demographics
# How many donors are teachers?  Quite a few, and they know the struggle of having to raise funds to complete projects in their own classrooms.

# In[70]:


df_pie = df_donations.groupby('Donor Is Teacher').size().to_frame().reset_index().rename(columns={0: 'count'})

trace = go.Pie(labels=df_pie['Donor Is Teacher'], values=df_pie['count'], marker=dict(colors=['#ea7c96', '#75e575']))

py.iplot([trace], filename='basic_pie_chart')


# Optional donations can be included - these are donations directly to DonorsChoose and not the cause being fundraised, so that DonorsChoose can continue to operate and expand it's business.  What percentage of donations actually include this optional fee to support the organization?

# In[71]:


df_pie = df_donations.groupby('Donation Included Optional Donation').size().to_frame().reset_index().rename(columns={0: 'count'})

trace = go.Pie(labels=df_pie['Donation Included Optional Donation'], values=df_pie['count'], marker=dict(colors=['#ea7c96', '#75e575']))

py.iplot([trace], filename='basic_pie_chart')


# I think 82.1% is a pretty impressive amount.  We are not sure of what percentage they give, but that is quite a good participation rate, and is above what I would have guessed myself.

# # Free Lunch Qualification
# Let's explore this a bit.  What's the range like?  This is sorted on the 50% median values.  This metric is described as the "Integer describing percentage of students qualifying for free or reduced lunch, obtained from NCES data. For schools without NCES data, a district average is used."

# In[72]:


df_schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe().reset_index().sort_values('50%')


# I'm a bit surprised at these numbers overall, really, as they are higher than I would have expected.  I think the point of this metric is to use it as a general poverty or need/assistance metric for a school.  The suburbs fare better, at least on the lower end of the scale, while urban areas generally do the worst.

# In[73]:


df_schools.sort_values('School Percentage Free Lunch', ascending=False).head()


# Cool Valley is in the northwest area of St. Louis, pretty close to the airport.  It's in the Ferguson-Florissant school district.  You may very well recognize Ferguson from fairly recent US news; The Atlatnic has an article that touches on [local economics and school funding as well](https://www.theatlantic.com/politics/archive/2015/04/fergusons-fortune-500-company/390492/).  Detroit International Academy is the only all women's public school in Detroit, which is interesting.  Columbus Gifted Academy was [established recently](http://www.dispatch.com/content/stories/local/2015/08/26/given-room-to-grow.html) (2015) for high performing students.  As the article notes it's considered a special program and not necessarily a school itself.  It doesn't exactly seem like a program/school that would actually lack funding.  Ronan is a small city on an [Indian reservation](https://en.wikipedia.org/wiki/Flathead_Indian_Reservation).  These areas have not all necessarily grown economically at the rate of other parts of the US due to management and ownership between residents/local government and the US federal government.  These are 5 of 42 schools with 100%.
# 
# # Chicago
# I'm from the Chicagoland area.  I was curious how the stats looked for the city.

# In[127]:


df_schools[df_schools['city'] == 'Chicago']['School Percentage Free Lunch'].describe().to_frame()


# Assigning school zips to lat/longs as placing them within Chicago communities, we can take an average of the School Percentage Free Lunch and use it as a proxy for need.  What's that look like?  Note, communities which did not have data using the methodology in this code are not graphed.

# In[131]:


df_zips = pd.read_csv("../input/us-zip-codes-with-lat-and-long/zip_lat_long.csv")
gdf_areas = gpd.read_file('../input/chicago-community-areas-geojson/chicago-community-areas.geojson')

epsg = '32616'

gdf_schools = df_schools.merge(df_zips, how='left', left_on='School Zip', right_on='ZIP')
gdf_schools['geometry'] = gdf_schools.apply(lambda row: Point(row['LNG'], row['LAT']), axis=1)
gdf_schools = gpd.GeoDataFrame(gdf_schools, geometry='geometry')
gdf_schools.drop(columns=['ZIP', 'LAT', 'LNG'], inplace=True)
gdf_schools.crs = {'init': epsg}

gdf_chi_schools = gdf_schools[gdf_schools['city'] == 'Chicago']


# In[132]:


gdf_chi_schools['community'] = np.NaN
gdf_chi_schools['r_map'] = np.NaN
### POINTS IN POLYGONS
for i in range(0, len(gdf_areas)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_chi_schools['r_map'] = gdf_chi_schools.within(gdf_areas['geometry'][i])
    gdf_chi_schools['community'] = np.where(gdf_chi_schools['r_map'], gdf_areas['community'][i], gdf_chi_schools['community'])


# In[133]:


df_temp = gdf_chi_schools.groupby('community')['School Percentage Free Lunch'].mean().to_frame().reset_index()
df_temp.rename(columns={'School Percentage Free Lunch': 'free_lunch_mean_zip'}, inplace=True)
gdf_areas = gdf_areas.merge(df_temp, how='left', on='community')


# In[141]:


# create Chicago map
CHICAGO_COORDINATES = [41.85, -87.68]
map_attributions = ('&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors, &copy; <a href="http://cartodb.com/attributions">CartoDB</a>')
community_map = folium.Map(location=CHICAGO_COORDINATES, 
                                 attr=map_attributions,
                         #        tiles=None, #'Cartodb Positron', #'OpenStreetMap',
                                 zoom_start=10, min_zoom=10,
                                 control_scale=True)

geojson = gdf_areas.to_json()
geojson = gdf_areas[~gdf_areas['free_lunch_mean_zip'].isnull()].to_json()

# map Chicago communities crime
community_map.choropleth(
    #geo_data='../input/chicago-community-areas-geojson/chicago-community-areas.geojson',
    geo_data=geojson,
    data=gdf_areas[~gdf_areas['free_lunch_mean_zip'].isnull()],
    columns=['community', 'free_lunch_mean_zip'],
    key_on='feature.properties.community',
    line_opacity=0.3,
    fill_opacity=0.5,
    fill_color='YlOrRd',
    #legend_name='Chicago Crime by Community (2001-2017)', 
    highlight=True, 
    #threshold_scale=[30, 41, 63, 73, 84, 95],
    smooth_factor=2
)

# add fullscreen toggle
#plugins.Fullscreen(
#    position='topright',
#    title='full screen',
#    title_cancel='exit full screen',
#    force_separate_button=True).add_to(community_crime_map)

# add base map tile options
folium.TileLayer('OpenStreetMap').add_to(community_map)
#folium.TileLayer('stamentoner').add_to(community_map)
folium.TileLayer('Cartodb Positron').add_to(community_map)
folium.LayerControl().add_to(community_map)

# show map
community_map


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[140]:


Blues = plt.get_cmap('Blues')

regions = gdf_areas[~gdf_areas['free_lunch_mean_zip'].isnull()]['community']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_areas[gdf_areas['community'] == r].plot(ax=ax, color=Blues(gdf_areas[gdf_areas['community'] == r]['free_lunch_mean_zip'] / 100/1.35))
    #gdf_areas[gdf_areas['community'] == r].plot(ax=ax, color='powderblue')

gdf_schools[(gdf_schools['city'] == 'Chicago')].plot(ax=ax, markersize=10, color='red')

for i, point in gdf_areas.centroid.iteritems():
    reg_n = gdf_areas.iloc[i]['community']
    reg_n = gdf_areas.loc[i, 'community']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='small')
    

ax.set_title('Chicago Neighborhoods; Darker Shade = Higher Percentage Free Lunch Qualification')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
#new_title = 'Partner ID'
#leg.set_title(new_title)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# **Consider an upvote if you like my work and you feel you learned something interesting from it.**  Thanks so much!
# 
# # Draft

# In[89]:


df_projects.head()


# In[90]:


df_resources.head()


# In[91]:


df_schools.head()


# In[92]:


df_teachers.head()


# In[93]:


df_donations.head()


# In[94]:


df_schools.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




