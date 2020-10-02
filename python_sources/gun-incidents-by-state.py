#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The goal of this kernel is to evaluate the severity of gun incidents across the United States.

# # Setup

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.offline as py
from matplotlib import pyplot as plt
from gc import collect as gc_collect
from time import time as t_time
from sklearn.cluster import KMeans

plt.style.use('fivethirtyeight')
py.init_notebook_mode()

def timer():
    print('Time taken: {:.3f}s'.format(t_time()-start))

def dt(date):
    return np.datetime64(date)


# # Preparing data

# ## Importing gun incidents data

# In[ ]:


start = t_time()

incidents = pd.read_csv('../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv',
                        parse_dates=['date'],
                        usecols=['date', 'state', 'city_or_county', 'n_killed', 'n_injured', 'longitude', 'latitude']
                       ).rename(columns={'city_or_county':'city'})

incidents['city'] = incidents['city'].apply(lambda x: x.replace('(county)', '').replace(' County', ''))

timer()

incidents.head()


# ## Importing census data

# In[ ]:


start = t_time()

state_codes = {
    'Alabama' : 'AL',
    'Alaska' : 'AK',
    'Arizona' : 'AZ',
    'Arkansas' : 'AR',
    'California' : 'CA',
    'Colorado' : 'CO',
    'Connecticut' : 'CT',
    'Delaware' : 'DE',
    'District of Columbia' : 'DC',
    'Florida' : 'FL',
    'Georgia' : 'GA',
    'Hawaii' : 'HI',
    'Idaho' : 'ID',
    'Illinois' : 'IL',
    'Indiana' : 'IN',
    'Iowa' : 'IA',
    'Kansas' : 'KS',
    'Kentucky' : 'KY',
    'Louisiana' : 'LA',
    'Maine' : 'ME',
    'Maryland' : 'MD',
    'Massachusetts' : 'MA',
    'Michigan' : 'MI',
    'Minnesota' : 'MN',
    'Mississippi' : 'MS',
    'Missouri' : 'MO',
    'Montana' : 'MT',
    'Nebraska' : 'NE',
    'Nevada' : 'NV',
    'New Hampshire' : 'NH',
    'New Jersey' : 'NJ',
    'New Mexico' : 'NM',
    'New York' : 'NY',
    'North Carolina' : 'NC',
    'North Dakota' : 'ND',
    'Ohio' : 'OH',
    'Oklahoma' : 'OK',
    'Oregon' : 'OR',
    'Pennsylvania' : 'PA',
    'Puerto Rico' : 'PR',
    'Rhode Island' : 'RI',
    'South Carolina' : 'SC',
    'South Dakota' : 'SD',
    'Tennessee' : 'TN',
    'Texas' : 'TX',
    'Utah' : 'UT',
    'Vermont' : 'VT',
    'Virginia' : 'VA',
    'Washington' : 'WA',
    'West Virginia' : 'WV',
    'Wisconsin' : 'WI',
    'Wyoming' : 'WY'
}

census = pd.read_csv('../input/us-census-demographic-data/acs2015_county_data.csv', usecols=['State', 'TotalPop']).rename(columns={'State':'state', 'TotalPop':'population'})
census = census.groupby('state').sum().reset_index()
census['state_code'] = census['state'].apply(lambda x: state_codes[x])

timer()

census.head()


# ## Importing areas data

# In[ ]:


start = t_time()
areas = pd.read_csv('../input/usa-areas/usa-areas.csv')
timer()
areas.head()


# ## Checking data coherence

# In[ ]:


start = t_time()
incidents.groupby('date').sum()[['n_killed', 'n_injured']].plot(ax=plt.subplots(figsize=(15,2))[1])
timer()


# The data from 2013 seems a little too different from the rest of the dataset. Let's drop it.

# In[ ]:


start = t_time()
incidents = incidents[incidents['date']>dt('2014')].reset_index(drop=True)
gc_collect()
incidents.groupby('date').sum()[['n_killed', 'n_injured']].plot(ax=plt.subplots(figsize=(15,2))[1])
timer()


# ## Spreading date feature

# In[ ]:


start = t_time()
incidents['year'] = incidents['date'].dt.year
incidents['weekday'] = incidents['date'].apply(lambda x: x.weekday())
incidents = incidents[['date', 'year', 'weekday', 'state', 'city', 'n_injured', 'n_killed', 'longitude', 'latitude']]
gc_collect()
timer()
incidents.head()


# # Data visualization

# ## Gun incidents through weekdays

# In[ ]:


start = t_time()
incidents.groupby('weekday').sum()[['n_killed', 'n_injured']].plot(ax=plt.subplots(figsize=(15,5))[1])
plt.title('Injuries and deaths per weekday (0=monday, ..., 6=sunday)')
timer()


# ## How violent is each state?

# In[ ]:


start = t_time()
incidents_by_state = incidents.drop(columns=['year', 'weekday', 'longitude', 'latitude']).groupby('state').sum().reset_index()
incidents_by_state['n_damaged'] = 2*incidents_by_state['n_killed'] + incidents_by_state['n_injured']
incidents_by_state = pd.merge(left=incidents_by_state, right=census, how='left', on='state')
incidents_by_state = pd.merge(left=incidents_by_state, right=areas, how='left', on='state')
gc_collect()
timer()
incidents_by_state.drop(columns=['n_damaged']).head()


# In[ ]:


start = t_time()
incidents_by_state.groupby('state').mean().sort_values(by='n_damaged', ascending=False)[['n_injured', 'n_killed']].plot(kind='bar', ax=plt.subplots(figsize=(15,5))[1])
plt.title('Gun incidents')
timer()


# In[ ]:


data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = incidents_by_state['state_code'],
        z = incidents_by_state['n_injured'],
        locationmode = 'USA-states',
        text = incidents_by_state['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Injuries")
        ) ]

layout = dict(
        title = 'Injuries from gun incidents (2014 - 2018)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
        )
    
figure = dict(data=data, layout=layout)
py.iplot(figure)

data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = incidents_by_state['state_code'],
        z = incidents_by_state['n_killed'],
        locationmode = 'USA-states',
        text = incidents_by_state['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Deaths")
        ) ]

layout = dict(
        title = 'Deaths from gun incidents (2014 - 2018)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
        )
    
figure = dict(data=data, layout=layout)
py.iplot(figure)


# Some areas have higher population densities so comparing their gun incidents' absolute numbers is not fair. Let's normalize the data.

# In[ ]:


start = t_time()
for column in ['n_killed', 'n_injured', 'n_damaged']:
    incidents_by_state[column+'_norm'] = incidents_by_state[column]*incidents_by_state['area']/incidents_by_state['population']
incidents_by_state.groupby('state').mean().sort_values(by='n_damaged_norm', ascending=False)[['n_injured_norm', 'n_killed_norm']].plot(kind='bar', ax=plt.subplots(figsize=(15,5))[1])
plt.title('Violence (normalized by population density)')
timer()


# Alaska is way too big, thus our statistics are unbalanced. Let's exclude it from our visualizations.

# In[ ]:


start = t_time()
incidents_by_state.groupby('state').mean().sort_values(by='n_damaged_norm', ascending=False)[['n_injured_norm', 'n_killed_norm']][1:].plot(kind='bar', ax=plt.subplots(figsize=(15,5))[1])
plt.title('Violence (normalized by population density)')
timer()


# In[ ]:


data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = incidents_by_state[incidents_by_state['state']!='Alaska']['state_code'],
        z = incidents_by_state[incidents_by_state['state']!='Alaska']['n_injured_norm'],
        locationmode = 'USA-states',
        text = incidents_by_state[incidents_by_state['state']!='Alaska']['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Injuries')
        ) ]

layout = dict(
        title = 'Injuries (normalized by population density)<br>Except for Alaska',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
        )
    
figure = dict(data=data, layout=layout)
py.iplot(figure)

data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = incidents_by_state[incidents_by_state['state']!='Alaska']['state_code'],
        z = incidents_by_state[incidents_by_state['state']!='Alaska']['n_killed_norm'],
        locationmode = 'USA-states',
        text = incidents_by_state[incidents_by_state['state']!='Alaska']['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Deaths')
        ) ]

layout = dict(
        title = 'Deaths (normalized by population density)<br>Except for Alaska',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
        )
    
figure = dict(data=data, layout=layout)

gc_collect()

py.iplot(figure)


# ### Where are the most violent incidents?

# In[ ]:


start = t_time()
incidents_by_state = incidents.drop(columns=['year', 'weekday', 'longitude', 'latitude']).groupby('state').mean().reset_index()
incidents_by_state['n_damaged'] = 2*incidents_by_state['n_killed'] + incidents_by_state['n_injured']
incidents_by_state = pd.merge(left=incidents_by_state, right=census[['state', 'state_code']], how='left', on='state')
timer()
incidents_by_state.drop(columns=['n_damaged']).head()


# In[ ]:


start = t_time()
incidents_by_state.groupby('state').mean().sort_values(by='n_damaged', ascending=False)[['n_injured', 'n_killed']].plot(kind='bar', ax=plt.subplots(figsize=(15,5))[1])
plt.title('Violence per incident')
timer()


# In[ ]:


data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = incidents_by_state['state_code'],
        z = incidents_by_state['n_injured'],
        locationmode = 'USA-states',
        text = incidents_by_state['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Injuries")
        ) ]

layout = dict(
        title = 'Injuries per gun incident (2014 - 2018)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
        )
    
figure = dict(data=data, layout=layout)
py.iplot(figure)

data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = incidents_by_state['state_code'],
        z = incidents_by_state['n_killed'],
        locationmode = 'USA-states',
        text = incidents_by_state['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Deaths")
        ) ]

layout = dict(
        title = 'Deaths per gun incident (2014 - 2018)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
        )

del incidents_by_state
gc_collect()

figure = dict(data=data, layout=layout)
py.iplot(figure)


# ## Focuses of violence

# ### Clustering by longitude and latitude

# In[ ]:


start = t_time()

N_CLUSTERS = 250

incidents = incidents[(incidents['longitude'].notna()) & (incidents['longitude'].notna())].reset_index(drop=True)

incidents['cluster'] = KMeans(N_CLUSTERS, n_init=2, max_iter=50, random_state=42).fit_predict(incidents[['longitude', 'latitude']])
incidents_groupby_cluster = incidents[['state', 'city', 'longitude', 'latitude', 'n_injured', 'n_killed', 'cluster']].groupby('cluster')

incidents_by_cluster = incidents_groupby_cluster.sum().reset_index(drop=True)
incidents_by_cluster['longitude'] = incidents_groupby_cluster.mean()[['longitude']]
incidents_by_cluster['latitude'] = incidents_groupby_cluster.mean()[['latitude']]

timer()


# ### Plotting

# In[ ]:


def join_names(names):
    return '/'.join([name for name in sorted(set(names))])

incidents_by_cluster['states'] = incidents_groupby_cluster['state'].apply(join_names)
incidents_by_cluster['severity'] = (2*incidents_by_cluster['n_killed']+incidents_by_cluster['n_injured'])**0.3
incidents_by_cluster['death_pct'] = incidents_by_cluster['n_killed']/(incidents_by_cluster['n_killed']+incidents_by_cluster['n_injured'])
incidents_by_cluster.sort_values(by=['severity', 'death_pct'], inplace=True)

data = [ dict(
    type = 'scattergeo',
    lon = incidents_by_cluster['longitude'],
    lat = incidents_by_cluster['latitude'],
    text = '<br><b>States</b><br>'+incidents_by_cluster['states']
        +'<br><br><b>Longitude</b><br>'+incidents_by_cluster['longitude'].apply(str)
        +'<br><br><b>Latitude</b><br>'+incidents_by_cluster['latitude'].apply(str)
        +'<br><br><b>Injuries</b><br>'+incidents_by_cluster['n_injured'].apply(str)
        +'<br><br><b>Deaths</b><br>'+incidents_by_cluster['n_killed'].apply(str),
    marker = dict(
        size = 3*incidents_by_cluster['severity'],
        line = dict(
            width = 0.5
        ),
        cmin = incidents_by_cluster['severity'].min(),
        color = incidents_by_cluster['severity'],
        cmax = incidents_by_cluster['severity'].max(),
        opacity = 0.95,
    )
)]

layout = dict(
    title = 'Focuses of violence',
    geo = dict(
        scope = 'usa'
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig)


# # Conclusion
# 
# ## Summary
# 
# * The USA is more susceptible to gun incidents on the weekends
# 
# ### Rankings
# 
# * States with the most injuries from gun incidents:
# 
#     * Illinois, California, Florida, Texas, Ohio
# 
# * States with the most deaths from gun incidents:
# 
#     * California, Texas, Florida, Illinois, Ohio
# 
# * States with with the most injuries from gun incidents (normalized by population density):
# 
#     * Texas, Illinois, Louisiana, Missouri, Nevada
# 
# * States with with the most deaths from gun incidents (normalized by population density):
# 
#     * Texas, Nevada, New Mexico, California, Missouri
# 
# * States with the highest injury ratio per gun incident:
# 
#     * Illinois, Virginia, Tennessee, Pennsylvania, Ohio
# 
# * States with the highest death ratio per gun incident:
# 
#     * Arizona, Nevada, Texas, Alabama, California
# 
# Now let's see the frequency that these states appear on the rankings:
# 
# 1. Texas (5)
# 1. California/Illinois (4)
# 1. Ohio/Nevada (3)
# 1. Missouri/Florida (2)
# 1. Louisiana/New Mexico/Virginia/Tennessee/Pennsylvania/Arizona/Alabama (1)
# 
# ## Grains of salt
# 
# * The census dataset is from **2015**
# * The considered area was the **total** area of each state
# * There is no guarantee that the data about gun incidents is equally complete for each state
