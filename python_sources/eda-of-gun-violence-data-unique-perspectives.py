#!/usr/bin/env python
# coding: utf-8

# # Gun Violence in USA
# 
# A casual search on Google points out to the fact that "Even though it has half the population of the other 22 nations combined, the United States accounted for 82 percent of all gun deaths."
# 
# I have presented essential and some unique perspectives in the explorations and visualizations that I have undertaken using the Gun Violence Dataset by James Ko. This dataset consists of more than 230K records from gun incidents in USA between 2013 till 2018.
# 
# 1. **Data Cleaning**
#     2. Missing Values
# 3. **Nationwide Statistics**
# 4. **Statewide Statistics**
#     1. California
#     2. Texas
#     3. Florida
#     4. Illinois
#     5. Summary of States
# 10. **Incident wise breakdown**
# 11. **Gender perspective look at the data**
# 
# I share the same vision as uploader of the datset James Ko. Can we use this dataset to come up with more effective measured to curb violence based on guns? If so what needs to be done? I wish to look at this dataset in a nonpartisan manner, without taking any side in the gun debate and constantly update it as I make more new findings. Not all gun incidents are criminal in nature, if defensive use leads to death of perpetrator it is not a bad thing, though regrettably it has led to loss of life.
# 
# Currently given the time and computational resources at my disposal I limit my findings to exploratory nature. Will update it soon with further analysis.
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt, matplotlib
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import seaborn as sns

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import string, random
init_notebook_mode(connected=True)

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 20})
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
gun = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv', parse_dates=True)


# ## 1. Data Cleaning
# #### A. Missing Values

# In[2]:


percent_NA = pd.Series(["{0:.2f}%".format(val * 100) for val in gun.isnull().sum()/gun.shape[0]], index=list(gun.keys()))
print(percent_NA)


# At this juncture it is important to understand these null values. The invariable big advantage is the complete lack of any missing or null values in variables - $date$, $state$, $city\_or\_county$, $n\_killed$ and $n\_injured$ (and also $incident\_url$ and $incident\_url\_fields\_missing$). This is excellent, and points towards the excellent quality of the dataset. We need to use every statistic here to count in our analysis of casualties and fatalities of gun violence.
# 
# The variable $incident\_characteristics$ has only 0.13 % missing values which is so low it is almost in the acceptable. It can almost be regarded as miscategorization owing to complex nature of incidents or cracks in the system due to which due dilligence had not been carried out. It should not have any bearing on our conclusions. 6.88 % of $address$ values are not known, this only poses a mild problem. $notes$ column has high missing values, further notes was also found to be highly subjective in nature, hence for now notes are not being directly included as a reliable measure.
# 
# The 41.5% missing $gun\_stolen$, $gun\_type$ and $n\_guns\_involved$ probably refers to guns that were never recovered in the first place. It is very important to take this fact into account in our analysis. The very fact of missing data in these two variables itself should be a factor in our analysis. My analysis for now doesn't take into account gun type due to this unreliability.
# 
# Features $participant\_age$ and $participant\_type$ have substantial missing values. We have to exercise caution in drawing conclusions, because these are important metrics in our analysis. 
# 
# The largest percentage of missing values is observed in $participant\_relationship$ and $location\_description$, both of these are acceptable.

# In[3]:


gun_NA_gs = gun[['gun_stolen', 'gun_type', 'n_guns_involved']].isnull().any(axis=1)
NA_gs_bystate = gun[gun_NA_gs].groupby('state')['n_killed'].value_counts().unstack('state')
kills_bystate = gun.groupby('state')['n_killed'].value_counts().unstack('state')

plt.figure(1, figsize=(16, 6))
# (100* NA_gs_bystate.sum(axis=0)/kills_bystate.sum(axis=0)).sort_values().plot.bar()
val = (100* NA_gs_bystate.sum(axis=0)/kills_bystate.sum(axis=0)).sort_values()
g = sns.barplot(x=val.index, y=val.values)
g.set_xticklabels(labels=val.index, rotation=90)
plt.title('All reported cases considered')
plt.ylabel('% of missing Gun Data per Incident')


# As shown in the graph above, missing gun data is prevalent across the records obtained from all states and the missing percentage lies between 25 % and 50 %.

# ## 2. Nationwide Statistics

# In[4]:


city_fatalities = gun.groupby('city_or_county')['n_killed'].sum().sort_values(ascending=False)[:20]
city_latandlong = gun.groupby('city_or_county')[['latitude', 'longitude']].mean().loc[city_fatalities.index]
state_injured = gun.groupby('state')['n_injured'].sum().sort_values(ascending=False)
state_killed = gun.groupby('state')['n_killed'].sum().sort_values(ascending=False)
state_incidents = gun.groupby('state')['n_killed'].count()
# city_fatalities = gun.groupby('city_or_county')['n_killed'].sum().sort_values(ascending=False)[:20]
# latandlong = gun.groupby('city_or_county')[['latitude', 'longitude']].mean().loc[city_fatalities.index]

def plot_statewide_trend(state_trend, plot_title = 'State wise trends'):
    state_to_code = {'District of Columbia' : 'dc', 'Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}
    df = pd.DataFrame()
    df['state'] = state_trend.index
    df['counts'] = state_trend.values
    df['state_code'] = df['state'].apply(lambda x : state_to_code[x])
    data = [ dict(type='choropleth', colorscale = 'Reds', 
            autocolorscale = False, locations = df['state_code'],
            z = df['counts'], locationmode = 'USA-states',
            text = df['state'], marker = dict(line = dict(color = 'rgb(0, 0, 0)', width = 1)),
            colorbar = dict())]

    layout = dict(title = plot_title, geo=dict(scope='usa', projection=dict(type='albers usa'), 
                                               showlakes=True, lakecolor='rgb(255, 255, 255)'))
    fig = dict(data=data,layout=layout)
    iplot(fig, filename='map')


# In[5]:


plot_statewide_trend(state_incidents, 'State wise number of Gun Violence Incidents')


# In[6]:


plot_statewide_trend(state_killed, 'State wise number of lives lost to Gun Violence')


# * Undoubtedly, **California** lost most lives to gun violence (5562), followed closely by **Texas** (5046) and **Florida** (3909). 
# * Please, do note that the order changes when we take incidents statistics from above into account - **California** (16.3k), **Florida** (15k) and **Texas** (13.58k)

# In[7]:


def plot_city_trend(city_trend, latandlong, plot_title = 'City wise trends'):
    data = [ dict(type='scattergeo', locationmode = 'USA-states',# colorscale = 'Reds', autocolorscale = False, 
            lon=latandlong['longitude'], lat=latandlong['latitude'], text=city_trend.index,
            mode='markers',
            marker = dict(size = city_trend.values/20, opacity=0.7, cmin=0))]

    layout = dict(title = plot_title, colorbar=True, geo=dict(scope='usa', projection=dict(type='albers usa'), 
                                                              subunitcolor='rgb(0, 0, 0)', subunitwidth=0.5))
    fig = dict(data=data,layout=layout)
    iplot(fig, validate=False)


# In[8]:


plot_city_trend(city_fatalities, city_latandlong, 'Top 20 Deadliest cities in US')
print(city_fatalities)


# Chicago, Houston, Baltimore, St. Louis and Philadelphia are the deadliest cities in US in that order. Though Chicago and Houston are very highly populated cities, the level of violence in the Baltimore and St. Louis are far more dense population wise. This is also an observation reported by other kernels [here](https://www.kaggle.com/shivamb/stop-gun-violence-updated-exploration) and [here](https://www.kaggle.com/jpmiller/gun-violence-per-capita) .

# ## 3. Statewide Statistics

# In[9]:


gun_i = gun[gun['state'] == 'Illinois']
gun_c = gun[gun['state'] == 'California']
gun_t = gun[gun['state'] == 'Texas']
gun_f = gun[gun['state'] == 'Florida']
gun_m = gun[gun['state'] == 'Michigan']

gun_is = gun_i.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')
gun_cs = gun_c.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')
gun_ts = gun_t.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')
gun_fs = gun_f.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')
gun_ms = gun_m.groupby(['n_killed', 'city_or_county'])['n_injured'].value_counts().unstack('city_or_county')

gun_is.fillna(0, inplace=True)
gun_cs.fillna(0, inplace=True)
gun_ts.fillna(0, inplace=True)
gun_fs.fillna(0, inplace=True)

# Gun Database grouped by state and city
gun_groupedbystateandcity = gun.groupby(['state', 'city_or_county'])
# Gun Database longitude and latitude mean locations 
gun_bystateandcity_loc = gun_groupedbystateandcity[['latitude', 'longitude']].mean()
# Gun Database - statistics of number killed
gun_bystateandcity_kills = gun_groupedbystateandcity['n_killed'].value_counts()
# Gun Database - statistics of number injured
gun_bystateandcity_injured = gun_groupedbystateandcity['n_injured'].value_counts()

def get_state_stats(state, statistics_scale=20):
    # Incident statistic indices where no fatalities occur
    killzero = gun_bystateandcity_kills.loc[state].index.get_level_values(1) == 0
    counties = gun_bystateandcity_kills.loc[state][killzero].sort_values(ascending=False)[:20].index.get_level_values(0)
    killzerostats = gun_bystateandcity_kills.loc[state][killzero].loc[list(counties)].sort_values(ascending=False).values
    # Scaling the statistics using some measure to show 
    killzerostats = (killzerostats)/statistics_scale
    nkillednumbers = gun_groupedbystateandcity['n_killed'].sum().loc[state].sort_values(ascending=False)[:20]
    latandlong = gun_bystateandcity_loc.loc[state].loc[nkillednumbers.index]
    # Scaling the statistics using some measure to show 
    nkillednumbers = nkillednumbers/statistics_scale
    return killzerostats, nkillednumbers, latandlong

def plot_stats_stackedbar(gun_state):
    highcrimeareas = gun_state.sum().sort_values()[-20:].index

    data =[go.Bar(
        x=[str(indval) for indval in gun_state[highcrimeareas].loc[0:2].index],
        y=gun_state[highcrimeareas].loc[0:2].values[:,i],
        name=gun_state[highcrimeareas].loc[0:2].columns[i]
    ) for i in range(20)]

    layout = go.Layout(barmode='stack', width=800, height=600, 
                       xaxis=dict(title='Number of deaths, Number of injuries'),
                      yaxis=dict(title='Incident Statistics'))
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
   

def plot_state_focused(state_focused, latandlong, lonaxis, lataxis, state_scope, plot_title = 'State trends'):
    data = [ dict(type='scattergeo', locationmode = 'usa', # colorscale = 'Reds', autocolorscale = False, 
            lon=latandlong['longitude'], lat=latandlong['latitude'], text=nkillednumbers.index + ' - ' + nkillednumbers.astype('str'),#state_focused,
            mode='markers', marker = dict(size = state_focused/10, opacity=0.7, cmin=0))]

    layout = dict(title = plot_title, colorbar=True, 
                  geo=dict(resolution = 50, width=1000, height=1000, scope = state_scope, #['CA', 'AZ', 'Nevada', 'Oregon', ' Idaho'], 
                           showframe = True, showland = True, landcolor = "rgb(229, 229, 229)", showrivers = True,
                           showlakes = True,
                           showsubunits=True, subunitcolor = "#111",subunitwidth = 2,
#                           countrycolor = "rgb(0, 0, 0)", 
                           coastlinecolor = "rgb(0, 0, 0)", 
                           projection = dict(type = "Mercator"), # Mercator
                          county_outline={'color': 'rgb(0,0,0)', 'width': 2.5}, 
                           lonaxis = dict(range=lonaxis), lataxis = dict(range=lataxis), domain = dict(x = [0, 1], y = [0, 1])))
    fig = dict(data=data,layout=layout)
    iplot(fig, validate=False)


# ## California

# In[10]:


killzerostats, nkillednumbers, latandlong = get_state_stats('California', 1)

plot_stats_stackedbar(gun_cs)
plot_state_focused(nkillednumbers, latandlong, [-125, -114], [30.0, 40], 'CA', 'California State Numer of lives lost to gun violence')


# The problem in California seems widespread, Oakland leads the pack in terms of gun related incidents. But fatalities wise Los Angeles seems to be deadlier. We are grouping the gun incident statistics in terms of Number of people killed and injured. This gives us a very detailed insight into which city has gun incident problems but is still not so deadly and which city has the highest fatalities. Geospatial map gives us an idea of number of people killed, the aim of using Geospatial map is to see if we can identify trends in killings based on location! Here for California it is a mixed bag, big cities seem to attrack crime but so also does other regions which just seemingly suffer from high crime rate.

# ## Texas

# In[11]:


killzerostats, nkillednumbers, latandlong = get_state_stats('Texas', 1)

plot_stats_stackedbar(gun_ts)
plot_state_focused(nkillednumbers, latandlong, [-105, -90], [25.0, 35], 'TX', 'Texas Gun Violence trends')


# Houston leads the state as the deadliest place for gun violence, the major cities of Texas in terms of their economic importance also attract gun related incidents and fatalities. The surprise is number of gun related incidents in San Antonio!! 

# ## Florida

# In[12]:


killzerostats, nkillednumbers, latandlong = get_state_stats('Florida', 1)

plot_stats_stackedbar(gun_fs)
plot_state_focused(nkillednumbers, latandlong, [-90, -78], [23.0, 35], 'FL', 'Florida Gun Violence trends')


# The statistics in Florida are again a big surprise, the leading contender for violence is Jacksonville! I can't think of a reason why Jacksonville earns this dubious distinction in sunny state. Miami and Orlando owing to their economic importance atleast are not that big of a surprise but Jacksonille definitely was unexpected. 

# ## Illinois

# In[13]:


killzerostats, nkillednumbers, latandlong = get_state_stats('Illinois', 1)

plot_stats_stackedbar(gun_is)
plot_state_focused(nkillednumbers, latandlong, [-95, -85], [35.0, 45], 'IL', 'Illinois Gun Violence trends')


# Here the gun statistics arising from Illinois are unbelievable! The sheer number of incidents where no one is killed but someone is injured is so high in Chicago that the statistic definitely calls for some form of intervention. Plotted as a bubble plot (even scaled by a value of 10) it engulfs the entire state. Given the sheer numbers, there is asbolutely no precedent to the amount of gun violence apparent here. This is the ground zero for intervention, action or some measure to be adopted to address the scourge of gun violence.

# ## Summary of States

# In[14]:


gun_bystate = gun.groupby(['state'])
xval = gun_bystate['n_killed'].sum()
yval = gun_bystate['n_injured'].sum()
zval = xval + yval

data = [{'x': gun_bystate['n_killed'].sum()[0:6],'y': gun_bystate['n_injured'].sum()[0:6],
        'mode': 'markers',
        'marker': {'size': gun_bystate['n_killed'].sum(), 'showscale': True}}]
data = [{'x': xval, 'y': yval,
        'text': gun_bystate['n_killed'].sum().index,
        'mode': 'markers',
        'marker': { 'color': zval/100, 'size': zval/100, 'showscale': True}}]
layout = go.Layout(autosize=False,
    width=800, height=700,
    title='State wise Gun Fatality Statistics',
    xaxis=dict(title='Number of people killed'),
    yaxis=dict(title='Number of people injured'),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig = dict(data=data, layout=layout)
iplot(fig)


# While undoubtedly **California** has the highest number of lives lot to Gun violence, crime is pretty high in **Illinois** as shown in the above plot based on the number of people injured from gun incidents. The bubble plot here plots the bubble size as a sum of number of lives lost plus people injured from gun incidents, what makes it even more disturbing is Chicago is single handedly responsible for skewing this number. If crime in that city alone is controlled, no other city or region comes even close to the statistics exhibited by this city. 

# ## 4. Incident Wise Breakdown

# In[15]:


from collections import Counter

total_incidents = []
for i, each_inc in enumerate(gun['incident_characteristics'].fillna('Not Available')):
    split_vals = [x for x in re.split('\|', each_inc) if len(x)>0]
    total_incidents.append(split_vals)
    if i == 0:
        unique_incidents = Counter(split_vals)
    else:
        for x in split_vals:
            unique_incidents[x] +=1

unique_incidents = pd.DataFrame.from_dict(unique_incidents, orient='index')
colvals = unique_incidents[0].sort_values(ascending=False).index.values
find_val = lambda searchList, elem: [[i for i, x in enumerate(searchList) if (x == e)][0] for e in elem]

a = np.zeros((gun.shape[0], len(colvals)))
for i, incident in enumerate(total_incidents):
    aval = find_val(colvals, incident)
    a[i, np.array(aval)] = 1
incident = pd.DataFrame(a, index=gun.index, columns=colvals)


# 
# Again, a lot of incident characteristics is only a record of what transpired like "Shot - Wounded/Injured" so we will have to look into the data to see and filter characteristics that point towards nature of crime committed. We do that manually here. 

# In[16]:


prominent_incidents = incident.sum()[[4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 23]]
fig = {
    'data': [
        {
            'labels': prominent_incidents.index,
            'values': prominent_incidents,
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .45]},
        }
    ],
    'layout': {'title': 'Prominent Incidents of Gun Violence',
               'showlegend': False}
}
iplot(fig)

print('Number of people affected by Felons with guns')
print(gun[incident.iloc[:, 9]>0][['n_killed', 'n_injured']].sum())


# The most disturbing trend I can see here is "Possession of gun by felon or prohibited person"!! How can so many people with record have access to guns? Due to such incidents a total of 2306 people have lost their lives and 3986 people have been injured. This is the issue that needs to be addressed. Most of the incidents here bear hallmarks of crime, for ex: Drug involvement, Drive-by, Home invasion, Gang involvement.
# 
# "Defensive Use" doesn't show up as a prominent statistic compared to other incident characteristics, and it is so ironic that this statistic shows up right beside "Accidental/Negligient Discharge".
# 
# We can draw far more inferences from such a closer look at incident characteristics, let us now focus on Domestic Violence.

# ### Participant Type

# Here we are trying to get expand all the text fields in the dataframe columns - $participant\_gender$, $participant\_type$ and $participant\_status$ mainly to get an statistical idea of if there exists any reasoning to how the gun violence perpetrated is influenced based on which gender the perpetrator belongs to and which gender the victim belongs to. 
# 
# Such comparisons become more pertient depending on the incident type, for example in cases of domestic violence, let us see if this analysis gives raise to any particular insights.

# In[17]:


make_dict_from_entry = lambda value: dict([re.split(':+', item) for item in [s for s in re.split(r'(\|)', value) if len(s)>1]])
d = {'Gender' : pd.DataFrame(gun['participant_gender'].fillna('0::Not Available').map(make_dict_from_entry).tolist()), 
     'Type': pd.DataFrame(gun['participant_type'].fillna('0::Not Available').map(make_dict_from_entry).tolist()),
     'Status': pd.DataFrame(gun['participant_status'].fillna('0::Not Available').map(make_dict_from_entry).tolist())
    }
p = pd.concat(d.values(), axis=1, keys=d.keys())


# **Special Note**: This section is the most computationally intensive position of the entire notebook. This takes upwards of 20 minutes to run, I would warn anybody intending to fork this notebook of this computational expense. If there is a more computationally efficient way of implementing this, I would welcome suggestions.

# In[18]:


find_stats = lambda x: x.unstack().transpose().groupby(['Gender', 
                                        'Type'])['Status'].value_counts().unstack(['Type', 
                                                                                   'Gender']).sum()
Participant = p.apply(find_stats, axis=1)


# An anomaly is detected at location 49384 that is set as follows

# In[19]:


Participant.loc[49384]['Victim', 'Male'] = 1
Participant1 = Participant.drop(['Male, female', 'Not Available'], axis=1, level=1)
Participant1 = Participant1.fillna(0)


# ## 5. A Gender perspective look at the data

# In[20]:


# Filtering only male suspect incidents
Male_suspect = (Participant1['Subject-Suspect', 'Male']>0) & (Participant1['Subject-Suspect', 'Female']==0)
# Filtering only female suspect incidents
Female_suspect  = (Participant1['Subject-Suspect', 'Female']>0) & (Participant1['Subject-Suspect', 'Male']==0)
# Filtering incidents where suspects are male and female
Mix_suspect = (Participant1['Subject-Suspect', 'Male']>0) & (Participant1['Subject-Suspect', 'Female']>0)
# Filtering Domestic Violence incidents
Domestic_indices = (incident['Domestic Violence']>0)

gun[Male_suspect]['n_killed'].sum(), gun[Female_suspect]['n_killed'].sum(), gun[Mix_suspect]['n_killed'].sum()
fig = {
    'data': [
        {
            'labels': ['Male suspect only', 'Female suspect only', 'Both Male as well as Female suspects', 'Suspect information unavailable'],
            'values': [gun[Male_suspect]['n_killed'].sum(), gun[Female_suspect]['n_killed'].sum(), gun[Mix_suspect]['n_killed'].sum(), gun[~(Male_suspect | Female_suspect | Mix_suspect)]['n_killed'].sum()],
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .45]},
        }
    ],
    'layout': {'title': 'Gender wise decomposition of the fatalities caused by suspects',
               'showlegend': True}
}
iplot(fig)


# Sadly 53.6% of the lives lost are due to crimes perpetuated by male suspects compared to 2.6% of crimes perpetrated by female suspects. Isn't this clear enough as to why a gender based perspective is needed into Gun Violence?
# 
# **Correction:** Since last time I have made a correction by also including instances where suspect information is unavailable. It is important to give complete information regarding this statistic. 

# In[21]:


male_incident = incident[Male_suspect].sum().sort_values(ascending=False)[[3, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18,21, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 50, 53]]/incident[Male_suspect].sum().sum()
fig = {
    'data': [
        {
            'labels': male_incident.index,
            'values': male_incident,
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .45]},
        }
    ],
    'layout': {'title': 'When suspect is male',
               'showlegend': False}
}
iplot(fig)


# Shooting crimes statistics indicate that crimes commited predominantly by male suspects are motivated by crime. But Domestic violence is a significant statistic.

# In[22]:


female_incident = incident[Female_suspect].sum().sort_values(ascending=False)[[3, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20]]

fig = {
    'data': [
        {
            'labels': female_incident.index,
            'values': female_incident,
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .35]},
        }
    ],
    'layout': {'title': 'When suspect is female',
               'showlegend': False}
}
iplot(fig)


# The trend is extremely strong that when the perpetrator is female the motive in majority of cases is due to Domestic Violence.

# In[25]:


trace1 = go.Bar(x=['Female Suspect','Male Suspect','Female Victim','Male Victim'], 
                y=Participant1[Male_suspect & Domestic_indices].sum().values,
               marker=dict(color=['rgba(28,45,204,1)', 'rgba(222,45,38,1)',
               'rgba(204,28,104,1)', 'rgba(24,204,20,1)']))
trace2 = go.Bar(x=['Female Suspect','Male Suspect','Female Victim','Male Victim'], 
                y=Participant1[Female_suspect & Domestic_indices].sum().values,
               marker=dict(color=['rgba(28,45,204,1)', 'rgba(222,45,38,1)',
               'rgba(204,28,104,1)', 'rgba(24,204,20,1)']), yaxis='y2')

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Male suspect only', 'Female suspect only'), shared_yaxes=True, print_grid=False)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

layout = go.Layout(
    xaxis=dict(domain=[0, 0.45], title='Considering only Male Suspects'), xaxis2=dict(domain=[0.5, 1], title='Considering only Female Suspects'),
    yaxis=dict(title='Number of Suspects and Victims')
)
fig = go.Figure(data=[trace1, trace2], layout=layout)
iplot(fig)


# As is evident here, there are a lot more statistics of gun violence related incidents where the suspect is male for Domestic Violence related incidents and in those instance we see the larger percentage of victims are women though we also see that in such incidents men are also victims. 
# 
# Though women suspect related incidents are fewer in number, in such cases men are clearly majority victims. 
# 

# 
# # Acknowledgements
# 
# In closing, I would like to mention that I learnt a lot from the following kernels mentioned in references. I believe so that using the information we have and drawing correct insights from it is far more important, many others have drawn important insights already and I hope to have added to the value of their findings. For now, we only have information on what transpired after the shooting incident. We might be able to get from different sources more information on indications that were missed prior to these shooting incidents. That will lead us more towards indications of how to prevent or control this spiralling crime. That is what we should work towards and hopefully that will lead to a safer society. This is the true desire of the owner/organizer of this dataset, me and many other people who seek a purpose out of this statistic. I welcome any feedback in this regard from anybody to improve upon the insights I was able to retrieve, any mistakes that might have occured on my behalf and any further improvements that can be done. 
# 
# Please feel free to build further using snippets of this code.
# 
# ## References:
# 1. https://www.kaggle.com/shivamb/stop-gun-violence-updated-exploration
# 2. https://www.kaggle.com/jpmiller/gun-violence-per-capita
#     

# In[ ]:




