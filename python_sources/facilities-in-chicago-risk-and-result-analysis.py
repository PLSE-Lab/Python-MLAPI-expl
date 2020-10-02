#!/usr/bin/env python
# coding: utf-8

# # Facilities in Chicago
# 
# In this notebook, I'll use the Chicago's inspection dataset to draw conclusions about the various facilities and the level of risk at which they are.

# ## Import libraries
# 
# Apart from the general libraries, I'll use `plotly` to plot maps.

# In[1]:


import numpy as np
import pandas as pd

import plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# We also need the `mapbox` access token for plotting. I've saved the key in a file which I'll import and use. You'll need to either create such a file for your own or just simply add a key to the notebook and use it.

# In[2]:


# Access token
from distutils.dir_util import copy_tree
copy_tree(src = "../input/tokens/", dst = "../working")

from access_tokens import *
mapbox_access_token = get_mapbox_token()


# ## Import dataset
# 
# I'll now import the dataset and place it in the `dataset` variable.

# In[3]:


dataset = pd.read_csv('../input/chicago-food-inspections/food-inspections.csv')
dataset.head(5)


# ## Data Analysis
# 
# Here, I'll take a look at the data and extract meaningful information for further visualization

# The first step is to identify all facilities and take the recent inspections for each facility. I'll also remove all rows where `'Risk', 'Facility Type', 'DBA Name', 'Latitude', 'Longitude'` will have null value. Some businesses are no longer operating or are no longer located and thus can be removed too. I'll create a new column `Name` which extracts the name from `AKA Name` and `DBA Name` with preference given to `AKA Name`.

# In[4]:


latest_data = dataset.sort_values('Inspection Date', ascending = False).groupby('License #').head(1)
latest_data.dropna(subset=['Risk', 'Facility Type', 'DBA Name', 'Latitude', 'Longitude'], axis = 0, how = 'all', inplace = True)
latest_data = latest_data[(latest_data['Results'] != 'Out of Business') & (latest_data['Results'] != 'Business Not Located')]
latest_data['Name'] = latest_data.apply(lambda row: row['AKA Name'] if not pd.isnull(row['AKA Name']) else row['DBA Name'], axis = 1)
latest_data['Name'] = latest_data['Name'] + '<br>' + latest_data['Address']


# I'll also create a `Risk Color` column which will help in plotting colors for each facility based on Risk.
# 1. All -> Black
# 2. High Risk -> Red
# 3. Medium Risk -> Yellow
# 4. Low Risk -> Green
# 
# For inspections, I'll crate the `Inspection Color` column.
# 1. Pass or Pass w/ Conditions -> Green
# 2. Fail or No Entry or Not Ready -> Red

# In[5]:


risk_color_map = { "All": "rgb(0, 0, 0)", "Risk 1 (High)": "rgb(255, 0, 0)", "Risk 2 (Medium)": "rgb(204, 204, 0)", "Risk 3 (Low)": "rgb(0, 100, 0)" }
latest_data['Risk Color'] = latest_data['Risk'].map(risk_color_map)

inspection_color_map = { 
    "Pass": "rgb(0, 255, 0)", 
    "Pass w/ Conditions": "rgb(0, 255, 0)",
    "Fail": "rgb(255, 0, 0)", 
    "No Entry": "rgb(255, 0, 0)", 
    "Not Ready": "rgb(255, 0, 0)" }
latest_data['Inspection Color'] = latest_data['Results'].map(inspection_color_map)
    
latest_data.reset_index(inplace=True)
print("Total businesses: {}".format(latest_data.shape[0]))


# ## Data Visualization
# 
# Next, I'll visualize the data to understand the data better and draw conclusions.

# ### Types of facilities
# 
# First, I'll extract all the different types of facilities and plot them as a pie chart. All facilities with total percentage less than 1% will be clubbed together as `Others`.

# In[ ]:


facility_types = latest_data['Facility Type'].value_counts().keys().tolist()
facility_count = latest_data['Facility Type'].value_counts().tolist()

final_types = []
final_count = []
others_count = 0
one_percent = 0.01 * latest_data.shape[0]
for count, facility_type in zip(facility_count, facility_types):
    if count > one_percent:
        final_types.append(facility_type)
        final_count.append(count)
    else:
        others_count += count
        
final_types.append('Others')
final_count.append(others_count)

# figure
fig = {
    "data": [{
        "values": final_count,
        "labels": final_types,
        "hoverinfo": "label+percent",
        "hole": .5,
        "type": "pie"
        },
    ],
    "layout": {
        "title": "Types of facilities",
        "width": 800,
        "height": 800
    }
}

iplot(fig)


# The majority facility types are **Restaurants** with approximately **55% of the total number of facilities**.

# ### Risk Analysis
# 
# I'll plot all facilities on the map of Chicago based on the colors we defined above.

# In[ ]:


data = [
    go.Scattermapbox(
        lat = latest_data['Latitude'],
        lon = latest_data['Longitude'],
        text = latest_data['Name'],
        hoverinfo = 'text',
        mode = 'markers',
        marker = go.scattermapbox.Marker(
            color = latest_data['Risk Color'],
            opacity = 0.7,
            size = 4
        )
    )
]

layout = go.Layout(
    mapbox = dict(
        accesstoken = mapbox_access_token,
        zoom = 10,
        center = dict(
            lat = 41.8781,
            lon = -87.6298
        ),
    ),
    height = 800,
    width = 800,
    title = "Facilities in Chicago")

fig = go.Figure(data, layout)
iplot(fig, filename = 'facilities')


# It appears that there are a lot of facilities with **High Risk**. We can also confirm the same using `value_counts`.

# In[ ]:


latest_data['Risk'].value_counts()


# It appears that indeed the maximum number of facilities have risk rating High.

# ### Success and Failure
# 
# I'll next take a look at the number of facilities that passed the inspection and the ones that did not.

# In[ ]:


data = [
    go.Bar(
        x = latest_data['Results'].value_counts().keys().tolist(),
        y = latest_data['Results'].value_counts().tolist(),
        marker = dict(
            color = [
                'rgb(0,100, 0)', 
                'rgb(0,100, 0)',
                'rgb(255, 0, 0)',
                'rgb(255, 0, 0)',
                'rgb(255, 0, 0)'
            ]
        )
    )
]

layout = go.Layout(
    title = 'Inspection Results',
)

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'inspections')


# We can consider `Pass` and `Pass w/ Conditions` to be positive outcome and the remaining as negative. Taking a look at the plot above, we can see that even though there are many facilities with high risk, most pass the inspection none the less. Let's plot these on a map.

# In[ ]:


data = [
    go.Scattermapbox(
        lat = latest_data['Latitude'],
        lon = latest_data['Longitude'],
        text = latest_data['Name'],
        hoverinfo = 'text',
        mode = 'markers',
        marker = go.scattermapbox.Marker(
            color = latest_data['Inspection Color'],
            opacity = 0.7,
            size = 4
        )
    )
]

layout = go.Layout(
    mapbox = dict(
        accesstoken = mapbox_access_token,
        zoom = 10,
        center = dict(
            lat = 41.8781,
            lon = -87.6298
        ),
    ),
    height = 800,
    width = 800,
    title = "Facilities in Chicago")

fig = go.Figure(data, layout)
iplot(fig, filename = 'facilities')


# The majority facilities passed their recent inspection.

# ### Ward-wise analysis
# 
# I'll now take a look at facilities based on their wards and compare how many passed and how many failed.

# In[ ]:


passed_inspections = latest_data[(latest_data['Results'] == 'Pass') | (latest_data['Results'] == 'Pass w/ Conditions')]
failed_inspections = latest_data[(latest_data['Results'] == 'Fail') | (latest_data['Results'] == 'No Entry') | (latest_data['Results'] == 'Not Ready')]

trace0 = go.Bar(
        x = passed_inspections.groupby('Wards').size().keys(),
        y = passed_inspections.groupby('Wards').size().tolist(),
        name = 'Passed inspections',
        marker = dict(
            color = 'rgb(55, 83, 109)'
        )
    )

trace1 = go.Bar(
        x = failed_inspections.groupby('Wards').size().keys(),
        y = failed_inspections.groupby('Wards').size().tolist(),
        name = 'Failed inspections',
        marker = dict(
            color = 'rgb(26, 118, 255)'
        )
    )

data = [trace0, trace1]
layout = go.Layout(
    title = 'Inspection Results',
)

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'ward-wise-inspections')


# It appears that ward 36 has the maximum ratio of passed to failed inspections.

# ### Majority violation
# 
# Let's also check the majority violation that is present among the selected dataset.

# In[290]:


import re
violators = latest_data.dropna(subset=['Violations'], axis = 0, how = 'all')
violations = violators.apply(lambda row: re.findall('\|\s([0-9]+)[.]', str(row['Violations'])), axis = 1)
first_violations = violators.apply(lambda row: row['Violations'].split('.')[0], axis = 1)

for violation, first_violation in zip(violations, first_violations):
    violation.append(first_violation)

flat_list = [item for sublist in violations for item in sublist]
unique, counts = np.unique(flat_list, return_counts=True)


# I'll select the violations that are more than 100 in count.

# In[291]:


violation = []
violation_count = []
for value, count in zip(unique, counts):
    if count > 100:
        violation.append(unique)
        violation_count.append(count)


# In[292]:


data = [
    go.Bar(
        x = violation,
        y = violation_count,
        marker = dict(
            color = 'rgb(55, 83, 109)'
        )
    )
]

layout = go.Layout(
    title = 'Majority Violations',
)

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'violations')


# Violation 41 is the majority violation which refers to **WIPING CLOTHS: PROPERLY USED & STORED**.

# ## Exploring nearby restaurants
# 
# Next, let's use the knowledge that we have, to identify nearby restaurants. I'll create a function that is customizable to get plots on the map based on risk level, current latitude, current longitude, and search distance.

# In[ ]:


from math import cos, asin, sqrt
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


# In[ ]:


def get_plot(dataset, curr_latitude = 41.8781, curr_longitude = -87.6298, risk_level = 'Low', search_distance = 5):
    dataset = dataset[dataset['Facility Type'] == 'Restaurant']
    
    if (risk_level == 'Low'):
        dataset = dataset[dataset['Risk'] == "Risk 3 (Low)"]
    elif (risk_level == 'Medium'):
        dataset = dataset[(dataset['Risk'] == "Risk 3 (Low)") | (dataset['Risk'] == "Risk 2 (Medium)")]
    elif (risk_level == 'High'):
        dataset = dataset[dataset['Risk'] != "All"]
    
    dataset = dataset[dataset.apply(lambda row: distance(curr_latitude, curr_longitude, row['Latitude'], row['Longitude']) < search_distance, axis = 1)]
    dataset.reset_index(inplace = True)
    
    data = [
        go.Scattermapbox(
            lat = dataset['Latitude'],
            lon = dataset['Longitude'],
            text = dataset['Name'],
            hoverinfo = 'text',
            mode = 'markers',
            marker = go.scattermapbox.Marker(
                color = dataset['Risk Color'],
                opacity = 0.7,
                size = 4
            )
        )
    ]

    layout = go.Layout(
        mapbox = dict(
            accesstoken = mapbox_access_token,
            zoom = 10,
            center = dict(
                lat = curr_latitude,
                lon = curr_longitude
            ),
        ),
        height = 800,
        width = 800,
        title = "Searched Restaurants in Chicago based on location and distance")

    fig = go.Figure(data, layout)
    iplot(fig, filename='restaurants')


# In[ ]:


get_plot(latest_data, 41.8781, -87.6298, 'Medium', 5)


# We can simply replace the parameter values and get the desired list of restaurants.

# ## Conclusion
# 
# In this notebook, I explored the Chicago food inspection dataset and used visualizations to draw really useful insights and plot map graphs.

# In[ ]:


# Removing token
from IPython.display import clear_output
clear_output(wait=True)
get_ipython().system('rm -rf ../working/access_tokens.py')


# In[ ]:




