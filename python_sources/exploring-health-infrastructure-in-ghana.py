#!/usr/bin/env python
# coding: utf-8

# # Exploring health infrastructure in Ghana
# 
# In this notebook, I take a look at the various health facilities in Ghana and how they are distributed across the country.

# # Description and Objective
# 
# The dataset includes information about various facilities such as their name, their location, the type of facility and the owners. The second file also includes the tier under which the facilities fall.
# 
# In this notebook, I'll explore the following in the dataset:
# 1. Most common facility type in each district, town, and region
# 2. Most common facility type across the country
# 3. Who owns the maximum number of facilities?
# 4. How are the facilities spread across the country based on their region?
# 5. Plot the tiers for each facility on map

# # Import libraries and dataset
# 
# I'll import the libraries that I'll use to draw conclusions from the data. I'll be using `Matplotlib` and `Seaborn` for plots. For maps, I'll use `Plotly`.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


facilities = pd.read_csv('../input/health-facilities-gh/health-facilities-gh.csv')
tiers = pd.read_csv('../input/health-facilities-gh/health-facility-tiers.csv')


# In[ ]:


print("Facilities: {}\n".format(facilities.info()))
print("Facilities and tiers: {}".format(tiers.info()))


# The facilities have some missing `Town` values but all other information is present completely.
# 

# In[ ]:


print(facilities.head())
print('-' * 60)
print(tiers.head())


# In the two datasets, the facility names are referred differently, `FacilityName` and `Facility` respectively. Also, the case of the two columns is different so on combining, I'll have to align them in the same case.

# # Most common facility type in each district, town and region
# 
# I'll group the information by `district` and then see which is the common health facility type in each.

# In[ ]:


# Most common health facility type in each district
pd.DataFrame(facilities.groupby(['District', 'Type']).size().unstack().idxmax(axis = 1))


# In[ ]:


# Most common health facility type in each town
pd.DataFrame(facilities.groupby(['Town', 'Type']).size().unstack().idxmax(axis = 1))


# In[ ]:


# Most common health facility type in each region
pd.DataFrame(facilities.groupby(['Region', 'Type']).size().unstack().idxmax(axis = 1))


# # Most common facility type across the country
# 
# Let's now explore all the facility types that exist across the country.

# In[ ]:


facility_types = facilities['Type'].value_counts()
plt.figure(figsize = (20, 12))
sns.barplot(x = facility_types.index, y = facility_types.values)
plt.xticks(rotation = 90)
plt.xlabel("Health Facility Type", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.title("Count of each health facility type", fontsize = 16)


# The **most common facility type** is a **Clinic with over 1000+ centres**.

# # Who owns the maximum number of facilities?

# In[ ]:


ownership_counts = facilities['Ownership'].str.capitalize().value_counts()
others = ownership_counts[4:].sum()
ownership_counts = ownership_counts[:4]
ownership_counts['Others'] = others
explode = [0.05]*len(ownership_counts)
plt.figure(figsize = (12, 8))
plt.pie(ownership_counts.values, labels = ownership_counts.index, explode = explode, autopct='%1.1f%%')
plt.title("Ownership distribution of health facilities", fontsize = 16)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


# As is evident from the pie chart above, **Government of Ghana owns the maximum number of health facilities**.

# # How are the facilities spread across the country based on their region?
# 
# Let's now plot the health facilities on the map of Ghana and see how they are distributed.

# In[ ]:


max_long = facilities['Longitude'].max()
min_long = facilities['Longitude'].min()
max_lat = facilities['Latitude'].max()
min_lat = facilities['Latitude'].min()


# In[ ]:


data = []
for index, region in enumerate(facilities['Region'].unique()):
    selected_facilities = facilities[facilities['Region'] == region]
    data.append(
        go.Scattergeo(
        lon = selected_facilities['Longitude'],
        lat = selected_facilities['Latitude'],
        text = selected_facilities['FacilityName'],
        mode = 'markers',
        marker_color = index,
        name = region
        )
    )

layout = dict(
        title = 'Health facilities in Ghana based on Region',
        geo = dict(
        scope = 'africa',
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ min_long - 5, max_long + 5 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [ min_lat - 1, max_lat + 1 ],
            dtick = 5
        )
    )
)
fig = dict(data = data, layout = layout)
go.Figure(fig)


# As is evident from the map above, health facilities are distributed across the whole country. However, **the facilities are sporadic in the middle while more concentrated near the borders.**

# # Plot the tiers for each facility on map
# 
# I'll now combine the two datasets together based on the Facility Name by changing them to lowercase and then plot the common information of tiers on a map.
# 

# In[ ]:


facilities['FacilityName'] = facilities['FacilityName'].str.lower()
tiers['Facility'] = tiers['Facility'].str.lower()
combined = pd.merge(facilities, tiers, left_on=['FacilityName'], right_on=['Facility'])


# In[ ]:


data = []
for index, tier in enumerate(combined['Tier'].unique()):
    selected_facilities = combined[combined['Tier'] == tier]
    data.append(
        go.Scattergeo(
        lon = selected_facilities['Longitude'],
        lat = selected_facilities['Latitude'],
        text = selected_facilities['FacilityName'],
        mode = 'markers',
        marker_color = index,
        name = "Tier " + str(tier)
        )
    )

layout = dict(
        title = 'Health facilities in Ghana based on Tier',
        geo = dict(
        scope = 'africa',
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ min_long - 5, max_long + 5 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [ min_lat - 1, max_lat + 1 ],
            dtick = 5
        )
    )
)
fig = dict(data = data, layout = layout)
go.Figure(fig)


# Clearly, the overlapping between the two datasets is very less. On combining the two datasets, it appears that the **Tier 3 facilities** are spread across the country but **Tier 2 facilities** are only located in the lower region of the country.

# # Conclusion
# 
# I began with the aim to answer a few questions. After exploring the data, I've reached to the following conclusions:
# - Most common facility type in each district, town, and region varies where for some it is a `Health Centre` while for others it is `CHPS`
# - Clinic is the most common facility type across the country
# - The government owns the maximum number of facilities?
# - Facilities are spread all acorss the country which major concentrations across the borders
# - Tier 3 facilities are across the country while tier 2 facilities are in the lower part of the country
