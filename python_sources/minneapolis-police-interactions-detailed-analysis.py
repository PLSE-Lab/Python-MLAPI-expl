#!/usr/bin/env python
# coding: utf-8

# ### This kernel is meant for educational purposes and isn't intented to hurt anyone's sentiments. The objective of the kernel is to explore the dataset and to not objectify/degrade particular communities.

# # Minneapolis Police Interactions: A Detailed Analysis
# 
# Minneapolis is the largest city in the U.S. state of Minnesota and the principal city of the 16th-largest metropolitan area in the United States.
# 
# The dataset contains interactions of the Minneapolis Police Department.
# 
# Let us explore the dataset by importing the libraries. 

# In[ ]:


import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px


# Now, we'll load the .csv file into a DataFrame.

# In[ ]:


df = pd.read_csv('/kaggle/input/minneapolis-police-stops-and-police-violence/police_stop_data.csv', low_memory = False)
force_df = pd.read_csv('/kaggle/input/minneapolis-police-stops-and-police-violence/police_use_of_force.csv')


# In[ ]:


df.head()


# Let us inspect the DataFrame and check which columns have missing values.

# In[ ]:


df.info()


# # Distribution of Cases Per Year
# 
# Now we'll extract years from dates, which will help us in further plotting.
# 
# We'll append the counts of each year to a DataFrame.

# In[ ]:


year_values = []
for i in range(len(df)):
    date = df['responseDate'][i].split(" ")[0]
    year = date.split("/")[0]
    year_values.append(year)
    
year_counts = dict(Counter(year_values))
year_counts = {'year': list(year_counts.keys()), 'count': list(year_counts.values())}
years_df = pd.DataFrame(year_counts)
years_df


# Here is our first donut chart, which contains the distribution of cases recorded in each year:

# In[ ]:


fig_yearly = px.pie(years_df, values = 'count', names = 'year', title = 'Yearly Cases Distribution', hole = .5, color_discrete_sequence = px.colors.diverging.Portland)
fig_yearly.show()


# Most cases were recorded in 2017 and 2018.

# We'll do the same pre-processing for other variables too:

# # Distribution of Case Types
# Let us see the distribution of cases on the basis of the problem:

# In[ ]:


problem_counts_dict = dict(Counter(df['problem']))
problem_df_dict = {'problem': list(problem_counts_dict.keys()), 'count': list(problem_counts_dict.values())}

problem_df = pd.DataFrame(problem_df_dict)
problem_df


# In[ ]:


fig_yearly = px.pie(problem_df, values = 'count', names = 'problem', title = 'Type of Cases', hole = .5, color_discrete_sequence = px.colors.sequential.Agsunset)
fig_yearly.show()


# Hence, most people were caught violating traffic laws or were displaying suspicious activity.

# # Interactive Maps
# 
# Now, we'll be using an interactive map to see at which locations were the cases recorded:

# In[ ]:


import folium
from folium.plugins import FastMarkerCluster
locations = df[['lat', 'long']]
locationlist = locations.values.tolist()


# In[ ]:


map = folium.Map(location=[44.986656, -93.258133], zoom_start=12)
FastMarkerCluster(data=list(zip(df['lat'].values, df['long'].values))).add_to(map)
map


# **This map is interactive. Click on the orange clusters to see more cases in the neighborhood.**
# 
# **Each cluster indicates the collective amount of cases in the surrounding areas highlighted in blue (visible on hover).**

# # Distribution of Races

# In[ ]:


df['race'].fillna('No Data', inplace = True)
race_counts_dict = dict(Counter(df['race']))

race_counts_dict['Unknown'] += race_counts_dict['No Data']
del race_counts_dict['No Data']

race_df_dict = {'race': list(race_counts_dict.keys()), 'count': list(race_counts_dict.values())}

race_df = pd.DataFrame(race_df_dict)
race_df


# In[ ]:


fig_race = px.pie(race_df, values = 'count', names = 'race', title = 'Distribution of Races', hole = .5, color_discrete_sequence = px.colors.diverging.Temps)
fig_race.show()


# Let us now use the second dataset:

# In[ ]:


force_new = force_df[['ForceType', 'EventAge', 'TypeOfResistance', 'Is911Call']]
force_new.head()


# # Forces Used by the Police
# 
# Let us now see the various types of forces used by the police in incidents:

# In[ ]:


force_counts_dict = dict(Counter(force_new['ForceType']))

force_counts_dict['Unknown'] = force_counts_dict[np.nan]
del force_counts_dict[np.nan]

force_df_dict = {'force': list(force_counts_dict.keys()), 'count': list(force_counts_dict.values())}

force_type_df = pd.DataFrame(force_df_dict)
force_type_df


# In[ ]:


fig_force = px.bar(force_type_df, x = 'force', y = 'count')
fig_force.show()


# We can see that most people were arrested with the help of bodily force or some chemical irritant. Tasers were also used in forceful arrest.

# # Distribution of Ages of People Involved
# 
# We can see that people between the 20-40 were arrested. 

# In[ ]:


fig_age_hist = px.histogram(force_new, x = 'EventAge', nbins=10, opacity = 0.7)
fig_age_hist.show()


# # Distribution of types of Resistance
# 
# The DataFrame contains several values of the same value in different formats.
# 
# For example: There are several rows with 'Assualting Police Horse' as the value which is similar to 'Assualted Police Horse'. We need to merge these values together.
# 
# This is a serious issue as the same type of resistance is classified into different bins. For ideal plotting, we'll process the values and add into several bins.
# 
# Here is the dataframe after pre-processing:

# In[ ]:


force_df['TypeOfResistance'].fillna('Unknown', inplace = True)
cleaned_types = []
for item in force_df['TypeOfResistance']:
    p1_item = item.strip()
    p2_item = p1_item.title()
    cleaned_types.append(p2_item)
    
force_df['TypeNew'] = cleaned_types

resistance_counts_dict = dict(Counter(force_df['TypeNew']))

resistance_counts_dict['Unspecified'] += resistance_counts_dict['Unknown']
del resistance_counts_dict['Unknown']

resistance_counts_dict['Commission Of Crime'] += resistance_counts_dict['Commission Of A Crime']
del resistance_counts_dict['Commission Of A Crime']

resistance_counts_dict['Fled In Vehicle'] += resistance_counts_dict['Fled In A Vehicle']
del resistance_counts_dict['Fled In A Vehicle']

resistance_counts_dict['Assaulting Police Horse'] += resistance_counts_dict['Assaulted Police Horse']
del resistance_counts_dict['Assaulted Police Horse']

resistance_counts_df_dict = {'type': list(resistance_counts_dict.keys()), 'count': list(resistance_counts_dict.values())}

resistance_df = pd.DataFrame(resistance_counts_df_dict)
resistance_df


# In[ ]:


fig_resistance = px.pie(resistance_df, values = 'count', names = 'type', title = 'Distribution of Resistance', hole = .5, color_discrete_sequence = px.colors.diverging.Picnic)
fig_resistance.show()


# # How many people called 911?

# In[ ]:


_911_counts_dict = dict(Counter(force_new['Is911Call']))

_911_counts_dict['Unspecified'] = _911_counts_dict[np.nan]
del _911_counts_dict[np.nan]

_911_df_dict = {'val': list(_911_counts_dict.keys()), 'count': list(_911_counts_dict.values())}

_911_df = pd.DataFrame(_911_df_dict)
_911_df


# In[ ]:


fig_911 = px.pie(_911_df, values = 'count', names = 'val', title = 'Distribution of 911 Calls', hole = .5, color_discrete_sequence = ['#ff4757', '#10ac84', '#2f3542'])
fig_911.show()


# ## Feel free to  give suggestions and upvote this kernel if you loved it!
