#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.express as px


# In[ ]:


data = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')


# In[ ]:


data


# Looking at this small snippet, we can already see some peculiarities. I have made a separate section at the end for this.
# 
# First, Let's answer the most obvious question - is the police racist.

# # Are the Police Racist? - The Data Can't Say for Sure

# In[ ]:


a = data['race'].value_counts().reset_index()
fig = px.pie(a, values='race', names='index', title='Race of Victim')
fig.show()


# The percentage of African Americans (labeled B) in America is 13.4%. So if the police shot randomly, about 13.4% should be African Americans. Here we see that this number is 26.5%, which is disproportionately large!
# 
# Let's look at Hispanics - about 18.3% of the population is Hispanic. This is roughly what we see here, so there doesn't seem to be a bias in this direction.
# 
# To complete this claim, let's look at the threat level as well.

# In[ ]:


counts = []
counts.append(data[data['threat_level'] == 'attack'].groupby('race').count()['id'].values)
counts.append(data[data['threat_level'] == 'undetermined'].groupby('race').count()['id'].values)
counts.append(data[data['threat_level'] == 'other'].groupby('race').count()['id'].values)
race = ['A', 'B', 'H', 'N', 'O', 'W']

fig = go.Figure(data=[
    go.Bar(name='Attack', x=race, y=counts[0]),
    go.Bar(name='Undetermined', x=race, y=counts[1]),
    go.Bar(name='Other', x=race, y=counts[2])
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# The ratio of Attack to others for whites is 2.207, and for African Americans is 2.05.
# This means that African Americans are not more dangerous than Whites, yet the percentage of African Americans killed was higher as compared to population.
# 
# Unfortunately, this shows signs that the police MIGHT be racist.
# 
# 
# Let's do the same for weapons, and see if anything odd pops up there, which may prove the above statement wrong.
# 

# In[ ]:


data['armed'].unique()[0:6]


# In[ ]:


graphs = []
race = ['A', 'B', 'H', 'N', 'O', 'W']


for weapon in data['armed'].unique()[0:6]:
    graphs.append(go.Bar(name=weapon, x=race, y=data[data['armed'] == weapon].groupby('race').count()['id'].values))


fig = go.Figure(data=graphs)
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# This graph also doesn't suggest anything out of the ordinary, such as one race having a much higher chance of having a weapon as compared to another.
# 
# Now let's also do it for the last category on fleeing. If we see no signs here, then it is possible that the police is racist.

# In[ ]:


graphs = []
race = ['A', 'B', 'H', 'N', 'O', 'W']


for weapon in data['flee'].unique():
    graphs.append(go.Bar(name=weapon, x=race, y=data[data['flee'] == weapon].groupby('race').count()['id'].values))


fig = go.Figure(data=graphs)
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# The ratio of people not fleeing to fleeing for African Americans is roughly 1.59. For Whites, it is 2.2. This means that African Americans were more likely to flee upon encounters, maybe because they are more scared.
# 
# This might be the reason for the higher shooting rate, as during a flee situation, there might be some misfiring on the police's part.
# 
# Hence, we can't be very confident as to whether the police are racist (we could have been if this plot hadn't shown anything peculiar).
# 
# Let's also see if there was anythign Peculiar with the Mental Illness Datapoint

# In[ ]:


graphs = []
race = ['A', 'B', 'H', 'N', 'O', 'W']

#Plotly doesn't accept boolean values
data['signs_of_mental_illness'] = data['signs_of_mental_illness'].astype('object')

for weapon in data['signs_of_mental_illness'].unique():
    graphs.append(go.Bar(name=weapon, x=race, y=data[data['signs_of_mental_illness'] == weapon].groupby('race').count()['id'].values))


fig = go.Figure(data=graphs)
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# Looking at the ratios again (of True to False), we see that for African Americans, it is 0.165, whereas for Whites, it is 0.4. This shows us the opposite, that Whites that were shot were more likely to have been mentally ill. This also doesn't account for the increased percentage of African American deaths, but we still can't be certain because the Fleeing datapoint showed the same increased percentage.
# 

# # How old are the Victims?
# Now let's look at the Distribution of the age of victims

# In[ ]:


trace1 = go.Violin(

        x = data['age'],
    text = 'Age'
)
layout = dict(title = 'Distribution of Age')

iplot(go.Figure(data=trace1, layout=layout))


# Surprisingly, the range is a lot thicker than I thought. I was expecting the plot to be thickest at around mid 20s, and then narrow down quickly, but it turns out that the age range of victims is quite large.
# 
# # Which State has The Most Shootings?
# 
# Let's also look at which state has the most shootings

# In[ ]:


a = data['state'].value_counts().reset_index()
fig = px.pie(a, values='state', names='index', title='State Wise Shooting Count')
fig.show()


# This graph has to be interpreted Carefully. We cannot just take it at the face value and say that California has the most shootings. We should also consider population.
# 
# Roughly 12% of the population lives in California, so 14% is a bit high. On the other hand, even though New York and New Jersey have a similar percentage of shootings, they have a 1% difference in population. So New Jersey actually has much less shootings as compared to New York.
# 
# To make this graph more readable, you could try to include the population, but I couldn't find any organized data, and just googled it for a few places.

# # A Peculiarity in the Data

# Let's see the snippet again

# In[ ]:


data


# First thing that is obvious is the name. This is most likely the name of the victim, but the name TK TK repeats quite a lot. In fact, let's check out how many are of this name.

# In[ ]:


data[data['name'] == 'TK TK']


# These are not repeat rows since the dates, manner of death, and every data point is different.
# 
# Since this is not clarified in the data explanation, my guess would be that this is just a fill value inplace of Unknown Name.
# 
# Let's try to confirm this by showing a few rows where age is unkown.

# In[ ]:


data[data['age'].isnull()]


# We can see that a lot of the rows have the name TK TK, so it is quite likely that this is a placeholder for NaN.
