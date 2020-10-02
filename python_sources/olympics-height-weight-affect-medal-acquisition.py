#!/usr/bin/env python
# coding: utf-8

# I will be exploring if height and weight has affected medal acquistion for athletes.

# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input/"))

athletes = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
scores = pd.read_csv("../input/divingscores2016/swimscores2016.csv")


# In[ ]:


athletes.info()


# Looks like the restricting factor in this dataset will be weight. Going to only work with the fields that contain both height and weight.

# In[ ]:


weight = {'Weight'}
improve = athletes.dropna(subset=weight)
#dropping rows which don't have weight values


# In[ ]:


improve['Height'].isnull().describe()
#looks like about 2000 of these entries don't have height values
height = {'Height'}
improve = improve.dropna(subset=height)


# In[ ]:


improve.info()
#now we have our data


# In[ ]:


#We can now select an event to look into first. I'm going to look at Olympic Synchronized Divers first.
#Have to look through the list of events, there are 590:
len(improve['Event'].unique())
improve['Event'].unique()


# Two types of syncronized diving events, so I will look at the Springboard event. Events name is: "Diving Women's Synchronized Springboard"

# In[ ]:


divingWS = improve.loc[improve['Event'] == "Diving Women's Synchronized Springboard"]
divingWS = divingWS.sort_values(by=['Year'])


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set_style('darkgrid')
k = sns.lmplot(data=divingWS,x='Year', y='Weight', hue='Medal', fit_reg=False)


# In[ ]:


k = sns.lmplot(data=divingWS,x='Year', y='Height', hue='Medal', fit_reg=False)


# In[ ]:


#The heights and especially the weights are quite close for the Gold winners.
#Going to add in the non-medalists to see how they differ
wdiving = divingWS
wdiving['Medal'] = wdiving['Medal'].fillna('Non-Medalists')
#filling the nulls of the Medal's column
k = sns.lmplot(data=divingWS,x='Year', y='Weight', hue='Medal', fit_reg=False)
k = sns.lmplot(data=divingWS,x='Year', y='Height', hue='Medal', fit_reg=False)


# In[ ]:


k = sns.lmplot(data=wdiving,x='Year', y='Height', hue = 'Team',markers=['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', '|', 'H','>','<',','], fit_reg=False )
#markers=['o','x','^','+']


# In[ ]:


#We can see that there are many countries, hard to get any information out of this
#focus it down to a specific year: 2016
k = sns.lmplot(data=wdiving[wdiving['Year'] == 2016],x='Team', y='Height', hue = 'Medal', fit_reg=False )
k = sns.lmplot(data=wdiving[wdiving['Year'] == 2016],x='Team', y='Weight', hue = 'Medal', fit_reg=False )
#We can see that the heights are always quite close, usually a difference of a couple centimeters
#Weights can vary, I don't believe that there is a huge effect from either.


# In[ ]:


g = sns.FacetGrid(data=wdiving,col = 'Year', row = 'Medal', hue = 'Team' )
g = g.map(plt.scatter,'Height', 'Weight' )
plt.legend()


# For Women's Springboard, the data only goes through the five most recent olympics. The Gold medalists for the last 4 olympics have been the Chinese team, Wu Minxia being a part of it for every year since 2004. Impressive. I would like to see if the height/weight relationship between the two divers does, but the fact that the gold medalists are mostly the same team 4 years running would skew toward suggesting that if you want Gold, be Wu Minxia. 
# 
# Looking further into this, the wikipedia page for Olympic Diving shows the scores of the each of the diving teams. The Chinese team consistently hit 80 points on the last dive. The margin of their wins is always quite large. 
# 
# https://en.wikipedia.org/wiki/Diving_at_the_2016_Summer_Olympics_%E2%80%93_Women%27s_synchronized_3_metre_springboard

# In[ ]:


team = wdiving[wdiving['Year'] == 2016]
heightweightteam = ['Height', 'Weight', 'Team']
team = team[heightweightteam]
team.reset_index()
team


# In[ ]:


team = team.sort_values('Team').reset_index()
team = team.drop('index', axis=1)


# In[ ]:





# In[ ]:



team['diffheight'] = abs(team['Height'].diff())
team['diffweight'] = abs(team['Weight'].diff())
team['distance'] = (team['diffheight'])**2 + (team['diffweight'])**2
team['distance'] = np.sqrt(team['distance'])
team = team[1::2]


# In[ ]:


team = team.drop(['Height', 'Weight'], axis=1)


# In[ ]:


team.columns = ['team','diffheight', 'diffweight','distance']
team


# At this point, I don't see any indication of correlation between the height/weights of the two divers in comparison to their medal standing. However, I would be curious if their actual diving scores in the competition are correlated. 
# 
# One of the reasons to check the scores is because each set of divers does 5 dives. There are certain required techniques that have to be completed at certain difficulty levels. So, I would be interested to see if the height/weight has an effect on the individual dives that comprise the final scores.
# 
# So, I scraped the 2016 diving scores off of wikipedia, couldn't find a direct source for the scores (the referenced website wasn't working for me).  The score chart for 5 dives is shown below.

# In[ ]:


import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

trace1 = go.Bar(
    x=scores.team,
    y=scores.dive1,
    name='Dive 1'
)
trace2 = go.Bar(
    x=scores.team,
    y=scores.dive2,
    name='Dive 2'
)
trace3 = go.Bar(
    x=scores.team,
    y=scores.dive3,
    name='Dive 3'
)
trace4 = go.Bar(
    x=scores.team,
    y=scores.dive4,
    name='Dive 4'
)
trace5 = go.Bar(
    x=scores.team,
    y=scores.dive5,
    name='Dive 5'
)

data = [trace1, trace2,trace3,trace4,trace5]
layout = go.Layout(
    barmode='group'
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# In[ ]:


mix = scores.merge(team, how='left',on='team')
mix


# In[ ]:


mix.columns = ['scrap','team','dive1','dive2','dive3','dive4','dive5','total','distance','diffheight','diffweight']
mix = mix.drop('scrap', axis=1)


# In[ ]:


corr = mix.corr()


# In[ ]:


mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Looks like there is a small correlation between height difference and dive 2. The 3 teams with a higher than 1cm height difference scored in the top 3 for that particular dive. From what I've researched on the particular dives, there are certain diving approaches that are to be completed. 
# 
# *A hypothesis might be that there could be an effect on how it looks from the judging stand. A perspective effect of 8cm(a little more than 3inches) could make the divers look like are the same height and feel more synced up from the judges point of view*
# 
# To really be sure that there isn't any sort of effect of the divers height on the scoring, I would have to check all the scores from multiple competitions to see if there is an actual correlation for the similarity in height of the divers. 
# 
# Reference for how diving is scored in the Olympics: https://www.britannica.com/story/how-is-diving-scored
