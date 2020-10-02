#!/usr/bin/env python
# coding: utf-8

# # Olympic Dataset

# In[ ]:


import pandas as pd
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


events = pd.read_csv('../input/athlete_events.csv')
events.head(2)


# In[ ]:


noc = pd.read_csv('../input/noc_regions.csv')
noc.head(2)


# ### Lets combine the dataframes

# In[ ]:


data = pd.merge(events, noc, on='NOC', how='left')
data.head(1)


# ### 205 countries have participated in the Olympics

# In[ ]:


region = data['region'].value_counts()
len(region)


# In[ ]:


region = region.sort_values()[::-1][:50]


# In[ ]:


plotly.offline.iplot({
    "data": [go.Bar(x=region.values, y=region.index, orientation = 'h')],
    "layout": go.Layout(title="Number of participants per country"
                           ,width=1000,height=1000,)
})


# ### USA has sent the most number of athletes to the olympics followed by Germany 

# In[ ]:


season = data['Season'].value_counts()


# In[ ]:


plotly.offline.iplot({
    "data": [go.Pie(labels=season.index, values=season.values)],
})


# In[ ]:


sport = data['Sport'].value_counts()
sport = sport.sort_values()[::-1][:40]
plotly.offline.iplot({
    "data": [go.Bar(x=sport.values, y=sport.index, orientation = 'h', marker = dict(color = 'rgba(222,45,38,0.8)'))],
    "layout": go.Layout(title="Sports with most events"
                           ,width=700,height=800,margin=go.Margin(
        l=300,
        r=100,
        b=100,
        t=100,
        pad=4
    ),)
})


# ## India's Medal Count and distribution

# In[ ]:


India = data[(data['region']=='India')]
medals = India['Medal'].value_counts()
medals


# In[ ]:


plotly.offline.iplot({
    "data": [go.Pie(labels=medals.index, values=medals.values)],
    
})


# In[ ]:


India['Gold'] = India['Medal'].map({'Gold': 1, 'Bronze': 0,'Silver':0})
India['Silver'] = India['Medal'].map({'Gold': 0, 'Bronze': 0,'Silver':1})
India['Bronze'] = India['Medal'].map({'Gold': 0, 'Bronze': 1,'Silver':0})
total_medals =India.groupby(['Year']).sum()


# In[ ]:


total_medals= total_medals.fillna(0)
trace0 = go.Scatter(
    x = total_medals.index,
    y = total_medals['Gold'],
    mode = 'lines',
    name = 'GOLD'
)
trace1 = go.Scatter(
    x = total_medals.index,
    y = total_medals['Silver'],
    mode = 'lines',
    name = 'SILVER'
)
trace2 = go.Scatter(
    x = total_medals.index,
    y = total_medals['Bronze'],
    mode = 'lines',
    name = 'BRONZE'
)

data = [trace0, trace1, trace2]

plotly.offline.iplot({
    "data": data
})


# ### The distribution is very lean, let's try USA

# In[ ]:


data = pd.merge(events, noc, on='NOC', how='left')
USA = data[(data['region']=='USA')]
medals = USA['Medal'].value_counts()
print(medals)
USA['Gold'] = USA['Medal'].map({'Gold': 1, 'Bronze': 0,'Silver':0})
USA['Silver'] = USA['Medal'].map({'Gold': 0, 'Bronze': 0,'Silver':1})
USA['Bronze'] = USA['Medal'].map({'Gold': 0, 'Bronze': 1,'Silver':0})
total_medals =USA.groupby(['Year']).sum()


# In[ ]:


total_medals= total_medals.fillna(0)
trace0 = go.Scatter(
    x = total_medals.index,
    y = total_medals['Gold'],
    mode = 'lines',
    name = 'GOLD'
)
trace1 = go.Scatter(
    x = total_medals.index,
    y = total_medals['Silver'],
    mode = 'lines',
    name = 'SILVER'
)
trace2 = go.Scatter(
    x = total_medals.index,
    y = total_medals['Bronze'],
    mode = 'lines',
    name = 'BRONZE'
)

data = [trace0, trace1, trace2]

plotly.offline.iplot({
    "data": data
})


# In[ ]:


data = pd.merge(events, noc, on='NOC', how='left')
mean = data.groupby('region').mean()
sort = mean.sort_values('Height')[::-1]


# In[ ]:


sort = sort.head(50)


# In[ ]:


trace1 = go.Bar(
    x= sort.index,
    y=sort['Weight'],
    name='Average Weight'
)
trace2 = go.Bar(
    x= sort.index,
    y= sort['Height'],
    name='Average Height'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)


plotly.offline.iplot({
    "data": data
})


# ### Almost all athletes lie in the same height, weight range

# In[ ]:


data = pd.merge(events, noc, on='NOC', how='left')
mean = data.groupby('region').mean()
sort = mean.sort_values('Height')[::-1]
sort.tail(1)


# - Micronesia has the Shortest Atheletes
# - Croatia has the tallest athletes

# In[ ]:


sort.sort_values('Weight').head(1)


# In[ ]:


sort.sort_values('Weight').tail(1)


# - Nauru has the heaviest athletes
# - Maldives has the lightest athletes
# 

# In[ ]:


sort.sort_values('Age')['Age'].mean()


# ## Average Age of Olympic atheltes is about 25

# In[ ]:


data.head(1)


# In[ ]:


data['region'] = data['region'].fillna(data['Team'])
data['Medal'] = data['Medal'].fillna(0)
data['Medal'] = data['Medal'].map({'Gold': 1, 'Bronze': 1,'Silver':1})


# In[ ]:


almedal = data.groupby('region').sum()
medal =almedal[(almedal['Medal'] != 0)]
nomedals = almedal[(almedal['Medal'] == 0)].index


# In[ ]:


medal['COUNTRY'] = medal.index
medal['Medal']=medal['Medal'].map(int)


# In[ ]:


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
d = pd.merge(medal,df, on='COUNTRY', how='left')
d.head(1)


# In[ ]:


d[(d['CODE'].isnull())]


# In[ ]:


d['CODE'] =  d['CODE'].fillna(d['COUNTRY'])


# In[ ]:


d = d.replace('Bahamas', 'BHM')
d = d.replace('Ivory Coast', 'CIV')
d = d.replace('North Korea', 'PRK')
d = d.replace('South Korea', 'KOR')
d = d.replace('Trinidad', 'BRB')
d = d.replace('UK', 'GBR')
d = d.replace('Virgin Islands, US', 'VGB')


# In[ ]:


import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)

data = dict(type='choropleth',
locations = d['CODE'],z = d['Medal'],
text = d['COUNTRY'], colorbar = {'title':'No of medals'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
reversescale = False)


# In[ ]:


layout = dict(title='Medal count with respect to each country',
geo = dict(showframe = True, projection={'type':'mercator'}))


# In[ ]:


choromap = go.Figure(data = [data], layout = layout)
iplot(choromap, validate=False)


# ### The Map does not Highlight the countries with no medals

# In[ ]:


print("#### COUNTRIES WITH NO OLYMPIC MEDALS ###\n")
for i in nomedals:
    print(i)


# # THANK YOU

# In[ ]:





# In[ ]:





# In[ ]:




