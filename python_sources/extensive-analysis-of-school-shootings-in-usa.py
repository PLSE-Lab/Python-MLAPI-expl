#!/usr/bin/env python
# coding: utf-8

#  # Extensive exploratory analysis of Various School Shootings across the United States of America 
# 
# 
# 

# <img src='http://1.bp.blogspot.com/-dCWwvElzjqA/UgOnLNDBgYI/AAAAAAAAApY/WvZv586ju-M/s640/Dont+Shoot.jpg'>

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import re


# In[3]:


data= pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
data.head()


# ### Data is given for 51 states

# In[4]:


states = list(data['state'].unique())
len(states)


# In[5]:


killed=[]
injured=[]
for i in states:
    s = data[(data['state']== i)]
    k = sum(s['n_killed'])
    killed.append(k)
    i = sum(s['n_injured'])
    injured.append(i)


# ### Killed vs injured

# In[9]:


import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Bar(
            x=states,
            y=killed,
            name = 'Killed'
    )
trace2 = go.Bar(
            x=states,
            y=injured,
            name = 'Injured'
    )

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

iplot(go.Figure(data=data, layout=layout))


# ### California has more number of deaths related to school shootings whereas Illinois has more injuries

# In[10]:


#TOTAL CASUALITIES IS NUMBER OF DEATHS + NUMBER OF INJURIES
total =[x + y for x, y in zip(killed, injured)]


# In[12]:


data = [go.Bar(x=states,y=total)]
iplot(go.Figure(data=data))


#    <h2 align = "center">       The state of Illinois has suffered the most from the school shootings

# <img src ='https://i2.cdn.turner.com/money/dam/assets/160203113813-welcome-to-illinois-sign-780x439.jpg'>

# In[14]:


data= pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')


# In[15]:


i = data.loc[data['state'] == 'Illinois',:]
city = list(i['city_or_county'].unique())


# ### Let us analyze the state of illinois(Land of Lincoln) by looking at its cities and counties

# In[16]:


illinois_killed = {}
illinois_injured = {}
killed=[]
injured=[]
for j in city:
    s = i[(i['city_or_county']== j)]
    k = sum(s['n_killed'])
    illinois_killed[j] = k
    kk = sum(s['n_injured'])
    illinois_injured[j] = kk


# ### Remove the cities or counties with less than 10 deaths 

# In[17]:


illinois_killed.values()
i_k = {x:y for x,y in illinois_killed.items() if y>10}


# In[18]:


data = [go.Bar(x=list(i_k.keys()),
            y=list(i_k.values()) ,
            marker=dict(color='#cc33ff'))]

iplot(go.Figure(data=data))


# ### Chicago is the most dangerous city for school going children in the USA

# In[19]:


data= pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')


# ### When considering all cities

# In[20]:


city = data['city_or_county'].value_counts()
city = city[:25]


# In[21]:


d = [go.Bar(x=list(city.index),y=list(city.values),marker=dict(color='#ffff4d'))]
iplot(go.Figure(data=d))


# ### The school shootings in chicago seems to be very much higher than all other cities in the United States
# 

# <img src='http://s14544.pcdn.co/wp-content/uploads/2017/01/Chicago-Crime-1-660x496.jpg'>

# ### Lets analyze the guns used in these shootings

# In[22]:


guns = data['gun_type']
guns = guns.dropna()
guns = [x for x in guns if x != '0::Unknown' and x!='0:Unknown']


# In[23]:


allguns=[]
for i in guns:
    result = re.sub("\d+::", "", i)
    result = re.sub("\d+:", "", result)
    result = result.split("|")
    for j in result:
        allguns.append(j)


# In[24]:


allguns = [x for x in allguns if x != 'Unknown']
allguns = [x for x in allguns if x]


# In[25]:


from collections import Counter
allguns = Counter(allguns)
labels, values = zip(*allguns.items())


# ### Guns used in the shootings

# In[26]:


d = [go.Bar(x=list(labels),y=list(values),marker=dict(color='#ff0055'))]
iplot(go.Figure(data=d))


# ### Handguns are the most used guns followed by 9mm and rifles

# In[27]:


data= pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')


# In[28]:


d = data.groupby('date').sum()


# In[29]:


d.head()


# In[1]:


data = [go.Scatter(
          x=d.index,
          y=d['n_killed'])]

killings = go.Scatter(
                x=d.index,
                y=d['n_killed'],
                name = "Killed",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

injured = go.Scatter(
                x=d.index,
                y=d['n_injured'],
                name = "injured",
                line = dict(color = '#7F7F7F'),
                opacity = 0.8)

data = [killings,injured]

iplot(go.Figure(data=data))


# ## Sudden spike in the number of school shootings from the year 2013-2014

# <img src='https://ichef.bbci.co.uk/news/660/cpsprodpb/2AFC/production/_100140011_hi045042704.jpg'>

# ### Researchers point out that owning a gun is an attribute of school shootings, but cannot be considered a cause in and of itself. Teens, of course, cannot legally purchase guns, but they can steal them from parents, buy them on the blackmarket, or obtain them through gang affiliations.
# 
# 
# 
# ### While gun control advocates argue tougher legislation would reduce gun violence, others believe that stricter gun laws would not prevent criminals from getting their hands on weapons, whether guns or homemade explosives. Some mental health experts even argue that the push for greater legislation overshadows the much deeper and more complex mental health issues driving gun violence in schools.
# 
# 

# ## POLITICIANS MUST UNDERSTAND THAT THE LIFE OF CHILDREN IS MORE IMPORTANT THAN THE NRA MONEY 
