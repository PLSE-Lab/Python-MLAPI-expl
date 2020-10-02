#!/usr/bin/env python
# coding: utf-8

# Hi there. I want to make practice with this data. You can see just some code and graphics that is it.

# In[96]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected =True)
import plotly.graph_objs as go
from wordcloud import WordCloud
import os
print(os.listdir("../input"))


# In[97]:


data = pd.read_csv("../input/tables_1968_2019.csv")


# In[122]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# As you can see, there is a important relation each other. I want to see points table of 2018/2019 season.

# In[130]:


data[data.season =="2018/2019"]


# Now, I will examine Arsenal, Chelsea and Liverpool from 196/1968 season to 2018/2019 season.

# In[131]:


trace1 = go.Scatter(
                    x = data[data.team == "Arsenal"].season,
                    y =  data[data.team == "Arsenal"].points,
                    name = "Arsenal",
                    marker = dict(color = 'rgba(16, 52, 200, 0.8)'))
trace2 = go.Scatter(
                    x = data[data.team == "Chelsea"].season,
                    y =  data[data.team == "Chelsea"].points,
                    name = "Chelsea",
                    marker = dict(color = 'rgba(160, 82, 100, 0.8)'))
trace3 = go.Scatter(
                    x = data[data.team == "Liverpool"].season,
                    y =  data[data.team == "Liverpool"].points,
                    name = "Liverpool",
                    marker = dict(color = 'rgba(16, 182, 50, 0.8)'))

temp_data = [trace1,trace2,trace3]
layout = dict(title = 'Arsenal-Chelsea-Liverpool Comparison of Points', xaxis= dict(title= 'Seasons',ticklen= 3,zeroline= False))
fig = dict(data = temp_data, layout = layout)
iplot(fig,filename="line-mode")


# Next step, I will examine scores of them from 1967/1968 season to 2018/2019 season.

# In[132]:


trace4 = go.Scatter(
                    x = data[data.team == "Arsenal"].season,
                    y =  data[data.team == "Arsenal"].f,
                    name = "Arsenal",
                    marker = dict(color = 'rgba(16, 52, 200, 0.8)'))
trace5 = go.Scatter(
                    x = data[data.team == "Chelsea"].season,
                    y =  data[data.team == "Chelsea"].f,
                    name = "Chelsea",
                    marker = dict(color = 'rgba(160, 82, 100, 0.8)'))
trace6 = go.Scatter(
                    x = data[data.team == "Liverpool"].season,
                    y =  data[data.team == "Liverpool"].f,
                    name = "Liverpool",
                    marker = dict(color = 'rgba(16, 182, 50, 0.8)'))

temp_data = [trace4,trace5,trace6]
layout = dict(title = 'Arsenal-Chelsea-Liverpool Comparison', xaxis= dict(title= 'Seasons',ticklen= 3,zeroline= False))
fig = dict(data = temp_data, layout = layout)
iplot(fig,filename="line-mode")


# Lets see points and goals of Chelsea from 1967/1968 season to 2018/2019 season.

# In[134]:


trace7 = go.Scatter(
                    x = data[data.team == "Chelsea"].season,
                    y =  data[data.team == "Chelsea"].points,
                    name = "Points of Chelsea",
                    marker = dict(color = 'rgba(16, 52, 200, 0.8)'))
trace8 = go.Scatter(
                    x = data[data.team == "Chelsea"].season,
                    y =  data[data.team == "Chelsea"].f,
                    name = "Goal of Chelsea",
                    marker = dict(color = 'rgba(160, 82, 100, 0.8)'))

temp_data = [trace7,trace8]
layout = dict(title = 'Chelsea Goal-Point', xaxis= dict(title= 'Seasons',ticklen= 3,zeroline= False))
fig = dict(data = temp_data, layout = layout)
iplot(fig,filename="line-mode")


# Lets see points and goals of Liverpool from 1967/1968 season to 2018/2019 season.

# In[135]:


trace9 = go.Scatter(
                    x = data[data.team == "Liverpool"].season,
                    y =  data[data.team == "Liverpool"].points,
                    name = "Points of Liveerpool",
                    marker = dict(color = 'rgba(16, 52, 100, 0.8)'))
trace10 = go.Scatter(
                    x = data[data.team == "Liverpool"].season,
                    y =  data[data.team == "Liverpool"].f,
                    name = "Goal of Liverpool",
                    marker = dict(color = 'rgba(160, 82, 50, 0.8)'))

temp_data = [trace9,trace10]
layout = dict(title = 'Liverpool Goal-Point', xaxis= dict(title= 'Seasons',ticklen= 3,zeroline= False))
fig = dict(data = temp_data, layout = layout)
iplot(fig,filename="line-mode")


# Lets see points and goals of Manchester City from 1967/1968 season to 2018/2019 season.

# In[136]:


trace11 = go.Scatter(
                    x = data[data.team == "Manchester City"].season,
                    y =  data[data.team == "Manchester City"].points,
                    name = "Points of Manchester City",
                    marker = dict(color = 'rgba(16, 152, 100, 0.8)'))
trace12 = go.Scatter(
                    x = data[data.team == "Manchester City"].season,
                    y =  data[data.team == "Manchester City"].f,
                    name = "Goal of Manchester City",
                    marker = dict(color = 'rgba(100, 82, 150, 0.8)'))

temp_data = [trace11,trace12]
layout = dict(title = 'Manchester City Goal-Point', xaxis= dict(title= 'Seasons',ticklen= 3,zeroline= False))
fig = dict(data = temp_data, layout = layout)
iplot(fig,filename="line-mode")


# Now, I will create a graph to see points, scored, and defeated goal of all teams in the 2018/2019 season.

# In[137]:


data1819 = data[data.season =="2018/2019"]
pointsallteams = go.Bar(
x=data1819.team,
y=data1819.points,
name="points",
marker = dict(color ="rgba(250,20,20,0.4)"))
scoresallteams = go.Bar(
x=data1819.team,
y=data1819.f,
name="Scores",
marker = dict(color ="rgba(50,20,220,0.4)"))
concedegoalallteams = go.Bar(
x=data1819.team,
y=data1819.a,
name="defeated goal",
marker = dict(color ="rgba(50,220,20,0.4)"))
data1=[pointsallteams, scoresallteams, concedegoalallteams]
layout = go.Layout(barmode ="group")
fig = go.Figure(data=data1,layout=layout)
iplot(fig)


# Lastly, I want to create a wordcloud to see champion teams.

# In[115]:


xchampion = data[data.pos == 1]
xchampion


# In[129]:


# data prepararion
#x2011 = timesData.country[timesData.year == 2011]


plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=1024,
                          height=768
                         ).generate(" ".join(xchampion.team))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()


# Your comments are important for me. Thank you for your time. 
# Best regards.
