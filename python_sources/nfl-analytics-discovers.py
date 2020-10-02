#!/usr/bin/env python
# coding: utf-8

#  # NFL Injury Analysis
# 
# 

# ### INTRODUCTION
# 
# 
# My goal in this notebook is to use given features and try to find some insights based on those features. As an data analyst, i tried to visualize every useful feature to find some the correlations and patterns between the features. I tried to split and and make the notebook as much as possible.
# **contents**
# 1. Importing libraries.
# 2. Overview of Injury Record and Play file.
# 3. Comparing missed days, injuried body part and surface with bar charts.
# 4. Looking for location of events, injuried body part and surface over the game field with scatter plots.
# 5. Looking for event types change during the game with horizontal bar chart.
# 6. Looking for the players movement during all game and just before getting injuried
# 7. Looking for acceleration change and mean difference of speed, distance and acceleration.

#  ### Importing the libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #visualization
import plotly.express as px #visualization
import chart_studio.plotly as py #visualization
import plotly.figure_factory as ff #visualization
import plotly.graph_objs as go #visualization
import matplotlib.pyplot as plt #visualization
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Injury Record

# In[ ]:


Injurydf=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')
Injurydf


# ## PlayList 

# * now lets look at Playlist data to see what we can mine from it.

# In[ ]:


Playdf=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')
Playdf


# * Fixing typos in StadiumType

# In[ ]:


Playdf['StadiumType']=Playdf['StadiumType'].replace(to_replace =['Outdoors', 'Outddors', 'Oudoors','Oudoor', 'Dome', 'Indoor'], 
                            value =["Outdoor", "Outdoor", "Outdoor","Outdoor",  "Indoors", "Indoors"])


# *  Visualizing some features after merge PlayList with InjuryRecord to see the patterns between Playlist features and number of Injuried players 

# In[ ]:


plays=list(Injurydf['PlayerKey'].unique())
Playdf_Copy=Playdf[Playdf.PlayerKey.isin(plays)]
Playdf_Copy


# In[ ]:


last_row0=Playdf_Copy.groupby('PlayerKey').tail(1)
rp=last_row0['StadiumType'].value_counts()
x1=rp[:5].index
y1=rp[:5].values

st=last_row0['Position'].value_counts()
x2=st[:5].index
y2=st[:5].values

wt=last_row0['RosterPosition'].value_counts()
x3=wt[:5].index
y3=wt[:5].values

pt=last_row0['PlayType'].value_counts()
x4=pt[:5].index
y4=pt[:5].values


# In[ ]:


plt.subplots_adjust(left=0.005, bottom=0.01, right=2.5, top=2.1)
plt.subplot(2, 2, 1)
sns.barplot(x1, y1)
plt.title('StadiumType',fontsize=20)
plt.ylabel('Number')
plt.subplot(2, 2, 2)
plt.title('RosterPosition', fontsize=20)
sns.barplot(x3, y3)
plt.ylabel('Number')
plt.subplot(2, 2, 3)
sns.barplot(x2, y2)
plt.title('Position', fontsize=20)
plt.ylabel('Number')
plt.subplot(2, 2, 4)
plt.title('PlayType', fontsize=20)
sns.barplot(x4, y4)
plt.ylabel('Number')

plt.show()


# * You can see PlayType, Position, RosterPosition and StadiumType comparing with number of injuries which can help us to enlighten us about which value causes most injury. The most dangreous positions are WR and OLB,  about %65 of injuries are on outdoor games,  Pass and Rush are the most couse playtype for injury and RosterPosition show us more then half of injuries are  due to Linebacker, Wide Receiver and Safety.

# ## PlayerTrackData

# * Now let's dive into some real work and work on players and try to understand what they are doing that causes them to be injuried

# In[ ]:


Trackdf=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')
Trackdf_Copy=Trackdf.merge(Injurydf, how='inner')


# In[ ]:


Trackdf


#  I will mark missed days to 1, 2, 3 and 4.
# * 1: more then 1 day
# * 2: more then 7 days
# * 3: more then 28 days
# * 4: more then 42 days

# In[ ]:


Trackdf_Copy['day_missed']=(Trackdf_Copy['DM_M1']+Trackdf_Copy['DM_M7']+Trackdf_Copy['DM_M28']+Trackdf_Copy['DM_M42'])
Trackdf_Copy=Trackdf_Copy.drop(['DM_M1', 'DM_M7', 'DM_M28','DM_M42'], axis=1)
Trackdf_Copy


# In[ ]:


last_row1=Trackdf_Copy.groupby('PlayerKey').tail(1)#last row for each player


# In[ ]:


rp=last_row1['BodyPart'].where(Trackdf_Copy['Surface']=='Natural').value_counts()
st=last_row1['BodyPart'].where(Trackdf_Copy['Surface']=='Synthetic').value_counts()

fig = go.Figure()
fig.add_trace(go.Bar(x=st.index, y=st.values, name="Synthetic"))
fig.add_trace(go.Bar(x=rp.index, y=rp.values, name="Natural"))
fig.update_layout(title_text="BodyPart & Surface  ",
                   width=850,
                   height=600,
                  title_font_size=20)
fig.show()


# As seen above on synthetic surface ankle is the most common injury type.

# In[ ]:


rp=last_row1['day_missed'].where(Trackdf_Copy['Surface']=='Natural').value_counts()
st=last_row1['day_missed'].where(Trackdf_Copy['Surface']=='Synthetic').value_counts()
fig = go.Figure()
fig.add_trace(go.Bar(x=st.index, y=st.values, name="Synthetic"))
fig.add_trace(go.Bar(x=rp.index, y=rp.values, name="Natural"))
fig.update_layout(title_text="day_missed & Surface ",
                   width=850,
                   height=500,
                  title_font_size=20,
                 barmode='stack')
fig.show()


# * Above you can see surface and day missed relationship 1 means more then 1 day, 2 means more then 7 days, 3 means more then 28 days and 4 means more then 42 days.

# In[ ]:


KN=last_row1['day_missed'].where(last_row1['BodyPart']=='Knee').value_counts()
FT=last_row1['day_missed'].where(last_row1['BodyPart']=='Foot').value_counts()
AN=last_row1['day_missed'].where(last_row1['BodyPart']=='Ankle').value_counts()
fig = go.Figure()
fig.add_trace(go.Bar(x=KN.index, y=KN.values, name="Knee"))
fig.add_trace(go.Bar(x=FT.index, y=FT.values, name="Foot"))
fig.add_trace(go.Bar(x=AN.index, y=AN.values, name="Ankle"))
fig.update_layout(title_text="day_missed & BodyPart",
                   width=850,
                   height=500,
                  title_font_size=20, 
                 barmode='stack')
fig.show()


# * Above we can see BodyPart and day_missed relation.

# ## What is happening on the field?

# In[ ]:


last_row2=Trackdf_Copy.groupby('PlayerKey').tail(1)#last row for each inuried player
fig = px.scatter(last_row2, x="x", y="y", 
              title="The players last position  ")
fig.show()


# In[ ]:


fig = px.scatter(last_row2, x="x", y="y", color="Surface", 
                 labels='BodyPart', title="Surface types for injuried player on the field")
fig.show()


# In[ ]:


fig = px.scatter(last_row2, x="x", y="y", color="BodyPart", title="BodyPart distrubiton on the field")
fig.show()


# In[ ]:


fig = px.scatter(last_row2, x="x", y="y", color="event", title="last event's distrubiton on the field")
fig.show() 


# ### Getting deeper looking for more

# * Below we have bar chart for last event that an injuried player were in and tackle is seems the most common one.

# In[ ]:


event_dist=last_row2['event'].value_counts()
fig = px.bar(x=event_dist.values, y=event_dist.index, orientation='h', title="last event of injuried players")
fig.show()


# In[ ]:


last_row5=Trackdf_Copy.groupby('PlayerKey').head(2)
event_dist1=last_row5['event'].value_counts()
fig = px.bar(x=event_dist1.values, y=event_dist1.index, orientation='h', title="first event  injuried players")
fig.show()


# *Down here i will see the all movements injuried players in all games. I used pandas isin to Play Type and Player Track Record.

# In[ ]:


players=list(Trackdf_Copy['PlayKey'].unique())
players_track=Trackdf[Trackdf.PlayKey.isin(players)]
players_track['PlayerKey']=players_track.PlayKey.str.split('-').str[0]
players_track.sample(4)


# In[ ]:


#Acceleration calculation found on stackoverflow
players_track['acceleration'] = (players_track['s'] - players_track['s'].shift(1)) / (players_track['time'] - players_track['time'].shift(1))


# In[ ]:


fig = px.line(players_track, x="x", y="y", color='PlayerKey',
              title="Movements of player on the field in the game ")
fig.show()


# In[ ]:


fig = px.line(Trackdf_Copy, x="x", y="y", color='PlayerKey',
              title="Movements of player on the field during injury ")
fig.show()


# In[ ]:


last_row3=players_track.groupby('PlayerKey').tail(15)#last row for each inuried player
fig = px.line(last_row3, x="x", y="y", color='PlayerKey',
              title="last moves of injuried players")
fig.show()


# In[ ]:


fig = px.line(players_track, x="time", y="acceleration", color="PlayerKey", title='Acceleration change with time')
fig.show()


# * Comparing some metrics for players during all game versus while injuried.

# In[ ]:


Injured_speed=Trackdf_Copy['s'].mean()
Total_speed=players_track['s'].mean()
Injured_dis=Trackdf_Copy['dis'].mean() 
Total_dis=players_track['dis'].mean()
Injured_ac=last_row3['acceleration'].mean() 
Total_ac=players_track['acceleration'].mean()
percentage=round(((Injured_speed-Total_speed)/Total_speed)*100)
percentage1=round(((Injured_dis-Total_dis)/Total_dis)*100)
percentage2=round(((Injured_ac-Total_ac)/Total_ac)*100)
print("The difference in speed, distance and acceleration changes throughout the game vs injury time:")
print(f"The players are moved  %{percentage} yards/meter faster then usual")
print(f"The players move %{percentage1} meter more then usual ")
print(f"The players moving acceleration %{percentage2} less then usual ")


# # Conlusion
# After some looking up and merging all data together, i felt free to do my approach. Here is what i think that players doing different get injuried.
# 1. event change is a considerable for not getting injuried.
# 2. When look at Surface type, day missed and BodyPart we can see some insights for instance,  Foot injuries tooks more then 28 days to heal and mostly happening on the synthetic surface.
# 3. Comparing to all values during the game, there is considerable change in the speed, distance and acceleration which can be a reason for injury during the game
