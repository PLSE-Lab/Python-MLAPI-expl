#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import Image
Image("/kaggle/input/nflanalyticsimages/breakawaybanner.JPG")


# This report identifies high risk factors for cleat-surface interaction injuries. I created novel player movement metrics using next gen stats and then created a machine learning model to predict the surface. As expected, this model shows that there are significant performance differences between synthetic and natural turf.  What surprised me is that it showed that weather, specifically wet or dry fields, are equally important.  The combination of surface type and weather, which I call surface conditions, play a key role in non contact injuries.  
# 
# The primary cause is simply the sport of football. The skill position players make very sharp cuts at high speeds, which are dangerous on all surfaces. Even if we replaced all synthetic fields with grass, that would only prevent 27% of the injuries. Turf design is currently constrained by the opposite objectives of high traction for performance and the need to not "catch" cleats to prevent injuries.  The way to escape that constraint can be found in a 50 year old idea -- breakaway cleats.  Whether ski boot style bindings or sheer pins on individual spikes, breakaway cleats would allow greater performance and greater safety.  If turf can ignore the "catch" requirement, they could focus on other areas such as concussion prevention.  

# In[ ]:


import numpy as np
import pandas as pd
from IPython.display import HTML, Image
import warnings
import plotly
import plotly.express as px

pd.set_option("display.max_columns", 100)
th_props = [('font-size', '13px'), ('background-color', 'white'), 
            ('color', '#666666')]
td_props = [('font-size', '15px'), ('background-color', 'white')]
styles = [dict(selector="td", props=td_props), dict(selector="th", 
            props=th_props)]


# In[ ]:


play = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')
inj = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
oneplayer = pd.read_parquet('/kaggle/input/breakaway-cleats-loading-track-data/oneplayer.parq')
track = pd.read_parquet('/kaggle/input/breakaway-cleats-loading-track-data/track.parq')
injtrack = pd.read_parquet('/kaggle/input/breakaway-cleats-loading-track-data/injtrack.parq')


# <a id='ia'></a>
# <div class="h2"> Analysis </div>
# When I analyzed the injury data, I was concerned about the low number of samples from a statistically point of view. To avoid drawing false conclusions on small sample sizes, I grouped data.  I simplified weather into wet or dry surface.  Roster positions were divided into skill positions and lineman. The adjusted injury counts are what the injury count would be if 100% of the games were played under that condition. 
# 
# The Analysis shows that there are 3 large risk factors for turf/cleat injuries -- Weather, Field Type, and Roster Position.  All three factors indicate support the theory that surfaces are currently at a delicate balancing point between too much and too little traction.

# In[ ]:


df = inj.groupby(['Surface'])['PlayerKey'].count().reset_index()
df.columns = ['Surface','InjuryCount']
df['InjuryCount'] = np.where(df.Surface=="Natural",df.InjuryCount*32/20,df.InjuryCount*32/12)
fig = px.bar(df, x='Surface', y='InjuryCount',
                 width=800, height=400)
fig.update_layout(
    title="Synthetic Fields have 2x injuries of Natural Fields",
    xaxis_title="Surface Type",
    yaxis_title="Adjusted Injury Count",
    font=dict(
        family="Courier New, monospace",
        size=18
    )
)
fig.show()


# In[ ]:


Weatherdict = {"10% Chance of Rain": "Dry",
               "30% Chance of Rain": "Dry",
              "Clear":"Dry",
"Clear Skies":"Dry",
"Clear and Cool":"Dry",
"Clear and Sunny":"Dry",
"Clear and cold":"Dry",
"Clear and sunny":"Dry",
"Clear and warm":"Dry",
"Clear skies":"Dry",
"Clear to Partly Cloudy":"Dry",
"Cloudy":"Dry",
"Cloudy and Cool":"Dry",
"Cloudy and cold":"Dry",
"Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.":"Dry",
"Cloudy, 50% change of rain":"Dry",
"Cloudy, Rain":"Wet",
"Cloudy, chance of rain":"Dry",
"Cloudy, fog started developing in 2nd quarter":"Dry",
"Cloudy, light snow accumulating 1-3":"Dry",
"Cold":"Dry",
"Controlled Climate":"Dry",
"Coudy":"Dry",
"Fair":"Dry",
"Hazy":"Dry",
"Heat Index 95":"Dry",
"Heavy lake effect snow":"Wet",
"Indoor":"Dry",
"Indoors":"Dry",
"Light Rain":"Wet",
"Mostly Cloudy":"Dry",
"Mostly Coudy":"Dry",
"Mostly Sunny":"Dry",
"Mostly Sunny Skies":"Dry",
"Mostly cloudy":"Dry",
"Mostly sunny":"Dry",
"N/A (Indoors)":"Dry",
"N/A Indoor":"Dry",
"Overcast":"Dry",
"Partly Cloudy":"Dry",
"Partly Clouidy":"Dry",
"Partly Sunny":"Dry",
"Partly clear":"Dry",
"Partly cloudy":"Dry",
"Partly sunny":"Dry",
"Party Cloudy":"Dry",
"Rain":"Wet",
"Rain Chance 40%":"Dry",
"Rain likely, temps in low 40s.":"Wet",
"Rain shower":"Wet",
"Rainy":"Wet",
"Scattered Showers":"Wet",
"Showers":"Wet",
"Snow":"Wet",
"Sun & clouds":"Dry",
"Sunny":"Dry",
"Sunny Skies":"Dry",
"Sunny and clear":"Dry",
"Sunny and cold":"Dry",
"Sunny and warm":"Dry",
"Sunny, Windy":"Dry",
"Sunny, highs to upper 80s":"Dry",
"cloudy":"Dry",
}
Rosterdict1 = {"Cornerback":"Fast",
"Defensive Lineman":"Slow",
"Linebacker":"Medium",
"Offensive Lineman":"Slow",
"Running Back":"Fast",
"Safety":"Medium",
"Tight End":"Medium",
"Wide Receiver":"Fast"}
Rosterdict2 = {
"Cornerback":"Defense",
"Defensive Lineman":"Defense",
"Linebacker":"Defense",
"Offensive Lineman":"Offense",
"Running Back":"Offense",
"Safety":"Defense",
"Tight End":"Offense",
"Wide Receiver":"Offense"}
Rosterdict3={
"Cornerback":"Defender",
"Defensive Lineman":"Blocker",
"Linebacker":"Defender",
"Offensive Lineman":"Blocker",
"Running Back":"Catcher",
"Safety":"Defender",
"Tight End":"Catcher",
"Wide Receiver":"Catcher"}
Rosterdict4={
"Cornerback":"Skill",
"Defensive Lineman":"Lineman",
"Linebacker":"Skill",
"Offensive Lineman":"Lineman",
"Running Back":"Skill",
"Safety":"Skill",
"Tight End":"Skill",
"Wide Receiver":"Skill"}
Bodydict={
"Ankle":"Joint",
"Foot":"Feet",
"Heel":"Feet",
"Knee":"Joint",
"Toes":"Feet"}

RosterEncode1 = {"Fast":2,
"Medium":1,
"Slow": 0}

RosterEncode = {"Skill":0,
"Lineman":1}

TurfEncode = {'Synthetic':1,'Natural':0}
WeatherEncode = {"Dry":0,"Wet":1}


play['Dry']=play['Weather']
play = play.replace({"Dry": Weatherdict})

play['Dry'] = np.where(play['Dry'].isin(["Dry",'Wet']),play['Dry'],'Wet')
play['Roster']=play['RosterPosition']
play = play.replace({"Roster": Rosterdict4})
play['Roster1']=play['RosterPosition']
play = play.replace({"Roster2": Rosterdict1})




play['IsDry'] = play['Dry']
play['Turf'] = play['FieldType']


play = play.replace({'Roster':RosterEncode})
play = play.replace({'Roster1':RosterEncode1})
play = play.replace({'IsDry':WeatherEncode})
play = play.replace({'Turf':TurfEncode})

play = play[play.Roster.isin([0,1])].copy()

play['IsPass'] = np.where(play['PlayType'].isin(['Rush','Pass']),play['PlayType'],2)
play['IsPass'] = np.where(play['PlayType']=='Pass',0,play.IsPass)
play['IsPass'] = np.where(play['PlayType']=='Rush',1,play.IsPass)

play['Dry']=play['Weather']
play = play.replace({"Dry": Weatherdict})

play['Dry'] = np.where(play['Dry'].isin(["Dry",'Wet']),play['Dry'],'Wet')
play['Roster']=play['RosterPosition']
play = play.replace({"Roster": Rosterdict4})
play['Roster1']=play['RosterPosition']
play = play.replace({"Roster2": Rosterdict1})

play['IsDry'] = play['Dry']
play['Turf'] = play['FieldType']

play['RosterType'] = play.Roster
play = play.replace({'Roster':RosterEncode})
play = play.replace({'Roster1':RosterEncode1})
play = play.replace({'IsDry':WeatherEncode})
play = play.replace({'Turf':TurfEncode})

play = play[play.Roster.isin([0,1])].copy()

play['IsPass'] = np.where(play['PlayType'].isin(['Rush','Pass']),play['PlayType'],2)
play['IsPass'] = np.where(play['PlayType']=='Pass',0,play.IsPass)
play['IsPass'] = np.where(play['PlayType']=='Rush',1,play.IsPass)

play['PlayKey'] = play.PlayKey.fillna('0-0-0')
id_array = play.PlayKey.str.split('-', expand=True).to_numpy()
play['PlayerKey'] = id_array[:,0].astype(int)
play['GameID'] = id_array[:,1].astype(int)
play['PlayKey'] = id_array[:,2].astype(int)


# In[ ]:


inj = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
inj['injGameKey'] = inj.GameID.fillna('0-0')
id_array = inj.injGameKey.str.split('-', expand=True).to_numpy()
inj['GameID'] = id_array[:,1].astype(int)
inj['PlayerKey'] = id_array[:,0].astype(int)
inj['injKey'] = inj.PlayKey.fillna('0-0-0')
id_array = inj.injKey.str.split('-', expand=True).to_numpy()
inj['PlayKey'] = id_array[:,2].astype(int)
injplay = inj[inj['PlayKey']>0]
injplay = injplay.merge(play, on=['PlayerKey','GameID','PlayKey'])
game = play[['GameID','PlayerKey','Dry','Roster','Turf']].drop_duplicates()
injgame = inj.merge(game, on=['GameID','PlayerKey'],how='left')


# In[ ]:


df = injplay.groupby(['RosterType'])['PlayerKey'].count().reset_index()
df.columns = ['RosterType','InjuryCount']
fig = px.bar(df, x='RosterType', y='InjuryCount',
                 width=800, height=400)
fig.update_layout(
    title="Skill Positions have 5x injuries of Lineman",
    xaxis_title="",
    yaxis_title="Injury Count",
    font=dict(
        family="Courier New, monospace",
        size=18
    )
)
fig.show()


# In[ ]:


df = injgame.groupby(['Dry','Surface'])['PlayerKey'].count().reset_index()
df.columns = ['Dry','FieldType','InjuryCount']
df2= play.groupby(['Dry','FieldType'])['PlayerKey'].count().reset_index()
df =df.merge(df2)
#Adjustment factor = number of plays/number of plays under that condition
df['AdjustedCount']=df['InjuryCount']*len(play)/df['PlayerKey']
fig = px.bar(df, x='FieldType', y='AdjustedCount', color='Dry',barmode='group' ,
                 width=800, height=400)
fig.update_layout(
    title="Rain makes synthetic better and natural worse",
    xaxis_title="Weather",
    yaxis_title="Adjusted Injury Count",
    font=dict(
        family="Courier New, monospace",
        size=18
    )
)
fig.show()


# #### <a id='st'></a>
# <div class="h2"> Statistics </div>
# In this section, I looked at performance under the 4 types of surface conditions (wet/dry, natural/synthetic).  There are significant differences in speed and change of direction between the 4 groups.  In the graphs below, I looked at max values for each play.

# In[ ]:


Play_trk=play.merge(track,on=['PlayerKey','GameID','PlayKey'])
Play_trk['DryTurf']=Play_trk['Dry'].astype(str)+Play_trk['FieldType'].astype(str)
dfMax = Play_trk.groupby(['GameID','PlayKey']).max().reset_index()


# In[ ]:


fig = px.box(dfMax, x="DryTurf", y="dis",
                 width=800, height=500)
fig.update_layout(
    title="Differences in Max Speed",
    xaxis_title="Surface Type + Weather",
    yaxis_title="Max Speed",
    font=dict(
        family="Courier New, monospace",
        size=18
    )
)
fig.show()


# In[ ]:



fig = px.box(dfMax, x="DryTurf", y="Delta_Dir_2",
                 width=800, height=600)
fig.update_layout(
    title="Differences in Max Change of Direction",
    xaxis_title="Surface Type + Weather",
    yaxis_title="Max Change of Direction",
    font=dict(
        family="Courier New, monospace",
        size=18
    )
)
fig.show()


# In[ ]:


fig = px.box(dfMax[dfMax.Delta_Dis_2<0.2], x="DryTurf", y="Delta_Dis_2",
                 width=800, height=600)
fig.update_layout(
    title="Differences in Max Acceleration",
    xaxis_title="Surface Type + Weather",
    yaxis_title="Max Acceleration",
    font=dict(
        family="Courier New, monospace",
        size=18
    )
)
fig.show()


# #### <a id='ml'></a>
# <div class="h2"> Machine Learning Model </div>
# **Variables**
# 1. Calculated the differences in dis,direction,orientation, and angle (direction-orientation) from one row to the other. I used dis as a proxy for speed because speed appeared to have some data quality issues.
# 2. Rolling mean and standard deviation over 5 and 10 rows (0.5 to 1 second).  This time period was picked because it correlates with approximately the time it takes an athlete to perform a cut.
# 
# **Initial Results**
# 
# The first model showed that playerkey was by far the most important variable. In other words, the best way to know whether a surface is natural or synthetic is to focus on the actions of an individual player on both.  For the next models, I used only the data from one player.
# 
# **Models Created**
# 1. Predict Turf given next gen stats (AUC = 0.94).  The most important feature by far was weather. 
# 2. Predict Weather given next gen stats (AUC = 0.88). The most important feature was turf.
# 
# Details can be found in the referenced notebooks.
# 
# 
# 

# In[ ]:


Image("/kaggle/input/breakaway-cleats-lightgbm/shap.png")


# In[ ]:


Image("/kaggle/input/breakaway-cleats-lightgbm/roc.png")


# #### <a id='c'></a>
# <div class="h2"> Conclusion </div>
# Changes in surface conditions impact the quality of the play and increase the risk of injury.  It is impossible to create a perfect field and control the weather.  It is possible to design a cleat that will prevent non contact injuries.
