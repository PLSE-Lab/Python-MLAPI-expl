#!/usr/bin/env python
# coding: utf-8

# ![](https://images.unsplash.com/photo-1512144253214-d94e86cd9189?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2689&q=80)

# In this notebook, I want to dig into all the features in depth. I am just getting started, so I will add more description here in future.
# 
# #### Credits
# 
# These are the Kernel's I have learned from and used in this Notebook:
# 
# * [neural networks + feature engineering for the win](https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win) by [bgmello](https://www.kaggle.com/bgmello) 

# In[ ]:


import numpy as np
import pandas as pd
from string import punctuation
import plotly.graph_objects as go
import re


# In[ ]:


train_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


# In[ ]:


train_df.head()


# ## Offense Formation
# 
# OffenseFormation field indicates the formation of the offensive team, and how each player is lined up in the field. For each type of formation presented in our dataset, I wrote down a short description below:
# 
# * [Singleback](https://www.xsosfootball.com/singleback-formation-sets/) - A classic running formation, where the running back lines up behind the quater back, and the wide receivers are spread out accross the field to create room for the running back.
# * [Shotgun](https://www.xsosfootball.com/shotgun-formation-sets/) - A classic passing formation, where the team is typically expected to throw the ball to the wide receivers. The running back usually lines up next to the quater back.
# * [I Formation](https://www.xsosfootball.com/i-formation-and-sets/) - A very common running formation, where the running back lines up behind the quater back, like the singleback formation. However, there is a full-back in between the quater back and the running back. The role of the full-back is to provide block (push other defenders away) and make room for the running back. Sometimes full-back also runs with the ball instead to throw in the element of surprise.
# * [Pistol](https://www.liveabout.com/the-shotgun-and-the-pistol-1335526) - Kind of a variation of Shotgun formation, with more hybrid formation to also give a good support for running back.
# * [Jumbo](https://www.xsosfootball.com/shotgun-empty-base-breakdown/) - This is the formation without a running back, or empty backfield. Very similar to Shotgun, except no running back at all.
# * [Wildcat](https://en.wikipedia.org/wiki/Wildcat_formation) - A player from another position (running back, full back) takes the snap of the ball instead of quaterback. 
# * [Ace](https://www.xsosfootball.com/formation-breakdown-singleback-ace/) - Very similar to I-Formation, except there is two tight-ends in the line of scrimmage. Tight Ends are big units, which means additional blocking option for the running back.
# * Empty - I am going to treat this as formation that are not well known, or documented.

# In[ ]:


# print yard per offensive formation
fig = go.Figure(go.Bar(
            x=train_df.groupby('OffenseFormation').mean()['Yards'].values,
            y=train_df.groupby('OffenseFormation').mean()['Yards'].index,
            orientation='h', ))

fig.update_layout(title="Avg Yard Per Play For Different Formation", xaxis_title="Avg Yards", yaxis_title="Formation")

fig.show()


# Interesting enough, for the passing formation Shotgun, the yard gain is very good! I agree with the idea of just simply doing one-hot encoding on this and not make things more complex.

# In[ ]:


# one hot encoding
train_df = pd.concat([train_df, pd.get_dummies(train_df['OffenseFormation'], prefix='Formation')], axis=1)
# drop one variable
train_df = train_df.drop(['Formation_EMPTY'], axis=1)


# ## Stadium Type
# 
# This field is supposed to give the "description of the stadium environment". In the cell below, we print out all the different types of values we have in this field. As you can see, there are typos and multiple ways of capturing the same information. One particular information we can gather from here is if the stadium roof was open or closed. Lets create a column called 'RoofOpen' and label 1 if the roof is open and 0 if the roof is closed.

# In[ ]:


', '.join(train_df['StadiumType'].value_counts().index)


# However, before we come to this conclusion (roof open or closed) we should gather more information and analyze the value in this field. I am using (an upgraded version of) the function from the kernel [neural networks + feature engineering for the win](https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win) for this part.

# In[ ]:


# fixing the typos in stadium
def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower() 
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub('-', ' ', txt)
    txt = ' '.join(txt.split()) # remove additional whitespace
    txt = txt.strip()
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('retractable', 'rtr')
    txt = txt.replace('retr', 'rtr') 
    txt = txt.replace('roofopen', 'roof open')
    txt = txt.replace('roofclosed', 'roof closed')
    txt = txt.replace('closed dome', 'dome closed')
    return txt


# In[ ]:


train_df['StadiumType'] = train_df['StadiumType'].apply(clean_StadiumType)


# In[ ]:


fig = go.Figure(data=[go.Pie(labels=train_df['StadiumType'].value_counts().index, values=train_df['StadiumType'].value_counts().values)])
fig.show()


# In[ ]:


train_df['StadiumType'].unique()


# Except for: rtr roof, nan, dome, heinz field, cloudy, bowl, domed, and null, we can clearly understand if the roof is open or closed. So for each of them I looked into the actual stadium the games are hosted in, and came to the following conclusions:
# * Heinz Field: is used when the game is hosted at "Heinz Field" Stadium. It is an open air stadium.
# * cloudy: Only used for "TIAA Bank Field" stadium, which is also an open air stadium.
# * bowl: Also only used for "TIAA Bank Field" stadium, so roof will be open.
# * domed: Used for "Mercedes-Benz Superdome", which is a domed stadium.  
# * null values: For null values, we have three stadiums: MetLife Stadium, StubHub Center, and TIAA Bank Field. All of them are open air stadiums. Check the cell below for the query.
# * rtr roof: Used only for "NRG Stadium", which can be open or closed based on different weather situation. For this stadium type, we have the following game weather: 'Sunny', 'Partly Cloudy', 'Mostly Cloudy', 'Cloudy', 'Clear', and 'Rainy'. The stadium and NFL policy (https://stadiumroofstatus.com/stadiuminfo.php?stadium=3) says that unless the weather is very bad, the roof should be open.

# In[ ]:


# Stadium names for where StadiumType is null
train_df[train_df['StadiumType'].isnull()]['Stadium'].value_counts()


# In[ ]:


def get_roofOpen(StadiumType, weather):
    
    roof_open = {'outdoor': 1, 'open': 1, 'indoor open roof': 1, 'outdoor rtr roof open': 1, 
                 'rtr roof open': 1, 'heinz field': 1, 'cloudy': 1, 'bowl': 1, 'domed open': 1,
                 'indoor': 0, 'rtr roof closed': 0, 'indoor roof closed': 0, 'dome closed': 0, 
                 'dome': 0, 'domed closed': 0, 'domed': 0}
    
    if StadiumType:
        # if stadium type is set look for it in the dict above
        if roof_open.get(StadiumType):
            return roof_open.get(StadiumType)
        # if 'rtr roof' then decide based on the weather
        else:
            if weather == 'Rainy':
                return 0
            else:
                return 1
    # if Stadium Type is empty then we know it is one of the open air stadiums
    else:
        return 1


# In[ ]:


train_df['RoofOpen'] = train_df.apply(lambda row: get_roofOpen(row.StadiumType, row.GameWeather), axis=1)


# In[ ]:


roof_open_performance = train_df.groupby(['RoofOpen']).mean()['Yards']
print('Avg Yard Per Play With Roof Open: {0:1.3f} '.format(roof_open_performance[1]))
print('Avg Yard Per Play With Roof Closed: {0:1.3f} '.format(roof_open_performance[0]))


# Apparently having the roof open is a big advantage for your run game!

# ## Turf
# 
# When it comes to the type of fields NFL teams prefers for their team, there is no clear consensus. The first generation of Artifical Turf gave it a bad reputation, as players really didn't feel they could perform well in those carpet-like fields. However, according to this [article](https://www.lawnstarter.com/blog/sports-turf/nfl-mlb-teams-artificial-turf-2019/) Artificial Turfs are making a big comeback. This is largely due to the new generation of Aritifical Turfs can help reduce injury and give players a consistent platform to perform despite of the weather condition. If you take a look at the following list of field types, there are too many and specific types. This may overwhelm or overfit our model, so like this [discussion](https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087) mentions, I am interested to know if the field is a grass field or not. 

# In[ ]:


', '.join(train_df['Turf'].unique())


# In[ ]:


# copied from the discussion https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087

def is_grass(Turf):
    
    grass = {'Field Turf': 0, 'A-Turf Titan': 0, 'Grass': 1, 'UBU Sports Speed S5-M': 0, 
            'Artificial': 0, 'DD GrassMaster': 0, 'Natural Grass': 1, 'UBU Speed Series-S5-M': 0, 
            'FieldTurf': 0, 'FieldTurf 360': 0, 'Natural grass': 1, 'grass': 1, 
            'Natural': 1, 'Artifical': 0, 'FieldTurf360': 0, 'Naturall Grass': 1, 'Field turf': 0, 
            'SISGrass': 0, 'Twenty-Four/Seven Turf': 0, 'natural grass': 1} 
    
    return grass.get(Turf)


# In[ ]:


train_df['Grass'] = train_df['Turf'].apply(is_grass)


# In[ ]:


turf_performance = train_df.groupby(['Grass']).mean()['Yards']
print('Avg Yard Per Play Without Grass: {0:1.3f} '.format(turf_performance[0]))
print('Avg Yard Per Play With Grass: {0:1.3f} '.format(turf_performance[1]))


# And we do see a slightly better performance on artifical turf as expected! Yes, the factor is not that much, but in a competitive league like NFL where every inch counts, this is a significant amount.
