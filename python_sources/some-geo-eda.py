#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


# ### Summarize Missings and Unexpected Values

# In[ ]:


train = pd.read_csv('../input/birdsong-recognition/train.csv')
print('{} observations in training dataset.'.format(train.shape[0]))
print(' ')
print('{} features contain missing values in training dataset:'.format(train.columns[train.isnull().sum() > 0].shape[0]))
print('{}'.format(train.columns[train.isnull().sum() > 0].tolist()))
print(' ')

for col in train.columns[train.isnull().sum() > 0]:
    print('{}: {} missing values / {} %'          .format(col, train.isnull().sum()[col], round(train.isnull().sum()[col]/train.shape[0]*100, 2)))
print(' ')

print('{} incomplete date values'.format(train['date'].str.endswith('00').sum()))
print('{} observations missing day value.'.format(train['date'].str.endswith('00').sum()))
print('{} observations missing month value.'.format((train['date'].str[-5:-3] == '00').sum()))
print('{} observations missing year value.'.format((train['date'].str[:4] == '0000').sum()))
print(' ')

print('{} observations without longitude'.format((train['longitude']=='Not specified').sum()))
print('{} observations without latitude'.format((train['latitude']=='Not specified').sum()))


# #### Date Adjustments

# In[ ]:


# Obs 5048, 5049 labeled with year 1012
train.loc[train['date'].str.startswith('1012'), 'date'] = '2012'+train['date'].str[4:]
# Sue Riffe was at matching long, lat in July 2019
train.loc[train['date'].str.startswith('0201'), 'date'] = '2019'+train['date'].str[4:]

# Temporarily ignore 34 obs with missing date info, treat 112 as belonging to first day of month
train.drop(train[train['date'].str[-5:-3] == '00'].index, axis=0, inplace=True)
train['date_edit'] = False
train.loc[train['date'].str.endswith('00'), 'date_edit'] = True
train.loc[train['date'].str.endswith('00'), 'date'] = train.loc[train['date'].str.endswith('00'), 'date'].str.replace('00','01')

train['date'] = pd.to_datetime(train['date'])


# ### Recording Dates

# In[ ]:


plt.figure(figsize=(16,8))
sns.set()

ax1 = plt.subplot(1,3,1)
plt.hist(train['date'].dt.year, bins=20, alpha=0.8)
plt.xlabel('Year of Recording')

ax2 = plt.subplot(1,3,2, sharey=ax1)
plt.hist(train['date'].dt.month, bins=12, alpha=0.8)
plt.xlabel('Month of Recording')

ax3 = plt.subplot(1,3,3, sharey=ax1)
plt.hist(train['date'].dt.day, bins=31, alpha=0.8)
plt.xlabel('Day of Recording')

plt.suptitle('Distribution of Recording Dates', fontsize=16)
("")


# We observe a relatively large spike in recordings from the spring and summer months to no surprise. We have relatively consistent recording frequency in winter months in the 600 - 900 range.

# ### Lat & Long Spread

# In[ ]:


plt.figure(figsize=(14,8))
sns.set()

# Ignore 226 observations with missing coordinates
ax1 = plt.subplot(1,2,1)
plt.hist(train.loc[train['latitude'] != 'Not specified','latitude'].astype(float), bins=20, alpha=0.8)
plt.xlabel('Latitude')

ax2 = plt.subplot(1,2,2, sharey=ax1)
plt.hist(train.loc[train['longitude'] != 'Not specified','longitude'].astype(float), bins=20, alpha=0.8)
plt.xlabel('Longitude')

plt.suptitle('Distribution of Lat & Long', fontsize=16)
("")


# In[ ]:


fig = go.Figure(data=go.Scattergeo(
        lon = train['longitude'],
        lat = train['latitude'],
        text = train['primary_label'],
        mode = 'markers',
        marker = dict(
            size = 3,
            opacity = 0.5)
        ))

fig.update_layout(
        title = 'Bird Recordings Locations',
    )
fig.show()


# We observe a concentration of recordings from North America and Europe, with sparse entries from South America and Asia. In the US, we can see geographic clusters of recordings in coastal regions and some inland regions.

# In[ ]:


def make_bird_map(df, birds):
    '''Plot map of recordings for specified ebird_codes'''
    if type(birds) != list:
        birds = birds.tolist() # for single bird specified as str
    colors  = sns.color_palette("hls", len(birds)).as_hex()

    fig = go.Figure()
    for i in range(len(birds)):
        df_sub = df.loc[df['ebird_code']==birds[i],:]
        df_sub['text'] = 'Bird: ' + df_sub['species'] +             '<br>Recordist: ' + df_sub['recordist'] +             '<br>Date: ' + df_sub['date'].astype(str)
        
        fig.add_trace(
            go.Scattergeo(
                lon = df_sub['longitude'],
                lat = df_sub['latitude'],
                text = df_sub['text'],
                mode = 'markers',
                marker = dict(
                    size = 3,
                    opacity = 0.75,
                    color = colors[i]),
                name = birds[i],
            )
        )

    fig.update_layout(
            title = 'Recordings Locations (Selected Birds)',
        )
    fig.show()


# In[ ]:


make_bird_map(train, train['ebird_code'].value_counts().index[:10])


# Iterating through different birds, we already see clear geographic concentrations of recordings. Incorporating geo cordinates and matching to climate & biome information should improve prediction compared to audio alone and will be explored.

# In[ ]:


def make_recordist_map(df, recorders):
    '''Plot map of recordings for specified ebird_code'''
    if type(recorders) != list:
        recorders = recorders.tolist()
    colors  = sns.color_palette("hls", len(recorders)).as_hex()

    fig = go.Figure()
    for i in range(len(recorders)):
        df_sub = df.loc[df['recordist']==recorders[i],:]
        df_sub['text'] = 'Bird: ' + df_sub['species'] + '<br>Date: '+ df_sub['date'].astype(str)
        
        fig.add_trace(
            go.Scattergeo(
                lon = df_sub['longitude'],
                lat = df_sub['latitude'],
                text = df_sub['text'],
                mode = 'markers',
                marker = dict(
                    size = 3,
                    opacity = 0.5,
                    color = colors[i]),
                name = recorders[i],
            )
        )

    fig.update_layout(
            title = 'Recordings Locations (Selected Recordists)',
        )
    fig.show()


# In[ ]:


make_recordist_map(train, train['recordist'].value_counts().index[:10])


# ### Summarize Geo Info of Birds

# In[ ]:


def make_bird_geo_summary(df, birds):
    if type(birds) != list:
        birds = birds.tolist()
    df_sub = train.loc[train['ebird_code'].isin(birds),:]
    df_sub = df_sub.loc[df_sub['latitude']!='Not specified',:]
    df_sub['longitude'] = df_sub['longitude'].astype(float)
    df_sub['latitude'] = df_sub['latitude'].astype(float)
    
    make_bird_map(df_sub, birds)
    
    plt.figure(figsize=(14,8))
    ax1 = plt.subplot(1,3,1)
    sns.countplot(df_sub['date'].dt.month, alpha=0.8, palette='Blues_d')
    plt.xlabel('Month of Recording')
    #plt.xticks(ticks = [1,4,7,10], labels=[1,4,7,10])
    
    ax2 = plt.subplot(1,3,2)
    sns.boxplot(x=df_sub['date'].dt.quarter, y="longitude", data=df_sub)
    plt.xlabel('Quarter', fontsize=14)
    
    ax3 = plt.subplot(1,3,3)
    sns.boxplot(x=df_sub['date'].dt.quarter, y="latitude", data=df_sub)
    plt.xlabel('Quarter', fontsize=14)


# In[ ]:


# Identify birds with widest geographic appearances
sub_lat = train.loc[train['longitude']!='Not specified',:]
sub_lat['latitude'] = sub_lat['latitude'].astype('float')
travel_birds = sub_lat.groupby('ebird_code')['latitude'].std().sort_values(ascending=False).index


# In[ ]:


make_bird_geo_summary(train, travel_birds[:5])


# Here, we observe some birds have recordings from a wide range of geographic locations. The latitude plot suggests some birds in our dataset do migrate seasonally, suggesting time of recording may interact with location in a meaningful way when predicting bird types.

# In[ ]:


make_bird_geo_summary(train, travel_birds[-10:])


# In contrast, some birds are highly concentrated in specific areas. Assuming the training dataset is representative of these birds living areas, geographic location may be an effective screening criteria. Further research into biomes, climate, and bird specifies will be pursued to confirm this.
