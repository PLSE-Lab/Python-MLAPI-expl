#!/usr/bin/env python
# coding: utf-8

# ## First and Future - Analytics

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os


# In[ ]:


# read in the files separately, all at once cause a kernel failure
injury_record = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")


# In[ ]:


injury_record.head()


# In[ ]:


injury_counts = injury_record['BodyPart'].value_counts()
injury_counts.plot.bar()
plt.xlabel('Body Part')
plt.show()


# In[ ]:


# compare body parts vs surface
injury_surface = pd.crosstab(injury_record['BodyPart'], injury_record['Surface'])


# In[ ]:


injury_surface.plot.bar()
plt.legend(title='Surface', loc = 'center')
plt.title('Surface vs Body Part Injured')
plt.show()
# plt.savefig('surf_body.jpg', bbox_inches='tight')


# In[ ]:


play_list = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")


# In[ ]:


injury_record.groupby('Surface').count()


# In[ ]:


play_list['Weather'] = play_list['Weather'].str.lower()


# In[ ]:


play_list['GameID'].nunique()


# In[ ]:


# id weather conditions by game, drop duplicates so it's clean to join to the injury record data
game_weather = play_list[['GameID','Weather']]
game_weather = game_weather.drop_duplicates(['GameID','Weather'])


# In[ ]:


# match player key and position, drop duplicates to join to injury record data
player_positions = play_list[['PlayerKey','RosterPosition']]
player_positions = player_positions.drop_duplicates('PlayerKey')


# In[ ]:


# same as above for stadium type
game_environment = play_list[['GameID','StadiumType']]
game_environment = game_environment.drop_duplicates(['GameID','StadiumType'])


# In[ ]:


# merge the data into injury record
injury_record = pd.merge(injury_record, game_environment, on = 'GameID', how = 'left')
injury_record = pd.merge(injury_record, player_positions, on = 'PlayerKey', how = 'left')
injury_record = pd.merge(injury_record, game_weather, on = 'GameID', how = 'left')


# In[ ]:


injury_record


# In[ ]:


injury_record['Weather'].unique()


# In[ ]:


injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Outdoors','Outdoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Open','Outdoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Outddors','Outdoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Oudoor','Outdoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Open Roof','Outdoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Retr. Roof - Open','Outdoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Heinz Field','Outdoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Retr. Roof - Outdoor','Outdoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Indoors','Indoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Dome','Indoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Retr. Roof-Closed','Indoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Domed, closed','Indoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Retr. Roof - Closed','Indoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Indoor, Roof Closed','Indoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Indoord, closed','Indoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Closed Indoor','Indoor')
injury_record['StadiumType'] = injury_record['StadiumType'].str.replace('Indoor, Outdoor Roof','Indoor')


# In[ ]:


injury_record['Weather'] = injury_record['Weather'].astype(str)


# In[ ]:


# function to create wet variable
def conditions(string):
        if "rain" in string:
            return 1
        else:
            return 0


# In[ ]:


injury_record['Wet'] = injury_record['Weather'].apply(conditions)


# In[ ]:


injury_record[injury_record['StadiumType'].isnull()]


# In[ ]:


# look at injuries with wet turf
pd.crosstab(injury_record['Wet'], injury_record['Surface'])


# In[ ]:


wet_injuries = pd.crosstab(injury_record['Wet'], injury_record['Surface'])
wet_injuries.plot.bar()
plt.legend(title='Surface', loc='upper right')
plt.xlabel('No Rain vs Rain')
plt.title('Wet vs Surface where Injuries Occur')
#plt.show()
plt.savefig('wet_surface_injuries.jpg', bbox_inches='tight')


# In[ ]:


pd.crosstab(injury_record['RosterPosition'], injury_record['Surface']).plot.bar()
plt.legend(title='Surface', loc='center')
plt.xlabel('Roster Position')
plt.title('Roster Position vs Surface Type Injury Breakdown')
#plt.show()
plt.savefig('surface_position.jpg', bbox_inches='tight')


# In[ ]:


# clean up the play_list DF for stadium type, indoor/outdoor
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Outdoors','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Open','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Outddors','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Oudoor','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Open Roof','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Retr. Roof - Open','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Heinz Field','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Retr. Roof - Outdoor','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoors','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Dome','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Retr. Roof-Closed','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Domed, closed','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Retr. Roof - Closed','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoor, Roof Closed','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoord, closed','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Closed Indoor','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoord','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Outdoor Retr Roof-Outdoor','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Ourdoor','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Retr. Roof-Outdoor','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoor, closed','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoord, Outdoor','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoord, open','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Retr. Roof Closed','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Outdor','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Outside','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Retractable Roof','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Bowl','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoor, Outdoor Roof','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoor, Outdoor','Indoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Indoor, open','Outdoor')
play_list['StadiumType'] = play_list['StadiumType'].str.replace('Cloudy','Outdoor')


# In[ ]:


# there are enough observations that we can model after dropping nulls
play_list['StadiumType'].dropna(inplace=True)
play_list['StadiumType'].unique()


# ## Game Injury Model - Predicting if a player will be injured in a game

# In[ ]:


mod1_data_pl = pd.concat([play_list,pd.get_dummies(play_list['RosterPosition'], prefix='Pos')],axis=1)


# In[ ]:


injured_gameid = list(injury_record['GameID'])


# In[ ]:


def cond_injured(df):
    if (df['GameID'] in injured_gameid):
        return 1
    else:
        return 0


# In[ ]:


mod1_data_pl['Injured'] = play_list.apply(cond_injured, axis=1)


# In[ ]:


mod1_data_pl = mod1_data_pl.drop_duplicates(['GameID'])


# In[ ]:


mod1_data_pl = mod1_data_pl.drop(columns=['PlayKey','RosterPosition','PlayType','PlayerGamePlay','Position',
                                          'PositionGroup','Pos_Kicker',])


# In[ ]:


mod1_data_pl


# In[ ]:


# create a list to identify wet turf
pl_wet = ['rain','showers','snow','scattered showers','light rain','cloudy, rain','rainy','cloudy, light snow accumulating 1-2"',
         'cloudy with periods of rain, thunder possible. winds shifting to wnw, 10-20 mph.','rain shower']


# In[ ]:


# for mapping for a model - new variable
turfs = {'Natural':0, 'Synthetic':1}
in_out = {'Outdoor':0, 'Indoor':1}


# In[ ]:


# function to create wet variable
def conditions(df):
    if (df['Weather'] in pl_wet) & (df['StadiumType'] == 'Outdoor'):
        return 1
    else:
        return 0


# In[ ]:


mod1_data_pl['Wet'] = mod1_data_pl.apply(conditions, axis = 1)


# In[ ]:


mod1_data_pl['FieldType'] = mod1_data_pl['FieldType'].map(turfs)


# In[ ]:


mod1_data_pl['StadiumType'] = mod1_data_pl['StadiumType'].map(in_out)


# In[ ]:


# these instances of injured players can be retained based on null values; weather is null but stadium is indoor
keep = ['36621-13','36696-24','40051-13','43826-7','47382-3']


# In[ ]:


mod1_data_pl.loc[(mod1_data_pl['Weather'].isnull()) &  (mod1_data_pl['GameID'].isin(keep)), 'Weather'] = 'ok'


# In[ ]:


drops_weather_stadiumtype = mod1_data_pl[(mod1_data_pl['Weather'].isnull()) | (mod1_data_pl['StadiumType'].isnull())].index


# In[ ]:


mod1_data_pl.drop(drops_weather_stadiumtype, inplace = True)


# In[ ]:


mod1_data_pl = mod1_data_pl.drop(columns=['PlayerKey','GameID','Weather'])


# In[ ]:


# correlation of the dataset
corr1 = mod1_data_pl.corr()
corr1


# In[ ]:


# create the X and y data for train_test_split
X = mod1_data_pl[[
 'PlayerDay',
 'PlayerGame',
 'StadiumType',
 'FieldType',
 'Temperature',
 'Pos_Cornerback',
 'Pos_Defensive Lineman',
 'Pos_Linebacker',
 'Pos_Offensive Lineman',
 'Pos_Quarterback',
 'Pos_Running Back',
 'Pos_Safety',
 'Pos_Tight End',
 'Pos_Wide Receiver',
    'Wet']].to_numpy(copy = True)
y = mod1_data_pl['Injured'].to_numpy(copy = True)


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)


# In[ ]:


# Fitting Random Forest to the Trng set
from sklearn.ensemble import RandomForestClassifier
# Create Classifier - entropy is ideal criterion
classifier = RandomForestClassifier(n_estimators = 150, max_depth = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


# Apply K-fold cross validation 
from sklearn.model_selection import cross_val_score
rf_scores = cross_val_score(classifier, X_train, y_train, cv=10)
rf_scores.mean()


# In[ ]:


# Predictions
y_pred = classifier.predict(X_test) 


# In[ ]:


# Make the confusion matrix - evaluate the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
cm


# In[ ]:


importances = classifier.feature_importances_
importances


# According to the RF model, field type (turf or natural) is the 4th most important feature in the model. Day, game, and temperature have more influence over an injury occurring in a game according to the random forest classifier. In general, the model does not predict well but helps understand the relationship of surface on the risk of being injured.

# 
# ## Access the tracking data

# In[ ]:


# due to the size of the tracking data file, the dask package is utilized
import dask.dataframe as dd
track_data = dd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')


# In[ ]:


track_data = track_data.drop('event', axis = 1)


# In[ ]:


track_data.head()


# In[ ]:


td = track_data.groupby('PlayKey').max().compute() - track_data.groupby('PlayKey').min().compute()


# In[ ]:


distances = track_data.groupby('PlayKey').dis.sum().compute()


# In[ ]:


mn_speed = track_data.groupby('PlayKey').s.mean().compute()


# In[ ]:


max_speed = track_data.groupby('PlayKey').s.max().compute()


# In[ ]:


max_speed.head()


# In[ ]:


td = td.drop(['dis','s'], axis = 1)


# In[ ]:


td = pd.merge(td, distances, on = 'PlayKey', how = 'left')
td = pd.merge(td, mn_speed, on = 'PlayKey', how = 'left')


# In[ ]:


td['speed_time_ratio'] = td['s']/td['time']
td['distance_time_ratio'] = td['dis']/td['time']
td['mean_speed'] = td['s']
td = td.drop('s', axis = 1)


# In[ ]:


td.head()


# ## Merging the track data with play list

# In[ ]:


play_and_injury_data = pd.merge(play_list, injury_record, how='left', on='PlayKey')
play_and_injury_data.drop(['PlayerKey_y', 'GameID_y', 'Weather_y', 'RosterPosition_y', 'StadiumType_y'], axis=1, inplace=True)


# In[ ]:


import math
def encode_injured_players(string):
    if (pd.isnull(string)):
        return 0
    else:
        return 1


# In[ ]:


play_and_injury_data['Injured'] = play_and_injury_data['BodyPart'].apply(encode_injured_players)


# In[ ]:


# function to create wet variable
def conditions(string):
        if "rain" in string:
            return 1
        else:
            return 0


# In[ ]:


play_and_injury_data['Weather_x'] = play_and_injury_data['Weather_x'].astype(str)
play_and_injury_data['Wet'] = play_and_injury_data['Weather_x'].apply(conditions)
play_and_injury_data = play_and_injury_data[play_and_injury_data['PlayType'] != '0']
play_and_injury_data = play_and_injury_data.rename(columns={'PlayerKey_x': 'PlayerKey', 
                                     'GameID_x': 'GameID', 
                                     'RosterPosition_x':'RosterPosition', 
                                     'StadiumType_x': 'StadiumType', 
                                     'Weather_x': 'Weather'})


# In[ ]:


play_and_injury_data.columns


# In[ ]:


mn_player_plays = play_and_injury_data.groupby('PlayerKey').mean()
mn_player_plays['PlayerGamePlay'].mean()


# In[ ]:


mod2_data_pl = play_and_injury_data


# In[ ]:


mod2_data_pl['FieldType'] = mod2_data_pl['FieldType'].map(turfs)
mod2_data_pl['StadiumType'] = mod2_data_pl['StadiumType'].map(in_out)


# In[ ]:


mod2_data_pl['StadiumType'].unique()


# In[ ]:


drops_weather_stadiumtype2 = mod2_data_pl[(mod2_data_pl['Weather'].isnull()) | (mod2_data_pl['StadiumType'].isnull())].index


# In[ ]:


mod2_data_pl.drop(drops_weather_stadiumtype2, inplace = True)


# In[ ]:


# merge tracking data onto mod2_data_pl
mod2_data_pl = pd.merge(mod2_data_pl, td, on = 'PlayKey', how = 'left')


# In[ ]:


mod2_data_pl


# In[ ]:


sns.boxplot(mod2_data_pl['Injured'], mod2_data_pl['mean_speed'])
plt.title('Mean Speed vs Injured')
plt.xlabel('Not injured vs injured')
plt.show()
#plt.savefig('mnspeed_injured.jpg', bbox_inches='tight')


# In[ ]:


sns.boxplot(mod2_data_pl['Injured'], mod2_data_pl['x'])
plt.title('X range (max-min) vs Injured')
plt.xlabel('Not injured vs injured')
plt.show()
#plt.savefig('xrange_injured.jpg', bbox_inches='tight')


# In[ ]:


sns.boxplot(mod2_data_pl['Injured'], mod2_data_pl['y'])
plt.title('Y range (max-min) vs Injured')
plt.xlabel('Not injured vs injured')
plt.show()
#plt.savefig('yrange_injured.jpg', bbox_inches='tight')


# In[ ]:


sns.boxplot(mod2_data_pl['Injured'], mod2_data_pl['time'])
plt.title('Time range (max-min) vs Injured')
plt.xlabel('Not injured vs injured')
plt.show()


# In[ ]:


sns.boxplot(mod2_data_pl['FieldType'], mod2_data_pl['mean_speed'])
plt.title('Average Speed vs Field Type')
plt.xlabel('Natural vs Turf')
plt.show()


# In[ ]:


sns.boxplot(mod2_data_pl['FieldType'], mod2_data_pl['x'])
plt.title('X range (max-min) vs Field Type')
plt.xlabel('Natural vs Turf')
plt.show()


# In[ ]:


sns.boxplot(mod2_data_pl['FieldType'], mod2_data_pl['y'])
plt.title('Y range (max-min) vs Field Type')
plt.xlabel('Natural vs Turf')
plt.show()


# In[ ]:


sns.boxplot(mod2_data_pl['FieldType'], mod2_data_pl['time'])
plt.title('Time range (max-min) vs Field Type')
plt.xlabel('Natural vs Turf')
plt.show()


# In[ ]:


mod2_injured = mod2_data_pl[mod2_data_pl['Injured'] == 1]
mod2_notinjured = mod2_data_pl[mod2_data_pl['Injured'] == 0]


# In[ ]:


mod2_injured['mean_speed'].mean()


# In[ ]:


mod2_notinjured['mean_speed'].mean()


# In[ ]:


mod2_injured_turf = mod2_injured[mod2_data_pl['FieldType'] == 1]
mod2_injured_grass = mod2_injured[mod2_data_pl['FieldType'] == 0]


# In[ ]:


mod2_injured_turf['mean_speed'].mean()


# In[ ]:


mod2_injured_grass['mean_speed'].mean()


# In[ ]:


sns.boxplot(mod2_injured['FieldType'], mod2_injured['mean_speed'])
plt.title('Mean speed vs Field Type - Injured Only')
plt.xlabel('Natural vs Turf')
plt.show()
#plt.savefig('mean_spd_field_type.jpg', bbox_inches='tight')


# In[ ]:


sns.boxplot(mod2_injured['FieldType'], mod2_injured['x'])
plt.title('X range (max-min) vs Field Type - Injured Only')
plt.xlabel('Natural vs Turf')
plt.show()


# In[ ]:


sns.boxplot(mod2_injured['FieldType'], mod2_injured['y'])
plt.title('Y range (max-min) vs Field Type - Injured Only')
plt.xlabel('Natural vs Turf')
plt.show()


# In[ ]:


sns.boxplot(mod2_data_pl['FieldType'], mod2_data_pl['dis'])
plt.title('Distance vs Field Type')
plt.xlabel('Natural vs Turf')
plt.show()
#plt.savefig('distance_turf.jpg', bbox_inches='tight')


# In[ ]:


sns.boxplot(mod2_data_pl['Injured'], mod2_data_pl['dis'])
plt.title('Distance vs Injured')
plt.xlabel('Not-injured vs Injured')
plt.show()
#plt.savefig('distance_injured.jpg', bbox_inches='tight')


# In[ ]:


sns.boxplot(mod2_data_pl['Injured'], mod2_data_pl['speed_time_ratio'])
plt.title('Speed_time_ratio vs Injured')
plt.xlabel('Not-injured vs Injured')
plt.show()


# In[ ]:


sns.boxplot(mod2_data_pl['FieldType'], mod2_data_pl['speed_time_ratio'])
plt.title('Speed_time_ratio vs FieldType')
plt.xlabel('Natural vs Turf')
plt.show()


# In[ ]:


turf = mod2_data_pl[mod2_data_pl['FieldType'] == 1]
turf['speed_time_ratio'].mean()


# In[ ]:


grass = mod2_data_pl[mod2_data_pl['FieldType'] == 0]
grass['speed_time_ratio'].mean()


# In[ ]:


grass['mean_speed'].mean()


# In[ ]:


turf['mean_speed'].mean()


# In[ ]:


sns.distplot(turf['mean_speed'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})


# In[ ]:


sns.distplot(grass['mean_speed'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})


# In[ ]:


grass['x'].mean()


# In[ ]:


grass['y'].mean()


# In[ ]:


turf['x'].mean()


# In[ ]:


turf['y'].mean()


# In[ ]:


sns.boxplot(mod2_data_pl['FieldType'], mod2_data_pl['distance_time_ratio'])
plt.title('Distance_time_ratio vs FieldType')
plt.xlabel('Natural vs Turf')
plt.show()


# In[ ]:


sns.boxplot(mod2_data_pl['Injured'], mod2_data_pl['distance_time_ratio'])
plt.title('Distance_time_ratio vs Injured')
plt.xlabel('Not-injured vs Injured')
plt.show()


# In[ ]:


injury_plays = play_and_injury_data[play_and_injury_data['Injured'] == 1]
pd.crosstab(injury_plays['PlayType'], injury_plays['FieldType']).plot.bar()
plt.legend(title='Surface', loc = 'upper right')
plt.title('Play Type by Field Type Where Injuries Occurred')
plt.show()
#plt.savefig('play_type_injuries.jpg', bbox_inches='tight')
# note - a previous plot was included where more injured instances occurred - the data was modified to capture
# only playkeys where injuries occur in this version


# In[ ]:


# Noticed all temperatures with -999 degrees were indoor stadiums that are temp controlled
play_and_injury_data['Temperature'].replace(-999, value=0, inplace=True)
injury_plays = play_and_injury_data[play_and_injury_data['Injured'] == 1]
injury_plays['Temperature'].hist(grid=False, bins=10)
plt.title('Injury Count by Temperature')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()
#plt.savefig('temp_injuries.jpg', bbox_inches='tight')


# In[ ]:


injury_plays['PlayerGamePlay'].hist(grid=False, bins=10)
plt.xlabel('Plays Played')
plt.ylabel('Count')
plt.title('Injury Count by Plays Played')
plt.show()
#plt.savefig('play_cnt_injuries.jpg', bbox_inches='tight')


# In[ ]:


injured_grass = grass[grass['Injured'] == 1]
injured_grass['PlayerGamePlay'].hist(grid=False, bins=10)
plt.xlabel('Plays Played')
plt.ylabel('Count')
plt.title('Injury Count by Plays Played on Natural Surface')
plt.show()
#plt.savefig('play_cnt_injuries_nat.jpg', bbox_inches='tight')


# In[ ]:


injured_turf = turf[turf['Injured'] == 1]
injured_turf['PlayerGamePlay'].hist(grid=False, bins=10)
plt.xlabel('Plays Played')
plt.ylabel('Count')
plt.title('Injury Count by Plays Played on Synthetic Surface')
plt.show()
#plt.savefig('play_cnt_injuries_turf.jpg', bbox_inches='tight')


# In[ ]:


mod2_data_pl = pd.merge(mod2_data_pl, max_speed, on = 'PlayKey', how = 'left')


# In[ ]:


mod2_data_pl['max_speed_play'] = mod2_data_pl['s']


# In[ ]:


max_spd_injured = mod2_data_pl[mod2_data_pl['Injured'] == 1]
max_spd_notinjured = mod2_data_pl[mod2_data_pl['Injured'] == 0]


# In[ ]:


max_spd_injured['max_speed_play'].hist(grid=False, bins=10)
plt.xlabel('Max Speed')
plt.ylabel('Count')
plt.title('Max Speed Injured Players - yards per second')
plt.show()
#plt.savefig('maxspeedinjured.jpg', bbox_inches='tight')


# In[ ]:


max_spd_notinjured['max_speed_play'].hist(grid=False, bins=100)
plt.xlim(left = 0)
plt.xlim(right = 10)
plt.xlabel('Max Speed')
plt.ylabel('Count')
plt.title('Max Speed Not Injured - yards per second')
plt.show()
#plt.savefig('maxspeednotinjured.jpg', bbox_inches='tight')


# In[ ]:




