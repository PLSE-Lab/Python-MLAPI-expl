#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Health issues in Americal football are common because of its nature of the full-contact game. But authority like the NFL is trying to reduce the health issues from its core. They are trying their best for finding solutions. The previous year, NFL created a competition for reducing concussion in punt and this year they created a competition for finding the relationship between playing surface, injury and player movement. 
# 
# This notebook is the main submission for this year NFL analytics competition. In this notebook, I have tried my best for finding the relationship between playing environmental factors, player movement and injury. I hope this will help the NFL to reduce the risk of injury in football.

# # Main findings and results 
# * **Result 1**: Previously we know, Synthetic turf has significantly higher injury rates on synthetic turf compared with natural turf (Mack et al., 2018; Loughran et al., 2019). Now we found that Toes and Ankle injuries are higher in synthetic turf compared to natural turf. Heel and foot injuries are higher in natural turf compared to synthetic turf. 
# 
# * **Result 2**: Temperature in injury play is higher compared to non-injury play. That means higher temperature contributes to injuries. Also, we found that most of the injuries happened in the outdoor stadium. 
# 
# * **Result 3**: 72% of injuries happened during the first 10 games of individual players. This is one of the important findings. Because this will help players, team, coach and supervisor to take previous necessary steps to reduce injury. We also found that the field that player got injured, in 55% injuries the field type is different from the immediate previous game. And goes up to 63% in synthetic turf. 
# 
# * **Result 3**: We have created several movement metrics for identifying injury risk in player movement. All of these metrics results are 
#     1. **Numer of Declaration, acceleration, no acceleration/deceleration in last 20 seconds**: We have noted that all these metrics slightly higher in the injured players but we don't find any significant differences in different surface condition. But we found that the number of declaration and acceleration are higher in synthetic turf. 
#     2. **Mean acceleration, speed, distance, direction, orientation, directional changes and orientation changes**: We found injured players have higher mean direction, orientation, directional changes and orientational changes compared to the non-injured players. But we don't find significant differences in a different field. 
#     3. **Max acceleration, speed, distance, direction, orientation, directional changes and orientation changes**: Among all these metrics, we found one metrics are useful. That is max-acceleration. We found that max acceleration is higher in injured player compared to the non-injured players and we found that max acceleration is very high in synthetic turf compared to natural turf. 

# # Proposal for reducing injuries based on our findings
# Now we saw surface, environment and movement contributions in injuries. But how do we solve this problem and reduce injuries?
# 
# We have to follow what FIFA has done. FIFA created a very important program called 'FIFA Quality Program' where they use some criteria to identify different materials for the game. One of them is FIFA quality turf. NFL is one of the top organization for American football. Because American football is vastly different from soccer, we have to make our own 'NFL Quality Standard Program'. Where the NFL can use criteria for identifying turf that does not contribute to injury rather, it helps reduce the risk of injuries.
# 
# * **Proposal 1**: Improve field response mechanism for artificial Turf. "Artificial turf is a harder surface than grass and does not have much "give" when forces are placed on it" [1]. We have to solve this problem. We have to make our artificial turf that mimics the natural grass. Because when players toys are bend or stressed during play and the turf doesn't help in that situation then player got injured in toy or ankle. 
# 
# * **Proposal 2**: Make artificial turf relief heat of the sun or in general. Because we saw a high temperature contributes to injuries. And artificial turf tends to retain heat compared to natural turf. 
# 
# * **Proposal 3**: American football is very tense and high contact game. Often player needs to move fast in any direction, jump and stop or tackle the player with high speed. For that high acceleration, speed and velocity are essential. But we saw in our analysis that in most injury play the acceleration and other movement metrics is high. But synthetic turf surfaces do not release cleats as readily as natural turf. For not releasing cleats like natural turf, players got injured. We have two ways to solve that problem.
# 
#     1. Improve field rubber and traction mechanism: We have to improve our field rubber and silicon and other variables for making easy releasing cleats and also we have to make sure it doesn't produce more friction or force that cause problems in player movement. 
#     
#     2. Use shoes that are exclusively built for synthetic turf: While many cleats manufactured today for football can be used on artificial turf or natural surfaces, many moulded cleats are designed and better served for use on artificial turf. The moulded cleats tend to provide better traction on artificial surfaces, whereas detachable studs on cleats tend to be too thick for artificial turf and do not provide adequate grip on such a surface [2].
#     
#     
#     
# * **Proposal 4**: One of the best benefits about artificial turf is that it needs less maintenance and most of the time it surfaces are plain and in rain, it doesn't have any decay or any other effect. In natural turf, the surface easily got problem after each game and in rain, the surface got a massive problem. One of the problems is not a plain surface. We saw in our analysis that heel and foot have mostly occurred in natural turf. 
# 
#   And from research [1] we know that foot and heel injuries occurred mostly because of uneven surface. So we have to make sure that natural turf that in each game and in-between game we have to repair natural turf. These will mostly reduce heel and foot injuries. **But we have to use real-time field analysis camera that uses machine learning to assess field condition. This way the NFL can make sure that field condition is good in the game. **
#   
# 
# * **Proposal 5**: One of the important things we found in our analysis is that most of the injuries happened in the player first 10 games. Secondly, we found that for injuries player has less experience with the injured surface, field and play type compared to the non-injured player. For that reason, we can say that player having less experience in playing surface, the environment is one of the main causes of injury. To solve this problem, we have two ways. 
# 
#     1. Assess player fitness and FM score: We have seen that 72% of the injuries happened during the first 10 games of each player. So to solve this problem we have to inform players, teams, coaches and teams about this information. And advice them to make sure to assess player fitness and functional movement score before games. 
#     
#     Also, we found that most of the injured player's previous game field is different. So coaches should make sure that player got enough practice on the field that he is going to play. 

# In[ ]:


import pandas as pd
import numpy as np
import gc
#################
# Import our dataset and libraies
#################

# basic libraries
import pandas as pd
import numpy as np
import textwrap

# plotly imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def split_id(dataframe):
    dataframe['PlayKey'] = dataframe.PlayKey.fillna('0-0-0')
    id_array = dataframe.PlayKey.str.split('-', expand=True).to_numpy()
    dataframe['PlayerKey'] = id_array[:,0]
    dataframe['GameID'] = id_array[:,1]
    dataframe['PlayKey'] = id_array[:,2]
    
    return dataframe


def percentage(x):
    """function for counting percentage in pandas column"""
    return round((x/x.sum())*100, 1)


def calculate_pre_experience(dataframe, column_name, new_col_name):
    unique_value_list = dataframe[column_name].dropna().unique().tolist()
    df_list = []
    for value in unique_value_list:
        temp = dataframe.loc[dataframe[column_name] == value]

        temp.loc[(temp.status == 'injured'), 'rolling_break'] = 1
        temp['rolling_break'] = temp['rolling_break'].shift(1)
        temp['rolling_break'] = temp['rolling_break'].fillna(0)

        temp['cumsum'] = temp['rolling_break'].cumsum()
        temp[new_col_name]= temp.groupby(['PlayerKey','cumsum'])[column_name].cumcount()
        
        df_list.append(temp)
    return pd.concat(df_list)




def calculate_velocity(x):
    temp = x.copy()
    displacement = temp[['x', 'y']].diff()
    displacement['displacement'] = np.sqrt(displacement.x**2 + displacement.y**2)
    displacement['displacement'] = displacement['displacement'].fillna(0)
    temp['velocity'] = (displacement['displacement'] / .10).values

    velocity = temp['velocity'].diff().fillna(0)
    temp['acceleration'] = (velocity / .10).values
    
    del velocity, displacement
    t = gc.collect()
    
    temp['velocity'] = temp['velocity'].astype('float32')
    temp['acceleration'] = temp['acceleration'].astype('float32')
    
    return temp



def calc_num_event(dataframe,
                   event_col_name,
                   event_col_value,
                   num_element,
                   new_col_name):
    temp = dataframe.groupby(['PlayerKey', 'GameID', 'PlayKey']).tail(num_element)
    temp = temp.loc[temp[event_col_name] == event_col_value]
    temp = temp.groupby(['PlayerKey', 'GameID', 'PlayKey'])[event_col_name].count().reset_index()
    temp = temp.rename(columns={event_col_name: new_col_name})
    return temp


# In[ ]:



df_track_data = pd.read_parquet('../input/using-track-data-with-small-memory/InjuryRecord.parq', engine='pyarrow')
df_injury = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
df_play_list = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')


df_injury = split_id(df_injury)
df_play_list = split_id(df_play_list)

df_injury[['GameID', 'PlayKey']] = df_injury[['GameID', 'PlayKey']].astype('int8')
df_injury['PlayerKey'] = df_injury['PlayerKey'].astype('int32')

df_play_list[['GameID', 'PlayKey']] = df_play_list[['GameID', 'PlayKey']].astype('int8')
df_play_list['PlayerKey'] = df_play_list['PlayerKey'].astype('int32')


df_injury['status'] = 'injured'




col_filter_for_merge = ['PlayerKey', 'GameID', 'PlayKey',
                        'RosterPosition', 'PlayerDay',
                        'PlayerGame', 'StadiumType', 'FieldType',
                        'Temperature', 'Weather','PlayType',
                        'PlayerGamePlay', 'Position', 'PositionGroup']

df_play_list_merge = df_play_list[col_filter_for_merge].merge(
    df_injury, on=['PlayerKey', 'GameID', 'PlayKey'], how='left')

df_play_list_merge['status'] = df_play_list_merge['status'].fillna('non injured')


# # Synthetic turf cause more injuries, but not all type of injuries!
# This section is the first and main section of our analysis. We will find out how surface and environmental variable contributes to injuries. We previously know that surface type affects injuries. From research, we know that synthetic turf causes more injuries compared to natural turf. But we have to dig dive a little more details. 

# In[ ]:


temp = df_injury.groupby(['BodyPart', 'Surface'])['PlayerKey'].count().to_frame('counts').reset_index()

temp['Percentage'] = temp.groupby(['BodyPart'])['counts'].apply(percentage)
px.bar(temp, y='Percentage', x='BodyPart', color='Surface', barmode='group')


# We are seeing very important results. Toes and Ankle are two major injuries that occurred most in synthetic turf and Foot and Heel injuries are prominent in natural turf. Why? 
# 
# ![](https://orthoinfo.aaos.org/link/3071e352c90a4cfbae465a89cfb176bb.aspx)
# 
# We are seeing foot position when toe injury occurred. Because of "Artificial turf is a harder surface than grass and does not have much "give" when forces are placed on it" [1], and from research, we know that Synthetic turf surfaces do not release cleats as readily as natural turf  (Kent et al., 2015). That' why synthetic turf contributes more in toes and ankle injuries. 
# 
# On the other hand, foot and heel injuries are prominent in natural turf. From medical research, we know that these two injuries happened because of uneven surfaces. One of the negative sides of natural turf is that it needs more maintenance. After using some moments the grass or field can get problems. One of the problems is surface becomes slightly bumpy or uneven. One of the main cause for foot and heel injury is uneven or bumpy surface. Because of the problem in natural grass, foot and heel injury is prominent in natural turf.

# # High temperature contributes more to injuries

# In[ ]:


px.box(df_play_list_merge.loc[df_play_list_merge.Temperature != -999], y='Temperature', color='status', facet_col='FieldType')


# We are seeing that in both field type (synthetic and natural ), injured game field temperature is higher compared to the non-injured population. This means that the higher temperature is one of the reasons for player injury. Why? From medical research that higher temperature makes player fatigue quickly and the muscle got tired fast. For that reason, a player is more likely to get injured at high temperature. 

# # Majority of the injuries occured in outdoor stadium

# In[ ]:


stadium_type_dict = {'Indoors': 'Indoor', 'Oudoor': 'Outdoor', 'Outdoors': 'Outdoor', 'Domed, closed': 'Closed Dome', 
                    'Domed': 'Dome', 'Ourdoor': 'Outdoor', 'Outddors': 'Outdoor', 'Outdor': 'Outdoor', 'Dome, closed': 'Closed Dome',
                    'Retr. Roof-Closed': 'Retr. Roof Closed', 'Retr. Roof-Open': 'Retr. Roof Open', 'Retr. Roof - Open': 'Retr. Roof Open',
                    'Dome, closed': 'Closed Dome', 'Domed, open': 'Domed, Open', 'Outside': 'Outdoor', 'Retr. Roof - Closed': 'Retr. Roof Closed',
                    'Closed Dome':'Dome', 'Domed, Open': 'Dome', 'Indoor, Open Roof': 'Ind', 'Closed Dome': 'Dome', 'Open': 'Outdoor',
                    'Retr. Roof Closed': 'Retractable Roof', 'Retr. Roof Open': 'Retractable Roof', 'Outdoor Retr Roof-Open': 'Retractable Roof',
                    'Indoor, Roof Closed':'Indoor', 'Ind': 'Indoor', 'Cloudy': 'Outdoor', 'Heinz Field': 'Outdoor', 'Bowl': 'Outdoor'}
df_play_list_merge = df_play_list_merge.replace(stadium_type_dict)

temp = df_play_list_merge.loc[df_play_list_merge.status == 'injured']
temp = temp.groupby(['StadiumType', 'FieldType'])['PlayerKey'].count().to_frame(name='counts').reset_index()
temp = temp.sort_values(['counts'], ascending=False)
temp['Percentage'] = temp.groupby(['FieldType'])['counts'].apply(percentage)

px.bar(temp, x='StadiumType', y='Percentage', color='FieldType', barmode='group')


# We are seeing that the majority of the injuries occurred in the outdoor stadium. Almost all the injuries of natural turf occurred in the outdoor stadium and synthetic turf injuries are also mostly happened in an outdoor stadium. We still don't know the exact cause for this but maybe in future, we will find.

# # 72% of the injuries happend during first 10 games of individual player
# We previously saw how surface effect in injuries and how to reduce it. But we need to focus on very important thing, that is player previous playing history. 

# In[ ]:


temp = df_play_list_merge[df_play_list_merge.status == 'injured']['PlayerGame'].value_counts().to_frame('counts').reset_index()
temp = temp.rename(columns={'index': 'Player Game'})
temp['Percentage'] = percentage(temp['counts'])
px.bar(temp, y='Percentage', x='Player Game')


# We are seeing very interesting and important results. 72% of injuries happened during the first 10 games of individual player. That means that player is not very experienced with the environment. Now for our curisity let's dig dive more. 

# # 55% of injuries, the field was different from immediate previous game

# In[ ]:


df_play_list = df_play_list.sort_values(['PlayerKey', 'GameID', 'PlayKey', 'PlayerDay'])
df_play_list_first = df_play_list.drop_duplicates(['PlayerKey', 'GameID'], keep='first')
df_play_list_first = df_play_list_first.merge(df_injury, on=['PlayerKey', 'GameID'], how='left')
df_play_list_first['status'] = df_play_list_first['status'].fillna('non injured')

df_play_list_first['previous_field_type'] = df_play_list_first.groupby(['PlayerKey'])['FieldType'].shift(1)


temp = df_play_list_first.loc[df_play_list_first.status == 'injured']
temp['is_same_field'] = np.where(temp['previous_field_type']==temp['FieldType'], 
                                           'yes', 'no')

temp = temp.groupby(['FieldType', 'is_same_field'])['PlayerKey'].count().to_frame('counts').reset_index()
temp['Percentage'] = temp.groupby(['FieldType'])['counts'].apply(percentage)

px.bar(temp, x='is_same_field', y='Percentage', color='FieldType', barmode='group')


# We are seeing very important results. 55% in injuries, field are different from the immediate previous field. And it goes up to 63% in synthetic turf. This is also one of the great findings of this notebooks analysis. This will help team, player and coach to prepare for such play that has different field from previous game so that player has less chance of getting injured. 

# # Numer of Declaration, acceleration, no acceleration/deceleration in last 20 seconds
# We previously know the differences in cleat-turf interactions, but it has yet to be determined whether player movement patterns and other measures of player performance differ across playing surfaces and how these may contribute to the incidence of lower limb injury.
# 
# We first need a way to characterize the player movement. To characterize player movement we have created several metrics. Now lets our first batch of movement metrics and how they affect injuries.

# In[ ]:


dataset = df_track_data.iloc[np.flatnonzero((df_track_data.event == 'ball_snap') | (df_track_data.event == 'kickoff'))[0]:]
dataset['event'].ffill(inplace=True)

displacement  = dataset[['x', 'y']].diff()
dataset['displacement'] = np.sqrt(displacement.x**2 + displacement.y**2)
dataset.displacement.iloc[0] = 0 # At the moment of ball_snap or kickoff, acceleration is likely 0

dataset['speed'] = (dataset['displacement'] / .10).values


dataset['a'] = (dataset.speed - dataset.speed.shift(1)) / .10
dataset.a.iloc[0] = 0 # At the moment of ball_snap or kickoff, acceleration is likely 0


dataset['dir_change'] = abs(dataset.dir - dataset.dir.shift(1))
dataset['o_change'] = abs(dataset.dir - dataset.o.shift(1))

dataset['dir_change'] = dataset['dir_change'].fillna(0)
dataset['o_change'] = dataset['o_change'].fillna(0)





dataset.loc[dataset.a < 0, 'acc_type'] = 'decelaration'
dataset.loc[dataset.a == 0, 'acc_type'] = 'no accelaration'
dataset.loc[dataset.a > 0, 'acc_type'] = 'accelaration'


# In[ ]:


num_declaration = calc_num_event(dataset, 'acc_type', 'decelaration', 200, 'num_decelaration')

temp = num_declaration.merge(df_play_list_merge, on=['PlayerKey', 'GameID', 'PlayKey'], how='inner')
final = temp.groupby(['status', 'FieldType'])['num_decelaration'].mean().reset_index()



num_accelaration = calc_num_event(dataset, 'acc_type', 'accelaration', 200, 'num_accelaration')

temp = num_accelaration.merge(df_play_list_merge, on=['PlayerKey', 'GameID', 'PlayKey'], how='inner')
final = final.merge(temp.groupby(['status', 'FieldType'])['num_accelaration'].mean().reset_index(), on=['status', 'FieldType'], how='inner')


num_no_accelaration = calc_num_event(dataset, 'acc_type', 'no accelaration', 200, 'num_no_accelaration')

temp = num_no_accelaration.merge(df_play_list_merge, on=['PlayerKey', 'GameID', 'PlayKey'], how='inner')
final = final.merge(temp.groupby(['status', 'FieldType'])['num_no_accelaration'].mean().reset_index(), on=['status', 'FieldType'], how='inner')


final = final.melt(id_vars=['status', 'FieldType'])


px.bar(final, x='variable', y='value', color='status', facet_col='FieldType', barmode='group')


# We are seeing three-movement metrics by field and injury status. The number of deceleration, number of acceleration is higher in injured player compared to the non-injured player. But we are not seeing any significant differences in these metrics in different field type. 

# # Mean acceleration, speed, distance, direction, orientation, directional changes and orientation changes

# In[ ]:


temp = dataset.groupby(['PlayerKey', 'GameID', 'PlayKey']).agg(mean_dis=('displacement', 'mean'),
     mean_speed = ('speed', 'mean'),
     mean_a = ('a', 'mean'),
     mean_dir = ('dir', 'mean'),
     mean_o = ('o', 'mean'),
     mean_dir_change = ('dir_change', 'mean'), 
     mean_o_change = ('o_change', 'mean')).reset_index()

temp = temp.merge(df_play_list_merge,
                  on=['PlayerKey', 'GameID', 'PlayKey'],
                  how='inner')

temp = temp.groupby(['status', 'FieldType'])[['mean_dis', 'mean_speed',
                                   'mean_a','mean_dir',
                                   'mean_o', 'mean_dir_change',
                                   'mean_o_change']].mean().reset_index()

temp = temp.melt(id_vars=['status', 'FieldType'])

px.bar(temp, x='variable', y='value', color='status', facet_col='FieldType', barmode='group')


# In here, we are seeing very important metrics and we should spend some time thinking about the results. Not all the metrics above are not important. The most important movement metrics are mean direction, mean orientation, mean directional changes and mean orientation changes. 
# 
# We are seeing on natural turf injured player has higher mean direction and orientation but on synthetic turf injured player has less mean direction and orientation compared to the non-injured player. Secondly, we are seeing that injured player has higher directional and orientation changes compared to non-injured players in natural turf and slightly in synthetic turf. 
# 
# Conclusion: We are seeing a high mean in direction, orientation changes. But we are not seeing any difference in different surface.

# # Max acceleration, speed, distance, direction, orientation, directional changes and orientation changes

# In[ ]:


temp = dataset.groupby(['PlayerKey', 'GameID', 'PlayKey']).agg(max_dis=('displacement', 'max'),
     max_speed = ('speed', 'max'),
     max_a = ('a', 'max'),
     max_dir = ('dir', 'max'),
     max_o = ('o', 'max'),
     max_dir_change = ('dir_change', 'max'), 
     max_o_change = ('o_change', 'max')).reset_index()

temp = temp.merge(df_play_list_merge,
                  on=['PlayerKey', 'GameID', 'PlayKey'],
                  how='inner')

temp = temp.groupby(['status', 'FieldType'])[['max_dis', 'max_speed','max_a',
  'max_dir', 'max_o','max_dir_change',
  'max_o_change']].mean().reset_index()

temp = temp.melt(id_vars=['status', 'FieldType'])

px.bar(temp, x='variable', y='value',
       color='status', facet_col='FieldType',
       barmode='group')


# Observations: We are seeing another 7 movement metrics. But among them one is important and that is max_a( max acceleration ). We are seeing max acceleration is way higher in injured player compared to the non-injured player. And the most important thing is that max acceleration also higher in synthetic turf compared to natural turf. 

# # Conclusion
# Finally, we have come to an end. I just want to thank you for reading this notebook until last. We together explored different factors of injuries in the NFL. We have explored how surface, movement metrics and environmental factors affect injuries and most importantly we have provided proposals for reducing injuries based on our findings.
# 
# I hope that these findings and proposal will help the NFL for reducing injuries and make football great sports just it always be. As always thanks for reading!

# References:
# 
# [1] [https://orthoinfo.aaos.org/en/diseases--conditions/turf-toe](https://orthoinfo.aaos.org/en/diseases--conditions/turf-toe)
# 
# [2] [https://en.wikipedia.org/wiki/Cleat_(shoe)](https://en.wikipedia.org/wiki/Cleat_(shoe))
