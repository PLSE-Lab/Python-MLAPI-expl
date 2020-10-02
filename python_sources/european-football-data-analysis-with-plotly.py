#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sqlite3
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from ipywidgets import widgets, interactive, VBox, HBox
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)


# In[ ]:


# connection
conn = sqlite3.connect('../input/soccer/database.sqlite') #../input/soccer/
df= pd.read_sql("select * from sqlite_master where type='table';", conn)


# In[ ]:


df


# In[ ]:


sql_seq = pd.read_sql_query('select * from sqlite_sequence', conn)
player_att = pd.read_sql_query('select * from Player_Attributes', conn)
player =  pd.read_sql_query('select * from Player', conn)
match  =  pd.read_sql_query('select * from Match', conn)
league =  pd.read_sql_query('select * from League', conn)
country_name = pd.read_sql_query('select * from Country', conn)
team =  pd.read_sql_query('select * from Team', conn)
team_att = pd.read_sql_query('select * from Team_attributes', conn)


# In[ ]:


match.head()


# In[ ]:


match.shape


# In[ ]:


list(match.columns)


# In[ ]:


match = match[['id', 'country_id', 'league_id', 'season', 'stage', 'date', 'match_api_id', 'home_team_api_id', 
               'away_team_api_id', 'home_team_goal', 'away_team_goal']]


# In[ ]:


match.isna().sum()


# In[ ]:


info = league.merge(match, left_on='country_id', right_on='country_id')
info.head()


# In[ ]:


info.columns


# In[ ]:


team.head()


# In[ ]:


team.isna().sum()


# In[ ]:


info['home_team_name'] = info['home_team_api_id'].map(team.set_index('team_api_id')['team_long_name'])
info['away_team_name'] = info['away_team_api_id'].map(team.set_index('team_api_id')['team_long_name'])
info.drop(columns=['home_team_api_id', 'away_team_api_id', 'country_id'], inplace=True)
info.head()


# In[ ]:


layout = go.Layout(height=600, width=800)


# In[ ]:


select_season = widgets.Dropdown(options=info['season'].unique(), description = 'Season')
fig  = go.FigureWidget(make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]]), layout=layout)

def response(value):
    lab = info['name'].unique().tolist()
    val_home = info.loc[info['season'] == value].groupby(['name'])['home_team_goal'].agg('sum')
    val_away = info.loc[info['season'] == value].groupby(['name'])['away_team_goal'].agg('sum')
    
    fig.add_trace(go.Pie(labels=lab, values=val_home, name='Home goal'), row=1, col=1)
    fig.add_trace(go.Pie(labels=lab, values=val_away, name='Away goal'), row=1, col=2)
    fig.update_traces(textinfo='value', hole=0.4)

    fig.update_layout(title='League goals per season',
                      annotations=[dict(text='Home Goals', x=0.17, y=0.5, showarrow=False),
                                  dict(text='Away Goals', x=0.83, y=0.5,  showarrow=False)])
   
   

VBox((fig, interactive(response, value=select_season)))
# widgets.interact(response, value=select_season)


# In[ ]:


select_sea_h= widgets.Dropdown(options=info['season'].unique(), description='Season')
select_league_h = widgets.Dropdown(options=info['name'].unique(), description='League')
fig_1 = go.FigureWidget(data=[{'type':'bar'}])


def response_1(season, league_name):
    home = info.loc[(info['season']==season) & 
                    (info['name']==league_name)].groupby(['home_team_name'])['home_team_goal'].sum().nlargest(5)
    val_h, lab_h = home.values, home.index.values
    fig_1.update_traces(go.Bar(x=lab_h, y=val_h),
                        marker=(dict(color=['olive', 'purple', 'forestgreen', 'chocolate','teal'])))
    fig_1.update_layout(title='Home Goals:- Top 5 Teams')
        
HBox((fig_1, interactive(response_1, season=select_sea_h, league_name=select_league_h)))


# In[ ]:


select_sea_a= widgets.Dropdown(options=info['season'].unique(), description='Season')
select_league_a = widgets.Dropdown(options=info['name'].unique(), description='League')
fig_2 = go.FigureWidget(data=[{'type':'bar'}])

def response_2(season, league_name):
    away = info.loc[(info['season']==season) & 
                    (info['name']==league_name)].groupby(['away_team_name'])['away_team_goal'].sum().nlargest(5)
    val_a, lab_a = away.values, away.index.values
    fig_2.update_traces(go.Bar(x=lab_a, y=val_a),
                        marker=(dict(color=['darkcyan', 'fuchsia', 'mediumslateblue', 'lightseagreen','darkorchid'])))
    fig_2.update_layout(title='Away Goals:- Top 5 Teams')
    
HBox((fig_2, interactive(response_2, season=select_sea_a, league_name=select_league_a)))


# In[ ]:


player.head()


# In[ ]:


player_att.head()


# In[ ]:


player_info = player_att.merge(player, how='left', left_on='player_api_id', right_on='player_api_id') 
player_info.head()


# In[ ]:


player_info.columns


# In[ ]:


player_info.isna().sum()


# In[ ]:


player_info.dropna(inplace=True)
player_info.shape


# In[ ]:


select_feat = widgets.Dropdown(options=['overall_rating', 'potential','crossing', 'finishing', 'heading_accuracy',
            'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy','long_passing', 'ball_control', 
            'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
            'strength', 'long_shots', 'aggression', 'interceptions', 'positioning','vision', 'penalties', 'marking',
            'standing_tackle', 'sliding_tackle'], description='Select')

fig_3 = go.FigureWidget(data=[{'type':'bar'}])

def response_3(feature):
    x = player_info.groupby(['player_name'])[feature].mean().nlargest(10)
    val = list(x.values)
    lab = list(x.index.values)
    fig_3.update_traces(go.Bar(x=lab, y=val, marker=dict(color=['purple','orange', 'silver', 'violet', 'indigo', 'red', 
                                                                'green', 'cyan', 'blue','maroon', ])))
    fig_3.update_layout(title='Top 10 players by average ***Select a feature from the thirty features.***', )
    

VBox((fig_3, interactive(response_3, feature=select_feat)))


# In[ ]:


player_info.columns


# In[ ]:


select_feat_gk = widgets.Dropdown(options=['gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning','gk_reflexes'],
                                 description='Select')

fig_4 = go.FigureWidget(data=[{'type':'bar'}])

def response_4(feature):
    x = player_info.groupby(['player_name'])[feature].mean().nlargest(10)
    val = list(x.values)
    lab = list(x.index.values)
    fig_4.update_traces(go.Bar(x=lab, y=val, marker=dict(color=['purple','orange', 'silver', 'violet', 'indigo', 'red', 
                                                                'green', 'cyan', 'blue','maroon'])))
    fig_4.update_layout(title='Top 10 Goalkeepers by average ***Select a feature.***')
    
VBox((fig_4, interactive(response_4, feature=select_feat_gk)))


# In[ ]:


team_att.columns


# In[ ]:


team_att.isna().sum()


# In[ ]:


team_att.iloc[0]


# In[ ]:


team_att.shape


# In[ ]:


team_att['team_name'] = team_att['team_api_id'].map(team.set_index('team_api_id')['team_long_name'])
# map team with league
temp = info[['name','home_team_name']].drop_duplicates()
team_att['league'] = team_att['team_name'].map(temp.set_index('home_team_name')['name'])


# In[ ]:


select_team_feat = widgets.Dropdown(options=['buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing', 
                                             'chanceCreationCrossing', 'chanceCreationShooting', 'defencePressure',
                                             'defenceAggression','defenceTeamWidth'], description='Team feature')
select_team_league = widgets.Dropdown(options=team_att['league'].unique(), description='League')
fig_5 = go.FigureWidget(data=[{'type':'bar'}])

def response_5(feature, league_name):
    x = team_att.loc[team_att['league']==league_name].groupby(['team_name'])[feature].mean().nlargest(5)
    val = list(x.values)
    lab = list(x.index.values)
    fig_5.update_traces(go.Bar(x=lab, y=val, marker=dict(color=['mediumseagreen', 'crimson', 
                                                                'darkgoldenrod','mediumorchid', 'peachpuff'])))
    fig_5.update_layout(title='Top 5 teams')
    

HBox((fig_5, interactive(response_5, feature=select_team_feat, league_name=select_team_league)))


# In[ ]:





# In[ ]:


import ipywidgets
ipywidgets.IntSlider()


# In[ ]:




