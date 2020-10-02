#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import sqlite3
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.figure_factory as ff
import ipywidgets as widgets
import math
plotly.offline.init_notebook_mode()
#THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 
#NOTEBOOK WHILE KERNEL IS RUNNING

# Any results you write to the current directory are saved as output.


# Soccer is one of the most exciting sport in the world. I know it might be insulting slightly, hearing the word Soccer, but hence forth I will refer it to as football because as a fan and follower of the sport, I .Based on the popularity, it is the most followed sport. In comparison to other sports, Football is played globally with differents leagues in different countries and there are a lot of people which are involved in this sports. From people working as scouts and coaches from the local clubs upto top players and management who earn million of euro's by being in this profession.
# 
# The data presented here is very exciting, there are so many things which this database contains like Match , Player & League Stats, with so much numbers present and the fact that I have been following football for almost 10 years, I think I should try my hand on this dataset, to excersie my quant skills and see if I can find answers to some things which has always comes up when watching the matches especially in terms of strategy and given the fact that this dataset contains information about those attributes, I think I will have good time going over the data and learn a lot along the way.
# 
# **League Chosen**
# 
# Even though the dataset contains so many League data, I feel going with the English Premier League would be a good idea. It has a large number of followers and the past years has created a lot of excitment with Leicester City winning the 2015-2016 premier league, Manchester City winning the league on the last day in 2011-12, 2013-14 & 2018-19. A lot keeps on happening, lets try and see what the reason behind this phenomenon.
# 
# Some of the question which I will try to answer:
# Which teams have been consistently performing well?
# What are the team attributes which contributes to the result?
# Which kind of players are the most valuable in such a winning team setup?
# Whether the transfer amount which a club pays is worth it or not?
# What kind of stratgies work against each other?
# 
# In the process, I hope to gain valuable analytics skills and even work with tools like SQL, Pandas & Numpy. Exploring this data will be a lot of fun and I will try to update it if and when possible.

# In[ ]:


#Lets set up the connection to database
database = "../input/database.sqlite"
conn = sqlite3.connect(database)


# In[ ]:


#Lets print out the name of the tables in the database
pd.read_sql("SELECT * FROM sqlite_master WHERE type='table';", conn)


# In[ ]:


pd.read_sql("SELECT * FROM League;", conn)


# In[ ]:


#Given that we want to find out infomration of the English Premier Leage, lets get the data using SQL query
epl = pd.read_sql("""SELECT id, league_id, stage, season,date, match_api_id,
                     home_team_api_id, (SELECT team_long_name FROM Team WHERE team_api_id = home_team_api_id) home_team, 
                     away_team_api_id, (SELECT team_long_name FROM Team WHERE team_api_id = away_team_api_id) away_team,
                     home_team_goal, away_team_goal, goal, shoton, shotoff, foulcommit,
                     card, cross, corner, possession, B365H, B365D, B365A
                     FROM Match m
                     WHERE league_id = (SELECT id FROM league WHERE name = 'England Premier League')
                     ORDER BY date;
                """, conn)


# In[ ]:


match_count = epl['id'].size
no_seasons = epl['season'].nunique()
print("The dataframe consists of %d rows spanning %d years of data" % (match_count, no_seasons))
epl['season'].value_counts()


# So based on the above results we can conclude that each season has 380 matches in the premier league. Which serves properly because, there are 20 teams. Each team will play in each round against another team and thus each team plays 38 matches in a seaon per round.
# Total Matches = 38 rounds * 10 matches per round. 
# Thus 380 seems a right number.
# 
# Lets have a look at the data contained in the epl dataset and try to figure out if there is any additional details we can extract from the columns.

# In[ ]:


epl.head()


# Looks the columns from goal - posession consist of xml data, it would be a good idea to inspect this xml file closely, to figure out if we can get more information.

# In[ ]:


epl['shoton'].iloc[0]


# The xml statements have loads of infomration. Having a look at the above result, we can see that it contains detailed information about the goal scored, the time the goal was scored, the team and the player who scored, the shot which he took. Similarly, I had closer look at all the other columns containing the XML data type & they also had very detailed information regarding each incident/event that took place. Let's try and extract at least a high level overview of all these data, so we can get the match statistics. We might have to use an XML parser.

# In[ ]:


def calculate_stats_both_teams(xml_document, home_team, away_team, card_type='y'):
    assert card_type == 'y' or card_type == 'r', "Please enter either y or r"
    tree = ET.fromstring(xml_document)
    stat_home_team = 0
    stat_away_team = 0
    
    #Dealing with card type using the root element & the card type argument
    if tree.tag == 'card':
        for child in tree.iter('value'):
            #Some xml docs have no card_type element in the tree. comment section seems to have that information
            try:
                if child.find('comment').text == card_type:
                    if int(child.find('team').text) == home_team:
                        stat_home_team += 1
                    else:
                        stat_away_team += 1
            except AttributeError:
                #Some values in the xml doc don't have team values, so there isn't much we can do at this stage
                pass
                
        return stat_home_team, stat_away_team
    
    #Lets take the last possession stat which is available from the xml doc
    if tree.tag == 'possession':
        try:
            last_value = [child for child in tree.iter('value')][-1]
            return int(last_value.find('homepos').text), int(last_value.find('awaypos').text)
        except:
            return None, None
    
    #Taking care of all other stats by extracting based on the home team & away team api id's
    for team in [int(stat.text) for stat in tree.findall('value/team')]:
        if team == home_team: 
            stat_home_team += 1
        else:
            stat_away_team += 1
    return stat_home_team, stat_away_team


# In[ ]:


epl[['on_target_shot_home_team','on_target_shot_away_team']] = epl[['shoton','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['shoton'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")  
epl[['off_target_shot_home_team','off_target_shot_away_team']] = epl[['shotoff','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['shotoff'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand") 
epl[['foul_home_team','foul_away_team']] = epl[['foulcommit','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['foulcommit'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")  
epl[['yellow_card_home_team','yellow_card_away_team']] = epl[['card','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['card'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")
epl[['red_card_home_team','red_card_away_team']] = epl[['card','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['card'],x['home_team_api_id'],x['away_team_api_id'], card_type='r'), axis = 1,result_type="expand")  
epl[['crosses_home_team','crosses_away_team']] = epl[['cross','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['cross'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")  
epl[['corner_home_team','corner_away_team']] = epl[['corner','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['corner'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")  
epl[['possession_home_team','possession_away_team']] = epl[['possession','home_team_api_id','away_team_api_id']].apply(lambda x: calculate_stats_both_teams(x['possession'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")


# In[ ]:


epl.describe()


# Looking at the last two columns we can see that there are about 6 games whose possession stats were not available. Thats not too bad.

# In[ ]:


epl.loc[epl['possession_home_team'].isnull()]


# So there are 5 matches which have no data at all. There is one match which we can still extract the data. I'll have a look at it later and try to improve my function.

# In[ ]:


#Function to calculate the outcome of the match with respect to the target team
def get_result(home,away, home_goal, away_goal, target):
    if home_goal == away_goal:
        return 'D'
    elif(home_goal > away_goal and target == home) or (home_goal<away_goal and target==away):
        return 'W'
    elif(home_goal > away_goal and target == away) or (home_goal<away_goal and target==home):
        return 'L'


# In[ ]:


#drop the XML columns from the datsaet
epl = epl.drop(['goal','shoton','shotoff','foulcommit','card','cross','corner','possession'], axis=1)


# In[ ]:


#remove the time from the date in the datset
epl['date'] = epl['date'].apply(lambda x: x.split()[0])
#Subset the columns concerning the stats of the home and away team & rename the column names
away_teams = epl[['id','league_id','stage','season','date','match_api_id','home_team_api_id','home_team','away_team_api_id','away_team',
                  'home_team_goal','away_team_goal','B365H','B365D','B365A','on_target_shot_away_team','off_target_shot_away_team',
                 'foul_away_team','yellow_card_away_team','red_card_away_team','crosses_away_team','corner_away_team','possession_away_team'
                 ]].copy(deep=True)
away_teams['team_name_stats'] = away_teams['away_team']
home_teams = epl[['id','league_id','stage','season','date','match_api_id','home_team_api_id','home_team','away_team_api_id','away_team',
                  'home_team_goal','away_team_goal','B365H','B365D','B365A','on_target_shot_home_team','off_target_shot_home_team',
                 'foul_home_team','yellow_card_home_team','red_card_home_team','crosses_home_team','corner_home_team','possession_home_team'
                 ]].copy(deep=True)
home_teams['team_name_stats'] = home_teams['home_team']
home_teams.columns = ['id', 'league_id', 'stage', 'season', 'date', 'match_api_id','home_team_api_id', 'home_team', 'away_team_api_id', 'away_team',
       'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A','on_target_shots', 'off_target_shots', 'foul_home', 'yellow_card', 'red_card',
       'crosses', 'corners', 'possession','team_name_stats']
away_teams.columns = ['id', 'league_id', 'stage', 'season', 'date', 'match_api_id', 'home_team_api_id', 'home_team', 'away_team_api_id', 'away_team',
       'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A', 'on_target_shots', 'off_target_shots', 'foul_home', 'yellow_card', 'red_card',
       'crosses', 'corners', 'possession', 'team_name_stats']

#Merge both the home & away datasets, order them by the date & the home team in order to get team stats consecutively
epl_team_stats = pd.concat([home_teams,away_teams]).sort_values(by=['date','home_team']).copy(deep=True)
epl_team_stats['outcome'] = epl_team_stats.apply(
    lambda x: get_result(x['home_team'],x['away_team'],x['home_team_goal'],x['away_team_goal'],x['team_name_stats']),
    axis = 1
)
epl_team_stats['points_earned'] = epl_team_stats['outcome'].apply(lambda x: 3 if x=='W' else 0 if x=='L' else 1)
epl_team_stats['date'] = epl_team_stats.date.map(lambda x: datetime.strptime(x,"%Y-%m-%d"))
epl_team_stats = epl_team_stats.reset_index()


# In[ ]:


epl_team_stats.head()


# In[ ]:


#Calculate cumulative points, Goals for, Goal against & Goal difference for each team and at each stage of the season 
epl_team_stats['cum_points']=epl_team_stats.groupby(['season','team_name_stats']).points_earned.cumsum()
epl_team_stats['goal_scored'] = epl_team_stats.apply(
    lambda x: x['home_team_goal'] if x['team_name_stats']==x['home_team'] else x['away_team_goal'], axis=1)
epl_team_stats['goal_conceded'] = epl_team_stats.apply(
    lambda x: x['away_team_goal'] if x['team_name_stats']==x['home_team'] else x['home_team_goal'], axis=1)
epl_team_stats['won'] = epl_team_stats.apply(lambda x:1 if x['outcome']=='W' else 0,axis=1)
epl_team_stats['draw'] = epl_team_stats.apply(lambda x:1 if x['outcome']=='D' else 0,axis=1)
epl_team_stats['loss'] = epl_team_stats.apply(lambda x:1 if x['outcome']=='L' else 0,axis=1)
epl_team_stats['W'] = epl_team_stats.groupby(['season','team_name_stats']).won.cumsum()
epl_team_stats['D'] = epl_team_stats.groupby(['season','team_name_stats']).draw.cumsum()
epl_team_stats['L'] = epl_team_stats.groupby(['season','team_name_stats']).loss.cumsum()
epl_team_stats['GF'] = epl_team_stats.groupby(['season','team_name_stats']).goal_scored.cumsum()
epl_team_stats['GA'] = epl_team_stats.groupby(['season','team_name_stats']).goal_conceded.cumsum()
epl_team_stats['GF'] = epl_team_stats.groupby(['season','team_name_stats']).goal_scored.cumsum()
epl_team_stats['GA'] = epl_team_stats.groupby(['season','team_name_stats']).goal_conceded.cumsum()
epl_team_stats['GD'] = epl_team_stats.apply( 
    lambda x: x['GF']-x['GA'], axis=1
)
epl_team_stats = epl_team_stats.drop(["won","draw","loss"],axis=1)


# In[ ]:


epl_team_stats.head(5)


# In[ ]:


#Lets calculate the table position of each team at each stage of the tournament. 
#The premier league ranks according to points followed by goal difference and then goals for. 
#So at the start of the season there is are multiple ties, & the minimum rank is given for the teams in the same position
epl_team_stats['rank_calc'] = epl_team_stats.apply(lambda x: x['cum_points']*10000 +x['GD']*50 +x['GF'],axis=1)
epl_team_stats['table_position'] = epl_team_stats.groupby(['stage','season'])['rank_calc'].rank(ascending=False, method='min').astype(int)
epl_team_stats = epl_team_stats.drop('rank_calc',axis=1)
epl_team_stats.head(20).sort_values(['table_position','team_name_stats'])[['table_position','team_name_stats','W','D','L','GF','GA','GD','cum_points']]


# <img src="https://i.imgur.com/A5oJI6J.png" width=500px></img>

# Seems like a proper match for the first round of 2008/2009 season
# Lets have a look at the final round of the 2015/2016 season. Just to be sure whether everything is working right

# In[ ]:


epl_team_stats.tail(20).sort_values(['table_position','team_name_stats'])[['table_position','team_name_stats','W','D','L','GF','GA','GD','cum_points']]


# <img src="https://i.imgur.com/BFatIV5.png" width=500px></img>

# Aha! Everything is working out fine. Even the 4th & 5th position tie was properly resolved. The extreme ends of the dataset are verified. I am glad that this dataset is pretty clean.

# Lets retrive the team attributes from the database and join them with the team in each row

# In[ ]:


team_attributes = pd.read_sql("""WITH epl_team AS(
                SELECT DISTINCT(m.home_team_api_id) 
                FROM  match m
                WHERE league_id = (SELECT id FROM league l WHERE l.name='England Premier League')
               )
               SELECT (SELECT t.team_long_name FROM team t where t.team_api_id = ta.team_api_id) AS team_name, ta.* 
               FROM Team_Attributes ta WHERE ta.team_api_id IN (SELECT * FROM epl_team);""",
            conn)


# In[ ]:


team_attributes.head(10)


# Based on the initial glance of team & date, there might be only 6 seasons data for team attributes in contrast to 8 seasons of match data. The data must have been collected at a random match day in each season.

# In[ ]:


print("There are %d unique teams in the attributes dataset which is also present in the EPL dataset we have extracted"%sum(team_attributes.team_name.unique() == epl_team_stats.sort_values('home_team').home_team.unique()))


# In[ ]:


team_attributes.pivot_table( columns='team_name', values='team_api_id', aggfunc=len)


# <p>Based on the output of the above two outputs, it seems that there are team attribute data of 6 seasons each.</p>

# In[ ]:


#Lets create the season attribute to match the team attributes
team_attributes['date'] = team_attributes.date.map(lambda x: datetime.strptime(x.split()[0],'%Y-%m-%d'))
bins = [datetime(2008,8,1),datetime(2009,6,15),
       datetime(2009,8,1),datetime(2010,6,15),
       datetime(2010,8,1),datetime(2011,6,15),
       datetime(2011,8,1),datetime(2012,6,15),
       datetime(2012,8,1),datetime(2013,6,15),
       datetime(2013,8,1),datetime(2014,6,15),
       datetime(2014,8,1),datetime(2015,6,15),
       datetime(2015,8,1),datetime(2016,6,15)]
team_attributes['season_bin'] = pd.cut(team_attributes['date'], bins)
team_attributes['season'] = team_attributes.season_bin.map(lambda x: "".join([x.left.strftime('%Y-%m-%d').split('-')[0], "/", x.right.strftime('%Y-%m-%d').split('-')[0]]))
team_attributes = team_attributes.drop("season_bin", axis=1)


# In[ ]:


def graph_update_team_and_season_selection(season, team):
    data_viz_season_performance = epl_team_stats[epl_team_stats.team_name_stats.eq(team) & 
                                  epl_team_stats.season.eq(season)]

    #Trace for points gained during the selected season
    trace_point_progression = go.Scatter(
        x=data_viz_season_performance.date, 
        y=data_viz_season_performance.cum_points,
        mode = "markers+lines",
        marker={'color': 'green'},
        name = "Point Progression"
    )
    #Trace for the Goal for during the selected team and season
    trace_GF_progression = go.Scatter(
        x=data_viz_season_performance.date,
        y=data_viz_season_performance.GF,
        mode = "markers+lines",
        marker={'color':'blue'},
        name="Goal For",
        yaxis="y2"
    )
    #Trace for the Goal Allowed during the selected team and season
    trace_GA_progression = go.Scatter(
        x=data_viz_season_performance.date,
        y=data_viz_season_performance.GA,
        mode = "markers+lines",
        marker={'color':'red'},
        name="Goal Against",
        yaxis="y2"
    )

    data1 = [trace_point_progression, trace_GF_progression, trace_GA_progression]
    layout1= go.Layout(
        title="".join([team,' - ',season]), 
        xaxis={'title':'Date'}, 
        yaxis={'title':'Points','side':'left'},
        yaxis2={'title':'Goals','overlaying':'y','side':'right'},
    )
    figure1 = go.FigureWidget(data = data1, layout=layout1)

    data_viz_team_attributes = team_attributes.loc[team_attributes.team_name.eq(team) & team_attributes.season.eq(season),
                                            ['team_name', 'buildUpPlaySpeed', 'buildUpPlayDribbling','buildUpPlayPassing',
                                             'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting', 
                                             'defencePressure', 'defenceAggression', 'defenceTeamWidth', 'season']]
    try:
        trace_team_attributes = go.Scatterpolar(r=data_viz_team_attributes.values.tolist()[0][1:-1],
                                            theta=data_viz_team_attributes.columns.tolist()[1:-1],
                                            fill='toself'
                                            )
    except IndexError:
        trace_team_attributes = go.Scatterpolar(r=[0*9],
                                            theta=data_viz_team_attributes.columns.tolist()[1:-1],
                                            fill='toself'
                                            )
    data2 = [trace_team_attributes]
    layout2= go.Layout( polar = dict(
                                 radialaxis = dict(
                                 visible = True,
                                 range = [25, 80]
                                                  )
                                ),showlegend = False
                 )
    figure2 = go.FigureWidget(data = data2, layout=layout2)
    display(widgets.VBox([widgets.VBox([figure1, figure2])]))
    

season = widgets.Dropdown(
    options=list(epl_team_stats.season.unique()),
    value='2015/2016',
    description='Season',
    disabled=False,
)

team = widgets.Dropdown(
    options=epl_team_stats.sort_values('team_name_stats').team_name_stats.unique(),
    value='Leicester City',
    description='Team',
    disabled=False,
)
widgets.interactive(graph_update_team_and_season_selection, season=season, team=team)


# In[ ]:


#Create pivot table for calculating mean points gained during season
epl_mean_point = epl_team_stats.pivot_table(index="team_name_stats", 
                                            columns="season", 
                                            values="points_earned", 
                                            aggfunc=np.mean)
#Calculate the mean across th entire dataset
epl_mean_point["All"] = epl_mean_point.mean(axis=1)

#Array of team points average as a list for the heatmap
z= epl_mean_point.fillna(0).values
# Seasons as columns 
x= epl_mean_point.columns.tolist()
#team name on y-axis
y= epl_mean_point.index.tolist()

#List to store the hover-info
epl_mean_text = []
#list to store the annotated text
epl_mean_annotated_text = []
#loop to extract the details form the epl_mean_stats table to give end of the season information 
for team in range(len(y)):
    team_season_list = []
    team_season_annotated = []
    for year in range(len(x)-1):
        if z[team][year] == 0:
            #team was not present in the epl for that season
            team_season_list.append('')
            team_season_annotated.append('')
        else:
            #subset the row with the selected row and season
            selected_row = epl_team_stats[epl_team_stats.stage.eq(38)&
                                          epl_team_stats.team_name_stats.eq(y[team])&
                                          epl_team_stats.season.eq(x[year])]
            team_season_list.append("".join(
                ["Team: ",y[team],"<br>",
                 "Season: ",x[year],",<br>",
                 "Total Points: ", str(selected_row.cum_points.values[0]),"<br>",
                 "Average Pts per game", str(round(z[team][year], 3))
                ]
            ))
            team_season_annotated.append(str(int(selected_row.table_position)))
    team_season_list.append("".join(["Avg over all seasons: ", str(round(z[team][year+1],4))]))
    team_season_annotated.append('')
    epl_mean_text.append(team_season_list)
    epl_mean_annotated_text.append(team_season_annotated)
    
ann_heatmap = ff.create_annotated_heatmap(x=x, y=y, z=z, annotation_text=epl_mean_annotated_text, text=epl_mean_text,
                                          colorscale='Jet', font_colors =['Black'], hoverinfo='text', showscale= True
                                         )
ann_heatmap.layout.height = 1250
ann_heatmap.layout.yaxis.automargin = True
iplot(ann_heatmap)


# Looking at the heatmap, there are some observations we can put forth.
# - Teams having point avergae of less than 1 have very high chance of relagation, having less than 0.9(i.e <34) they are always relagated. Interestingly, in the 2010-11 season, teams having 39 points were also relegated.
# - There are 9 teams which have maintained their status in the Premier League over the course of 8+ years.
# - Arsenal have been the most consistent performers during this periods, though never winning the league in the same period.
# - Manchester united have been the most successful during the 8 seasons, having won 3 league titles. After the 2012-13 season, there has been a drop off in their league consistency, which we can say might be due to the departure of their coach.
# - Following United, Chelsea and Manchester City have been the most sucessfull, each winning the league title twice.
# - At the other end of the spectrum, Sunderland has been able to maintain their place in premier league even after flirting with relagation during the 8 year period.

# In[ ]:




