#!/usr/bin/env python
# coding: utf-8

# <img src="https://static.dezeen.com/uploads/2016/08/designstudiopremier-league-rebrand-relaunch-logo-design-barclays-football_dezeen_slideshow-a.jpg" alt="Drawing" style="width: 200px;"/>

# # <font color=blue>Data Analysis<font/>
# # <font color=blue>English Premier League - Seasons 2009 - 2019<font/>

# ### We are analysing the data of the past 10 seasons of English Premier League.<br>Our aim is to answer the 25 questions associated with the dataset in the link below.<br>https://www.kaggle.com/aj7amigo/english-premier-league-data-2009-2019

# # <font color=brown>Preparing the data for Analysis<font/>

# ## a. Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ## b. Load the data as a pandas data frame and make a copy

# In[ ]:


ten_seasons_data = pd.read_csv('../input/english-premier-league-data-2009-2019/English_Premier_League_data_2009-2019.csv')
decade_data = ten_seasons_data.copy()
decade_data.head()


# ## c. Check if any data is missing

# In[ ]:


decade_data.describe()


# ### Check for null values

# In[ ]:


decade_data.isnull().sum()


# ## d. Check the datatypes of each column

# In[ ]:


decade_data.dtypes


# #### We change the datatype of the column 'Date' into a pandas datetime format

# In[ ]:


decade_data['Date'] = pd.to_datetime(decade_data['Date'], format="%Y/%m/%d")


# ## e. Drop irrelevant columns and rename other columns to understand the contents from column title

# In[ ]:


decade_data = decade_data.drop(['Div'], axis=1) #dropping first column 'Div' as value is always E0

decade_data.columns = ['Date','HomeTeam','AwayTeam','FT_Home_Goal','FT_Away_Goal','FT_Result','HT_Home_Goal','HT_Away_Goal',
                        'HT_Result','Referee','H_Shots','A_Shots','H_Shots_Target','A_Shots_Target','H_Foul',
                        'A_Foul','H_Corner','A_Corner','H_Yellow','A_Yellow','H_Red','A_Red']

decade_data.head()


# ## <font color=green>The data is now well prepared to proceed with analysis <font/>

# # <font color=brown>Data Analysis<font/>

# ## <font color=blue>1. Howmany matches were played in EPL between 2009 - 2019?<br>$\;\;$ Howmany teams played in the EPL during these 10 seasons?<br> $\;\;$ Which are they?<font/>

# #### <font color=violet>We basically have to get the number of entries in any column.<font/>

# In[ ]:


total_matches = decade_data['Date'].count()
print('Total matches played during the 10 seasons is : ' +str(total_matches))


# #### <font color=violet>Since every team has played home and away, we need to focus on only one of the 2 columns, HomeTeam and AwayTeam.<br>Find the unique entry in the HomeTeam Column and count the number of unique entries.<font/>

# In[ ]:


all_teams = decade_data['HomeTeam'].unique()
all_teams_count = decade_data['HomeTeam'].nunique()
print('Total teams which played in the EPL during the ten seasons : '+str(all_teams_count))
print('\n')
print('The teams are : \n'+str(all_teams))


# ## <font color=blue>2. Howmany games were played by each team in these 10 seasons?<br>$\;\;$ Plot it as a bar chart.<font/>

# #### <font color=violet>We need to get the value counts of each team in Home and Away games and add it to get the total matches played by each team.<font/>

# In[ ]:


total_games_each_team = decade_data['HomeTeam'].value_counts() + decade_data['AwayTeam'].value_counts()
each_team_games = pd.DataFrame(total_games_each_team).sort_index(axis = 0) 
each_team_games.columns = ['Total Games']
each_team_games


# In[ ]:


each_team_games.plot(kind='bar',color='brown', legend=False, figsize=(20,10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Teams',fontsize=20)
plt.ylabel('Total games (EPL) in seasons 2009 - 2019',fontsize=20)
plt.title('Total games per team during 2009 - 2019',fontsize=20, color='red')
plt.show()


# ## <font color=blue>3. Which teams played all the seasons in EPL from 2009 - 2019?<br>$\;\;$ Which teams played only 1 season in EPL between 2009 - 2019?<font/>

# #### <font color=violet>To find this, we need the following info.<br>Total games per team per season is 38.<br><br><font/><font color=violet>So, if a team played all 10 seasons, the total games played will be 380.<br>We search for all teams who played 380 games.<font/>

# In[ ]:


ten_season_teams = each_team_games[each_team_games['Total Games']==380]
ten_season_teams


# #### <font color=violet>Also, if a team played only one season, total games played is 38.<br>So, we search for all teams who played only 38 games.<font/>

# In[ ]:


one_season_teams = each_team_games[each_team_games['Total Games']==38]
one_season_teams


# ## <font color=blue>4. Howmuch percent of the total matches were won by the home team, away team or draw?<br>$\;\;$ Show the data in a pie chart.<font/>

# #### <font color=violet>To solve this, we focus on the column 'FT_Result' which contains info on the match result.<br>We look for the number of home team wins, away team wins and draws from this column.<font/>

# In[ ]:


all_matches_results = pd.DataFrame(decade_data['FT_Result'].value_counts())
all_matches_results


# In[ ]:


labels = ['Home Team Wins','Away Team Wins','Draw']
all_matches_results.plot(kind='pie', y = 'FT_Result', autopct='%1.1f%%', 
 startangle=180, shadow=False, labels=labels, legend = False, fontsize=14, figsize=(5,5))
plt.title('Percentage share of match results',fontsize=20, color='red')


# ## <font color=blue>5. Howmany referees officiated the EPL matches between 2009 and 2019?<br>$\;\;$ List them with the total number of matches officiated.<br>$\;\;$ Show the data as a horizontal bar plot.<font/>

# #### <font color=violet>We focus on the column 'Referee' which contains info on who officiated the match.<br>We look for the referee names and number of matches officiated by each of them.<font/>

# In[ ]:


referees_count = decade_data['Referee'].nunique()
print('Number of referees who officiated the EPL matches between 2009 and 2019 : '+str(referees_count))
all_referees = pd.DataFrame(decade_data['Referee'].value_counts()).sort_index(axis = 0)
all_referees


# In[ ]:


all_referees.plot(kind='barh',color='orange', legend=False, figsize=(15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Referees',fontsize=14)
plt.xlabel('Total games officiated in seasons 2009 - 2019',fontsize=14)
plt.title('Number of games officiated by referees during 2009 - 2019',fontsize=20, color='red')
plt.show()


# ## <font color=blue>6. In howmany games did the teams loosing at half time come back to win the game at full time (comeback wins)? List all of them.

# #### <font color=violet>We need to focus on two columns 'HT_Result' and 'FT_Result'.<br>We need to look for two sets of data.<br>1. Away team winning at half time and Home team wins at full time.<br>2. Home team winning at half time and Away team wins at full time<font/>

# In[ ]:


come_backs = decade_data[((decade_data['HT_Result']=='A') & (decade_data['FT_Result']=='H'))
                         | 
                         ((decade_data['HT_Result']=='H') & (decade_data['FT_Result']=='A'))]

come_back_wins = come_backs.shape[0]

print('Number of games in which teams loosing at half time come back to win the game at full time : '+str(come_back_wins))


# In[ ]:


#come_backs
come_backs.head()


# ## <font color=blue>7. Sort the comeback wins based on year.<font/>

# #### <font color=violet>We need to group the above findings about comeback wins on a yearly basis.<font/>

# In[ ]:


come_backs_year_sort = pd.DataFrame()
come_backs_year_sort['comeback_wins_per_year'] = come_backs['FT_Result'].groupby([come_backs.Date.dt.year]).agg('count')
come_backs_year_sort


# ## <font color=blue>8. Sort the comeback wins based on season.<br>$\;\;$ Make a donut plot of comeback wins split among seasons.<br>$\;\;$ Highlight the season with most comeback wins.<font/>

# #### <font color=violet>Since we don't have a info about the season in the data frame, we use the following info.<br>Each EPL season has 380 games.<br>So we can make a new column called season which changes after every 380 games.<font/>

# In[ ]:


season_start=9
season_end=10
season_list = []

for x in range (10):
    for y in range (380):
        season_list.append(('0'+str(season_start)+'-'+str(season_end))[-5:]) # the value '0' is added to make 9 as 09.
    season_start = season_start + 1
    season_end = season_end + 1


# #### <font color=violet>In the above step, we made an empty list and added the seasons into it changing every 380 games.<br>We will convert this list into a pandas dataframe and later concatenate this data frame with the decade_data.<br><font/>

# In[ ]:


season_df = pd.DataFrame({'Season':season_list})

decade_data_by_seasons = pd.concat([season_df,decade_data], axis=1)
decade_data_by_seasons.head()


# #### <font color=violet>We group the comeback wins based on season.<font/>

# In[ ]:


come_backs_updated = decade_data_by_seasons[
                        ((decade_data_by_seasons['HT_Result']=='A') & (decade_data_by_seasons['FT_Result']=='H'))
                         | 
                         ((decade_data_by_seasons['HT_Result']=='H') & (decade_data_by_seasons['FT_Result']=='A'))]


come_backs_season_sort = pd.DataFrame()
come_backs_season_sort['comeback_wins_per_season'] = come_backs['FT_Result'].groupby([come_backs_updated.Season]).agg('count')
come_backs_season_sort


# #### <font color=violet>We now split the total 153 comeback wins among 10 seasons in a donut plot.<br>We use explode method in pie chart to highlight the season with most comeback wins.<font/>

# In[ ]:


def value_and_percentage(x): 
    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)


plt.figure(figsize=(9,9))
values = come_backs_season_sort['comeback_wins_per_season']
labels = decade_data_by_seasons['Season'].unique()
total = np.sum(values)
colors = ['#8BC34A','Pink','#FE7043','Turquoise','#D4E157','Grey','#EAB300','#AA7043','Violet','Orange']
plt.pie (values , labels= labels , colors= colors , 
         startangle=45 , autopct=value_and_percentage, pctdistance=0.85, 
         explode=[0,0,0,0.1,0,0,0,0,0,0] )
my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Comeback wins split among seasons',fontsize=20, color='red')
plt.show()


# ## <font color=blue>9. Generate a data frame with the following info,<br>$\;\;\;\;$ a. Number of games per team.<br>$\;\;\;\;$ b. Home wins per team.<br>$\;\;\;\;$ c. Home defeats per team.<br>$\;\;\;\;$ d. Away wins per team.<br>$\;\;\;\;$ e. Away defeats per team.<br>$\;\;\;\;$ f. Number of draw per teams (home and away combined).<br>$\;\;\;\;$ g. Total points in EPL across 10 seasons.<font/>

# #### <font color=violet>a. We already have the total games played by each team in the data frame each_team_games.<font/>

# #### No need of any computation here. We can call the dataframe each_team_games when needed.

# #### <font color=violet>b. We group HomeTeam when Full time result was home team win.<font/>

# In[ ]:


home_team_wins = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='H']
                  .groupby([decade_data_by_seasons.HomeTeam]).agg('count'))[['FT_Result']]
home_team_wins.columns = ['Home_Wins']
home_team_wins.index.names = ['Team']


# #### <font color=violet>c. We group HomeTeam when Full time result was away team win.<font/>

# In[ ]:


home_team_loss = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='A']
                  .groupby([decade_data_by_seasons.HomeTeam]).agg('count'))[['FT_Result']]
home_team_loss.columns = ['Home_Loss']
home_team_loss.index.names = ['Team']


# #### <font color=violet>d. We group AwayTeam when Full time result was away team win.<font/>

# In[ ]:


away_team_wins = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='A']
                  .groupby([decade_data_by_seasons.AwayTeam]).agg('count'))[['FT_Result']]
away_team_wins.columns = ['Away_Wins']
away_team_wins.index.names = ['Team']


# #### <font color=violet>e. We group AwayTeam when Full time result was home team win.<font/>

# In[ ]:


away_team_loss = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='H']
                  .groupby([decade_data_by_seasons.AwayTeam]).agg('count'))[['FT_Result']]
away_team_loss.columns = ['Away_Loss']
away_team_loss.index.names = ['Team']


# #### <font color=violet>f. We group HomeTeam when Full time result was draw and group AwayTeam when Full time result was draw. Finally, we add them both.<font/>

# In[ ]:


home_team_draw = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='D']
                  .groupby([decade_data_by_seasons.HomeTeam]).agg('count'))[['FT_Result']]
home_team_draw.columns = ['Home_Draw']
home_team_draw.index.names = ['Team']


away_team_draw = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='D']
                  .groupby([decade_data_by_seasons.AwayTeam]).agg('count'))[['FT_Result']]
away_team_draw.columns = ['Away_Draw']
away_team_draw.index.names = ['Team']


total_draw_matches = home_team_draw['Home_Draw'] + away_team_draw['Away_Draw']
home_and_away_draws = pd.DataFrame(total_draw_matches)
home_and_away_draws.columns = ['Draws-Home_and_Away']


# #### <font color=violet>g. To calculate this, we need the following info.<br>$\;\;\;\;$Each win gives 3 points, each draw gives 1 point and each loss gives 0 points.<font/>

# In[ ]:


total_wins = home_team_wins['Home_Wins'] + away_team_wins['Away_Wins']
home_and_away_wins = pd.DataFrame(total_wins)
home_and_away_wins.columns = ['Wins-Home_and_Away']

total_points_decade = (home_and_away_wins['Wins-Home_and_Away'] * 3 ) + (home_and_away_draws['Draws-Home_and_Away'])

ten_season_points = pd.DataFrame(total_points_decade)
ten_season_points.columns = ['Total_points_in_decade']


# #### <font color=violet>Finally we concatenate dataframes from steps a to g into one single data frame<font/>

# In[ ]:


teams_stats_table = pd.concat([each_team_games, home_team_wins, home_team_loss, away_team_wins, 
                               away_team_loss, home_and_away_draws, ten_season_points], axis=1)
teams_stats_table


# ## <font color=blue>10. Extract the total points gained by teams who played in all 10 seasons.<br>$\;\;\;\;$ Plot the result as a bar chart with values of each bar indicated on top.<font/>

# #### <font color=violet>To extract the info from the dataframe created above, we need to match the index of the the dataframe 'teams_stats_table' with that of 'ten_season_teams' and extract the column 'Total_points_in_decade'.<font/>

# In[ ]:


allseason_teams = teams_stats_table[teams_stats_table.index.isin(ten_season_teams.index)]
allseason_teams_points = pd.DataFrame(allseason_teams['Total_points_in_decade'])
allseason_teams_points


# In[ ]:


plt.rcParams["figure.figsize"] = (15,10)
plt.bar(allseason_teams_points.index, allseason_teams_points['Total_points_in_decade'], 
        color=plt.cm.Paired((np.arange(len(allseason_teams_points)))),width = 0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Teams which played all seasons during 2009 - 2019',fontsize=20)
plt.ylabel('Total points collected in all seasons during 2009 - 2019',fontsize=20)
plt.title('Total points collected by teams which played in all seasons during 2009 - 2019',fontsize=20, color='red')

x_cord = -0.1
for z in range (len(allseason_teams_points)):
    plt.text(x_cord, allseason_teams_points.Total_points_in_decade[z] + 10, allseason_teams_points.Total_points_in_decade[z])
    x_cord = x_cord + 1
    
plt.show()


# ## <font color=blue>11. Which game/games produced most number of goals? List them.<font/>

# #### <font color=violet>We should add the FT_Home_Goal and the FT_Away_Goal values of each game and find the maximum goals scored in a single game. Then find all the games in which the maximum goals were scored.<font/>

# In[ ]:


max_goals_per_game = (decade_data_by_seasons['FT_Home_Goal'] + decade_data_by_seasons['FT_Away_Goal']).max()

print('Maximum number of goals scored in a single game is : '+str(max_goals_per_game))


# In[ ]:


decade_data_by_seasons[(decade_data_by_seasons['FT_Home_Goal'] + decade_data_by_seasons['FT_Away_Goal']) == max_goals_per_game]


# ## <font color=blue>12. Find out the total goals scored by each team at home, away and in total.<br>$\;\;\;\;$ Find the average goals scored per game at home and away (not combined).<br>$\;\;\;\;$ Represent it in a bar plot.<font/>

# #### <font color=violet>We find out the total goals scored by each team in home and away and add them.<br>To find average goals per game at home and away, divide goals at home and away by half of total games.<font/>

# In[ ]:


each_team_home_goals = decade_data_by_seasons['FT_Home_Goal'].groupby(decade_data_by_seasons['HomeTeam']).sum()

each_team_away_goals = decade_data_by_seasons['FT_Away_Goal'].groupby(decade_data_by_seasons['AwayTeam']).sum()

each_team_goal_stats = pd.DataFrame(index = each_team_games.index)
each_team_goal_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_goals), pd.DataFrame(each_team_away_goals)
                                , pd.DataFrame(each_team_home_goals + each_team_away_goals) ], axis=1)

each_team_goal_stats.columns = ['Total Games', 'HomeGoals', 'AwayGoals', 'TotalGoals']

each_team_goal_stats['AvgGoals_homeGame'] = (each_team_goal_stats['HomeGoals']/(each_team_goal_stats['Total Games']/2)).round(2)
each_team_goal_stats['AvgGoals_awayGame'] = (each_team_goal_stats['AwayGoals']/(each_team_goal_stats['Total Games']/2)).round(2)

each_team_goal_stats


# In[ ]:


each_team_goal_stats.plot(y=["AvgGoals_homeGame", "AvgGoals_awayGame"], kind="barh", legend=True, figsize=(15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Teams',fontsize=20)
plt.xlabel('Average number of goals per game',fontsize=20)
plt.title('Average number of goals scored per game during 2009 - 2019',fontsize=20, color='red') 
plt.show()


# ## <font color=blue>13. Find the total number of dominant performances by each team.<br>$\;\;\;\;$ (Winning by a goal margin of 3 or more goals at Full Time).<br>$\;\;\;\;$ Represent it in a donut plot (seperate for home and away performances).<br>$\;\;\;\;$ Include teams with less than 10 dominant wins into a group called 'others' in plots.<font/>

# #### <font color=violet>We group the dominant performances by home and away team and finally add them.<font/>

# In[ ]:


dominant_performances_home = decade_data_by_seasons[(decade_data_by_seasons['FT_Home_Goal'] - 
                                                     decade_data_by_seasons['FT_Away_Goal'] >= 3)
                                                   ].groupby([decade_data_by_seasons.HomeTeam]).agg('count')[['HomeTeam']]


dominant_performances_away = decade_data_by_seasons[(decade_data_by_seasons['FT_Away_Goal'] - 
                                                     decade_data_by_seasons['FT_Home_Goal'] >= 3)
                                                   ].groupby([decade_data_by_seasons.AwayTeam]).agg('count')[['AwayTeam']]

#since some teams have dominant performances only at home or away, we use merge by using index from both dataframes
dominant_performances = pd.merge(dominant_performances_home, dominant_performances_away, how = 'outer', 
                                 left_index=True, right_index=True)

dominant_performances.fillna(0, inplace = True)

dominant_performances['total_dominant_performances'] = dominant_performances['HomeTeam'] + dominant_performances['AwayTeam']

dominant_performances = dominant_performances.astype('int64')
dominant_performances


# #### <font color=violet>We create a category called others and the sum of all dominant performances less than 10 is assigned to it.<br>Then we select only teams whose dominant performances count is equal or more than 10.<br>We do it seperately for home and away performances.<font/>

# In[ ]:


others_home_dominant_games = dominant_performances[dominant_performances.HomeTeam < 10].sum()['HomeTeam']

others_away_dominant_games = dominant_performances[dominant_performances.AwayTeam < 10].sum()['AwayTeam']


# In[ ]:


dominant_home_games = pd.DataFrame(dominant_performances.HomeTeam)
others_home = pd.Series({'HomeTeam':others_home_dominant_games},name='Others')
dominant_home_games = dominant_home_games.append(others_home)
dominant_home_games = dominant_home_games[dominant_home_games.HomeTeam >= 10]
dominant_home_games


# In[ ]:


def value_and_percentage(x): 
    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)


plt.figure(figsize=(9,9))
values = dominant_home_games['HomeTeam']
labels = dominant_home_games.index
total = np.sum(values)
colors = ['#8BC34A','Pink','Olive','Grey','#FE7043','Turquoise',
          '#EAB300','Violet','Orange','Gold','Skyblue','#D4E157','#AA7043']
plt.pie (values , labels= labels , colors= colors , 
         startangle=45 , autopct=value_and_percentage, pctdistance=0.85)
my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Victories with a goal margin of 3 or more at home',fontsize=20, color='red')
plt.show()


# In[ ]:


dominant_away_games = pd.DataFrame(dominant_performances.AwayTeam)
others_away = pd.Series({'AwayTeam':others_away_dominant_games},name='Others')
dominant_away_games = dominant_away_games.append(others_away)
dominant_away_games = dominant_away_games[dominant_away_games.AwayTeam >= 10]
dominant_away_games


# In[ ]:


def value_and_percentage(x): 
    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)


plt.figure(figsize=(9,9))
values = dominant_away_games['AwayTeam']
labels = dominant_away_games.index
total = np.sum(values)
colors = ['#8BC34A','Pink','#FE7043','Turquoise','#EAB300','#D4E157','#AA7043']
plt.pie (values , labels= labels , colors= colors , 
         startangle=45 , autopct=value_and_percentage, pctdistance=0.85)
my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Victories with a goal margin of 3 or more away from home',fontsize=20, color='red')
plt.show()


# ## <font color=blue>14. Find the total points collected by each team per season.<br>$\;\;\;\;$ Extract the points table of the teams who played all 10 seasons.<br>$\;\;\;\;$ Plot it as a line graph.<font/>

# #### <font color=violet>We group the full time results (H, A and D) by Season and Home/Away Team.<font/>

# In[ ]:


season_home_wins = pd.DataFrame( decade_data_by_seasons[decade_data_by_seasons['FT_Result'] == 'H']
                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.HomeTeam]).agg('count')
                                .unstack().fillna(0).stack()['FT_Result']).reset_index()

season_home_wins.columns = ['Season', 'Team', 'H_Wins']


season_away_wins = pd.DataFrame( decade_data_by_seasons[decade_data_by_seasons['FT_Result'] == 'A']
                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.AwayTeam]).agg('count')
                                .unstack().fillna(0).stack()['FT_Result']).reset_index()

season_away_wins.columns = ['Season', 'Team', 'A_Wins']


season_home_draws = pd.DataFrame( decade_data_by_seasons[decade_data_by_seasons['FT_Result'] == 'D']
                                 .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.HomeTeam]).agg('count')
                                 .unstack().fillna(0).stack()['FT_Result']).reset_index()

season_home_draws.columns = ['Season', 'Team', 'H_Draws']


season_away_draws = pd.DataFrame( decade_data_by_seasons[decade_data_by_seasons['FT_Result'] == 'D']
                                 .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.AwayTeam]).agg('count')
                                 .unstack().fillna(0).stack()['FT_Result']).reset_index()

season_away_draws.columns = ['Season', 'Team', 'A_Draws']


# #### <font color=violet>We add the wins of every team (home and away) and multiply it by 3. To this, we add the number of draws to get the total points.<font/>

# In[ ]:


season_points_per_team = pd.DataFrame(season_home_wins['Team'])

season_points_per_team = pd.concat([season_points_per_team, pd.DataFrame(season_home_wins.H_Wins), 
                                    pd.DataFrame(season_away_wins.A_Wins), pd.DataFrame(season_home_draws.H_Draws), 
                                    pd.DataFrame(season_away_draws.A_Draws)], axis=1)

season_points_per_team = season_points_per_team.set_index(season_home_wins.Season)


season_points_per_team['Points'] = 3 * (season_points_per_team.H_Wins + 
                                        season_points_per_team.A_Wins) + (season_points_per_team.H_Draws + 
                                                                          season_points_per_team.A_Draws)

season_points_per_team = season_points_per_team[season_points_per_team.Points != 0]
#season_points_per_team
season_points_per_team.head()


# #### <font color=violet>To extract the points table of the teams who played all 10 seasons,<br>we match the Team column in season_points_per_team with the index of ten_season_teams.<font/>

# In[ ]:


ten_season_teams_points = season_points_per_team[season_points_per_team['Team'].isin(ten_season_teams.index)][['Team','Points']]
ten_season_teams_points = ten_season_teams_points.reset_index()
#ten_season_teams_points
ten_season_teams_points.head()


# In[ ]:


for club in ten_season_teams_points.Team.unique() :
    plt.plot(ten_season_teams_points[ten_season_teams_points['Team'] == club]['Season'], 
             ten_season_teams_points[ten_season_teams_points['Team'] == club]['Points'],  
             marker='o', label=club)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.xlabel('Season', fontsize = 20)
plt.ylabel('Points', fontsize = 20)
plt.title('Points collected by teams who played in all 10 seasons',fontsize=20, color='red')
plt.show()


# ## <font color=blue>15. Find the total shots on goal (shots and shots on target separately) made by each team.<br>$\;\;\;\;$ Present the result as a dataframe.<br>$\;\;\;\;$ Also find the average shots on goal in home and away games.<br>$\;\;\;\;$ Merge this dataframe with the dataframe containing goals stats.<font/>

# #### <font color=violet>We group the shots and shots on target (Home and Away) by Team (Home/Away).<font/>

# In[ ]:


each_team_home_shots = decade_data_by_seasons['H_Shots'].groupby(decade_data_by_seasons['HomeTeam']).sum()

each_team_away_shots = decade_data_by_seasons['A_Shots'].groupby(decade_data_by_seasons['AwayTeam']).sum()

each_team_home_shots_target = decade_data_by_seasons['H_Shots_Target'].groupby(decade_data_by_seasons['HomeTeam']).sum()

each_team_away_shots_target = decade_data_by_seasons['A_Shots_Target'].groupby(decade_data_by_seasons['AwayTeam']).sum()


each_team_shot_stats = pd.DataFrame(index = each_team_games.index)
each_team_shot_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_shots), pd.DataFrame(each_team_away_shots), 
                                  pd.DataFrame(each_team_home_shots + each_team_away_shots), 
                                  pd.DataFrame(each_team_home_shots_target), pd.DataFrame(each_team_away_shots_target), 
                                  pd.DataFrame(each_team_home_shots_target + each_team_away_shots_target)], axis=1)

each_team_shot_stats.columns = ['Total Games', 'HomeShots', 'AwayShots', 'TotalShots', 
                               'HomeShotsTarget', 'AwayShotsTarget', 'TotalShotsTarget']

each_team_shot_stats['AvgShots_homeGame'] = (each_team_shot_stats['HomeShots']/(each_team_shot_stats['Total Games']/2)).round(2)
each_team_shot_stats['AvgShots_awayGame'] = (each_team_shot_stats['AwayShots']/(each_team_shot_stats['Total Games']/2)).round(2)

each_team_shot_stats['AvgShotstarget_homeGame'] = (each_team_shot_stats['HomeShotsTarget']/
                                                   (each_team_shot_stats['Total Games']/2)).round(2)
each_team_shot_stats['AvgShotstarget_awayGame'] = (each_team_shot_stats['AwayShotsTarget']/
                                                   (each_team_shot_stats['Total Games']/2)).round(2)


# #### <font color=violet>Merging with the dataframe containing goal stats.<font/>

# In[ ]:


each_team_shot_and_goals = pd.concat([each_team_shot_stats, each_team_goal_stats], axis=1)
each_team_shot_and_goals = each_team_shot_and_goals.loc[:,~each_team_shot_and_goals.columns.duplicated()]

#reorder columns
each_team_shot_and_goals = each_team_shot_and_goals[['Total Games', 'HomeShots', 'HomeShotsTarget','HomeGoals',
                                                     'AwayShots', 'AwayShotsTarget', 'AwayGoals', 
                                                     'TotalShots', 'TotalShotsTarget', 'TotalGoals', 
                                                     'AvgShots_homeGame', 'AvgShotstarget_homeGame', 'AvgGoals_homeGame', 
                                                    'AvgShots_awayGame', 'AvgShotstarget_awayGame', 'AvgGoals_awayGame']]

#rename columns to fit
each_team_shot_and_goals.columns = ['Games', 'HS', 'HST', 'HG','AS', 'AST', 'AG', 
                                    'TS', 'TST', 'TG', 
                                    'AvgSHG', 'AvgSTHG', 'AvgGHG', 'AvgSAG', 'AvgSTAG', 'AvgGAG']

each_team_shot_and_goals


# ## <font color=blue>16. Extract the shots and goals stats of the teams who played all 10 seasons.<br>$\;\;\;\;$ Make bar plots for the following.<br>$\;\;\;\;$ a) TotalGoals, TotalShotsOnTarget, TotalShots. <br>$\;\;\;\;$ b) AvgGoals/Game@Home, AvgShotsOnTarget/Game@Home, AvgShots/Game@Home.<br>$\;\;\;\;$ c) AvgGoals/Game@Away, AvgShotsOnTarget/Game@Away, AvgShots/Game@Away.<font/>

# #### <font color=violet>To extract the points table of the teams who played all 10 seasons,<br>we match the index of each_team_shot_and_goals with the index of ten_season_teams.<font/>

# In[ ]:


ten_season_teams_shots_and_goals = each_team_shot_and_goals[each_team_shot_and_goals.index.isin(ten_season_teams.index)]
ten_season_teams_shots_and_goals


# In[ ]:


ten_season_teams_shots_and_goals.plot(y=['TG', 'TST','TS'], kind='bar', 
                                      label=['Total Goals', 'Total Shots on Target', 'Total Shots'], 
                                      color=['#8BC34A','#EAB300','tomato'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Team', fontsize = 20)
plt.ylabel('Shots', fontsize = 20)

plt.figtext(0.25, 0.93, "Shots and Goals stats of teams who played all 10 seasons", 
            fontsize=20, color='Black')

plt.figtext(0.26, 0.91, "Total Goals", fontsize=20, color='#8BC34A', ha ='left', va='top')
plt.figtext(0.38, 0.91, "vs", fontsize=20, color='Black', ha ='left', va='top')
plt.figtext(0.52, 0.91, "Total Shots on Target", fontsize=20, color='#EAB300', ha ='center', va='top')
plt.figtext(0.65, 0.91, "vs", fontsize=20, color='Black', ha ='center', va='top')
plt.figtext(0.78, 0.91, "Total Shots", fontsize=20, color='tomato', ha ='right', va='top')

plt.show()


# In[ ]:


ten_season_teams_shots_and_goals.plot(y=['AvgGHG', 'AvgSTHG','AvgSHG'], kind='bar', 
                                      label=['Avg Goals/Home game', 'Avg Shots on Target/Home Game', 'Avg Shots/Home Game'], 
                                      color=['#8BC34A','#EAB300','tomato'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Team', fontsize = 20)
plt.ylabel('Shots Avg', fontsize = 20)
plt.ylim(0, 20)

plt.figtext(0.15, 0.93, "Shots and Goals Averages/Game @Home of teams who played all 10 seasons", 
            fontsize=20, color='Black')

plt.figtext(0.11, 0.91, "Avg Goals/Home game", fontsize=20, color='#8BC34A', ha ='left', va='top')
plt.figtext(0.33, 0.91, "vs", fontsize=20, color='Black', ha ='left', va='top')
plt.figtext(0.51, 0.91, "Avg Shots on Target/Home Game", fontsize=20, color='#EAB300', ha ='center', va='top')
plt.figtext(0.68, 0.91, "vs", fontsize=20, color='Black', ha ='center', va='top')
plt.figtext(0.91, 0.91, "Avg Shots/Home Game", fontsize=20, color='tomato', ha ='right', va='top')

plt.show()


# In[ ]:


ten_season_teams_shots_and_goals.plot(y=['AvgGAG', 'AvgSTAG','AvgSAG'], kind='bar', 
                                      label=['Avg Goals/Away game', 'Avg Shots on Target/Away Game', 'Avg Shots/Away Game'], 
                                      color=['#8BC34A','#EAB300','tomato'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Team', fontsize = 20)
plt.ylabel('Shots Avg', fontsize = 20)
plt.ylim(0, 16)

plt.figtext(0.15, 0.93, "Shots and Goals Averages/Game @Away of teams who played all 10 seasons", 
            fontsize=20, color='Black')

plt.figtext(0.11, 0.91, "Avg Goals/Away game", fontsize=20, color='#8BC34A', ha ='left', va='top')
plt.figtext(0.33, 0.91, "vs", fontsize=20, color='Black', ha ='left', va='top')
plt.figtext(0.51, 0.91, "Avg Shots on Target/Away Game", fontsize=20, color='#EAB300', ha ='center', va='top')
plt.figtext(0.68, 0.91, "vs", fontsize=20, color='Black', ha ='center', va='top')
plt.figtext(0.91, 0.91, "Avg Shots/Away Game", fontsize=20, color='tomato', ha ='right', va='top')

plt.show()


# ## <font color=blue>17. Find the conversion rates (goals/shots) for the teams which played all 10 seasons.<br>$\;\;\;\;$ Present the result as a bar plot.<font/>

# #### <font color=violet>We add two more columns into ten_season_teams_shots_and_goals named 'CRH%' and 'CRA%'.<font/>

# In[ ]:


#CRH - coversion rate @ Home
#CRA - convertio rate @ Away

#two diff methods to divide columns
ten_season_teams_shots_and_goals['CRH%'] = (ten_season_teams_shots_and_goals['HG']/
                                           ten_season_teams_shots_and_goals['HS']*100).round(2)

ten_season_teams_shots_and_goals.loc[:,'CRA%'] = (ten_season_teams_shots_and_goals.loc[:,'AG']/
                                                 ten_season_teams_shots_and_goals.loc[:,'AS']*100).round(2)

ten_season_teams_shots_and_goals


# In[ ]:


ten_season_teams_shots_and_goals.plot(y=['CRH%', 'CRA%'], kind='bar', 
                                      label=['Conversion rate in Home Matches', 'Conversion rate in Away Mathes'], 
                                      color=['Turquoise','Violet'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Team', fontsize = 20)
plt.ylabel('Converison rate in %', fontsize = 20)

plt.title('Conversion rates (%) at home and away matches', fontsize = 20, color='red')

plt.show()


# ## <font color=blue>18. Find the average goals conceded @home and @away for all the teams.<br>$\;\;\;\;$ Present the result as a bar plot.<font/>

# #### <font color=violet>We find out the total goals conceded by each team at home (goals scored by away team) and away (goals scored by home team) and add them.<br>To find average goals conceded per game at home and away, divide goals conceded at home and away by half of total games.<font/>

# In[ ]:


each_team_home_goals_conceded = decade_data_by_seasons['FT_Away_Goal'].groupby(decade_data_by_seasons['HomeTeam']).sum()

each_team_away_goals_conceded = decade_data_by_seasons['FT_Home_Goal'].groupby(decade_data_by_seasons['AwayTeam']).sum()

each_team_goal_conceded_stats = pd.DataFrame(index = each_team_games.index)
each_team_goal_conceded_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_goals_conceded), 
                                           pd.DataFrame(each_team_away_goals_conceded), 
                                           pd.DataFrame(each_team_home_goals_conceded + each_team_away_goals_conceded) ], 
                                          axis=1)

each_team_goal_conceded_stats.columns = ['Total Games', 'GoalsConcededHome', 'GoalsConcededAway', 'TotalGoalsConceded']

each_team_goal_conceded_stats['AvgGoalsConceded_homeGame'] = (each_team_goal_conceded_stats['GoalsConcededHome']/
                                                              (each_team_goal_conceded_stats['Total Games']/2)).round(2)

each_team_goal_conceded_stats['AvgGoalsConceded_awayGame'] = (each_team_goal_conceded_stats['GoalsConcededAway']/
                                                              (each_team_goal_conceded_stats['Total Games']/2)).round(2)

each_team_goal_conceded_stats


# In[ ]:


each_team_goal_conceded_stats.plot(y=["AvgGoalsConceded_homeGame", "AvgGoalsConceded_awayGame"], 
                                   kind="barh", legend=True, figsize=(15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Teams',fontsize=20)
plt.xlabel('Average number of goals conceded per game',fontsize=20)
plt.title('Average number of goals conceded per game during 2009 - 2019',fontsize=20, color='red') 
plt.show()


# ## <font color=blue>19. Find the total yellow and red cards given to home and away teams.<br>$\;\;\;\;$ Present the result as a donut plot.<font/>

# #### <font color=violet>We find out the sum of yellow and red cards awarded to home and away sides.<font/>

# In[ ]:


total_yellows_home = decade_data_by_seasons['H_Yellow'].sum()
total_reds_home = decade_data_by_seasons['H_Red'].sum()
total_yellows_away = decade_data_by_seasons['A_Yellow'].sum()
total_reds_away = decade_data_by_seasons['A_Red'].sum()

yellow_and_red_cards = [total_yellows_home, total_reds_home, total_yellows_away, total_reds_away]

print('Total yellow cards awarded to home side : ' + str(yellow_and_red_cards[0]))
print('Total red cards awarded to home side : ' + str(yellow_and_red_cards[1]))
print('Total yellow cards awarded to away side : ' + str(yellow_and_red_cards[2]))
print('Total red cards awarded to away side : ' + str(yellow_and_red_cards[3]))


# In[ ]:


def value_and_percentage(x): 
    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)


plt.figure(figsize=(9,9))
values = yellow_and_red_cards
labels = ['Yellow cards for home team', 'Red cards for home team', 'Yellow cards for away team', 'Red cards for away team']
total = np.sum(values)
colors = ['Gold','#FE7043','Turquoise','Violet']
plt.pie (values , labels= labels , colors= colors , 
         startangle=45 , autopct=value_and_percentage, pctdistance=0.85, 
         textprops={'fontsize': 14}, explode=[0.02,0,0.02,0] )

my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Yellow and red cards distribution',fontsize=20, color='red')
plt.show()


# ## <font color=blue>20. Find the total yellow and red cards given to each team home and away.<br>$\;\;\;\;$ Plot the result (home and away combined, reds and yellows seperated) as a bar plot.<br>$\;\;\;\;$ Plot the average red and yellow cards per game at home and away as a bar plot.<font/>

# #### <font color=violet>We find out the total cards (yellow and red seperate) received by each team in home and away games and add them.<br>To find average cards received per game at home and away, divide the cards received at home and away by half of total games.<font/>

# In[ ]:


each_team_home_yellows = decade_data_by_seasons['H_Yellow'].groupby(decade_data_by_seasons['HomeTeam']).sum()
each_team_home_reds = decade_data_by_seasons['H_Red'].groupby(decade_data_by_seasons['HomeTeam']).sum()

each_team_away_yellows = decade_data_by_seasons['A_Yellow'].groupby(decade_data_by_seasons['AwayTeam']).sum()
each_team_away_reds = decade_data_by_seasons['A_Red'].groupby(decade_data_by_seasons['AwayTeam']).sum()


each_team_card_stats = pd.DataFrame(index = each_team_games.index)
each_team_card_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_yellows), pd.DataFrame(each_team_home_reds),
                                  pd.DataFrame(each_team_away_yellows), pd.DataFrame(each_team_away_reds), 
                                  pd.DataFrame(each_team_home_yellows + each_team_away_yellows), 
                                  pd.DataFrame(each_team_home_reds + each_team_away_reds)], axis=1)

each_team_card_stats.columns = ['Total Games', 'HomeYellows', 'HomeReds', 'AwayYellows', 
                                'AwayReds', 'TotalYellows', 'TotalReds']

#HGY - HomeGameYellow; HGR - HomeGameRed; AGY - AwayGameYellow; AGR - AwayGameRed
each_team_card_stats['AvgHGY'] = (each_team_card_stats['HomeYellows']/(each_team_card_stats['Total Games']/2)).round(2)
each_team_card_stats['AvgHGR'] = (each_team_card_stats['HomeReds']/(each_team_card_stats['Total Games']/2)).round(2)
each_team_card_stats['AvgAGY'] = (each_team_card_stats['AwayYellows']/(each_team_card_stats['Total Games']/2)).round(2)
each_team_card_stats['AvgAGR'] = (each_team_card_stats['AwayReds']/(each_team_card_stats['Total Games']/2)).round(2)

each_team_card_stats


# In[ ]:


ax1 = each_team_card_stats.plot( y=["TotalYellows"], kind="barh",
                          legend=False, color =('gold'), figsize=(15,15),
                          title='Yellow cards collected @home/away (combined) during 2009 - 2019', fontsize=14)
ax1.set(xlabel='Number of yellow cards', ylabel='Team')
ax1.title.set_size(20)
ax1yaxis_label = ax1.yaxis.get_label()
ax1yaxis_label.set_fontsize(14)
ax1xaxis_label = ax1.xaxis.get_label()
ax1xaxis_label.set_fontsize(14)
plt.show()


# In[ ]:


ax2 = each_team_card_stats.plot( y=["TotalReds"], kind="barh", 
                          legend=False, color =('red'), figsize=(15,15), 
                          title='Red cards collected @home/away (combined) during 2009 - 2019', fontsize=14)
ax2.set(xlabel='Number of cards', ylabel='Team')
ax2.title.set_size(20)
ax2yaxis_label = ax2.yaxis.get_label()
ax2yaxis_label.set_fontsize(14)
ax2xaxis_label = ax2.xaxis.get_label()
ax2xaxis_label.set_fontsize(14)

plt.show()


# In[ ]:


ax1 = each_team_card_stats.plot( y=["AvgHGY", "AvgAGY"], kind="bar",
                          legend=False, color =('olive','darkorange'), figsize=(40,10),
                          title='Average number of Yellow card collected per home/away game during 2009 - 2019', fontsize=30)
ax1.set(xlabel='', ylabel='Average number of cards per game\n') #we dont give x label here. Both plots will have same x axis.
ax1.title.set_size(30)
ax1yaxis_label = ax1.yaxis.get_label()
ax1yaxis_label.set_fontsize(30)

ax2 = each_team_card_stats.plot( y=["AvgHGR", "AvgAGR"], kind="bar", 
                          legend=False, color =('blue','red'), figsize=(40,10), 
                          title='Average number of Red card collected per home/away game during 2009 - 2019', fontsize=30)
ax2.set(xlabel='Team', ylabel='Average number of cards per game\n')
ax2.title.set_size(30)
ax2yaxis_label = ax2.yaxis.get_label()
ax2yaxis_label.set_fontsize(30)
ax2xaxis_label = ax2.xaxis.get_label()
ax2xaxis_label.set_fontsize(30)

plt.show()


# ## <font color=blue>21. Find the total number of yellow and red cards awarded by each referee.<br>$\;\;\;\;$ Plot the result as a bar plot.<font/>

# #### <font color=violet>We find out the total cards (yellow and red seperate) awarded by each referee in home and away games and add them.<font/>

# In[ ]:


referee_home_yellows = decade_data_by_seasons['H_Yellow'].groupby(decade_data_by_seasons['Referee']).sum()
referee_home_reds = decade_data_by_seasons['H_Red'].groupby(decade_data_by_seasons['Referee']).sum()
referee_away_yellows = decade_data_by_seasons['A_Yellow'].groupby(decade_data_by_seasons['Referee']).sum()
referee_away_reds = decade_data_by_seasons['A_Red'].groupby(decade_data_by_seasons['Referee']).sum()

each_referee_card_stats = pd.DataFrame(index = all_referees.index)
each_referee_card_stats = pd.concat([all_referees, pd.DataFrame(referee_home_yellows), pd.DataFrame(referee_home_reds),
                                  pd.DataFrame(referee_away_yellows), pd.DataFrame(referee_away_reds), 
                                  pd.DataFrame(referee_home_yellows + referee_away_yellows), 
                                  pd.DataFrame(referee_home_reds + referee_away_reds)], axis=1)

each_referee_card_stats.columns = ['Total Games', 'YellowsToHomeSide', 'RedsToHomeSide', 'YellowsToAwaySide', 
                                'RedsToAwaySide', 'YellowsTotal', 'RedsTotal']
each_referee_card_stats


# In[ ]:


ax1 = each_referee_card_stats.plot( y=["YellowsTotal"], kind="bar",
                          legend=False, color =('gold'), figsize=(40,10),
                          title='Yellow cards awarded during 2009 - 2019 in EPL', fontsize=30)

ax1.set(xlabel='', ylabel='Cards awarded') #we dont give x label here. Both plots will have same x axis.
ax1.title.set_size(30)
ax1yaxis_label = ax1.yaxis.get_label()
ax1yaxis_label.set_fontsize(30)

ax2 = each_referee_card_stats.plot( y=["RedsTotal"], kind="bar", 
                          legend=False, color =('crimson'), figsize=(40,10), 
                          title='Red cards awarded during 2009 - 2019 in EPL', fontsize=30)

ax2.set(xlabel='Referee', ylabel='Cards awarded')
ax2.title.set_size(30)
ax2yaxis_label = ax2.yaxis.get_label()
ax2yaxis_label.set_fontsize(30)

ax2xaxis_label = ax2.xaxis.get_label()
ax2xaxis_label.set_fontsize(30)

plt.show()


# ## <font color=blue>22. Find the total number of fouls committed by each team @ home and away.<br>$\;\;\;\;$ Plot the average fouls commited per game @ home and away by each team.<font/>

# #### <font color=violet>We group the fouls at home and away by each team and add them.<font/>

# In[ ]:


home_fouls = decade_data_by_seasons['H_Foul'].groupby(decade_data_by_seasons['HomeTeam']).sum()
away_fouls = decade_data_by_seasons['A_Foul'].groupby(decade_data_by_seasons['AwayTeam']).sum()


each_team_foul_stats = pd.DataFrame(index = each_team_games.index)
each_team_foul_stats = pd.concat([each_team_games, pd.DataFrame(home_fouls), pd.DataFrame(away_fouls), 
                                  pd.DataFrame(home_fouls + away_fouls)], axis=1)

each_team_foul_stats.columns = ['Total Games', 'FoulsCommitted@Home', 'FoulsCommitted@Away', 'TotalFoulsCommitted']


#HGF - HomeGameFoul; AGF - AwayGameFoul
each_team_foul_stats['AvgHGF'] = (each_team_foul_stats['FoulsCommitted@Home']/
                                  (each_team_foul_stats['Total Games']/2)).round(2)

each_team_foul_stats['AvgAGF'] = (each_team_foul_stats['FoulsCommitted@Away']/
                                  (each_team_foul_stats['Total Games']/2)).round(2)

each_team_foul_stats


# In[ ]:


each_team_foul_stats.plot(y=["AvgHGF", "AvgAGF"], 
                          kind="barh", legend=True, figsize=(15,15), color = ['crimson', 'darkgreen'], 
                          label=['Average Fouls in Home Game', 'Average Fouls in Away Game'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Teams',fontsize=20)
plt.xlabel('Number of fouls per game',fontsize=20)
plt.title('Average number of fouls commited per game at home and away games during 2009 - 2019',fontsize=18, color='red') 
plt.show()


# ## <font color=blue>23. Find the yellow card per foul in % and red card per foul in % for every team.<br>$\;\;\;\;$ Plot the result in separate bar plots.<font/>

# #### <font color=violet>We construct a new dataframe by using the following:<br>$\;\;\;\;$a) TotalFoulsCommitted from each_team_foul_stats.<br>$\;\;\;\;$b) TotalYellows and TotalReds from each_team_card_stats<font/>

# In[ ]:


each_team_fouls_to_cards = pd.DataFrame(index = each_team_games.index)
each_team_fouls_to_cards = pd.concat([each_team_games, each_team_foul_stats['TotalFoulsCommitted'], 
                                  each_team_card_stats['TotalYellows'], 
                                  each_team_card_stats['TotalReds']], axis=1)

each_team_fouls_to_cards['YellowsPerFoul %'] = ((each_team_fouls_to_cards['TotalYellows']/
                                             each_team_fouls_to_cards['TotalFoulsCommitted'])*100).round(2)

each_team_fouls_to_cards['RedsPerFoul %'] = ((each_team_fouls_to_cards['TotalReds']/
                                             each_team_fouls_to_cards['TotalFoulsCommitted'])*100).round(2)

each_team_fouls_to_cards


# In[ ]:


ax1 = each_team_fouls_to_cards.plot( y=["YellowsPerFoul %"], kind="bar",
                          legend=False, color =('gold'), figsize=(40,10),
                          title='Yellow cards per foul in % (cards per 100 fouls) for each team', fontsize=30)

ax1.set(xlabel='', ylabel='Yellow cards per foul in %\n (cards per 100 fouls) \n') #we dont give x label here. Both plots will have same x axis.
ax1.title.set_size(30)
ax1yaxis_label = ax1.yaxis.get_label()
ax1yaxis_label.set_fontsize(30)

ax2 = each_team_fouls_to_cards.plot( y=["RedsPerFoul %"], kind="bar", 
                          legend=False, color =('crimson'), figsize=(40,10), 
                          title='Red cards per foul in % (cards per 100 fouls) for each team', fontsize=30)

ax2.set(xlabel='Teams', ylabel='Red cards per foul in %\n (cards per 100 fouls) \n')
ax2.title.set_size(30)
ax2yaxis_label = ax2.yaxis.get_label()
ax2yaxis_label.set_fontsize(30)

ax2xaxis_label = ax2.xaxis.get_label()
ax2xaxis_label.set_fontsize(30)

plt.show()


# ## <font color=blue>24. Find the corners gained/conceded during home and away games for each team.<br>$\;\;\;\;$ Also find the total corners gained and conceded during 2009 - 2019.<br>$\;\;\;\;$ For teams who played all 10 seasons: <br>$\;\;\;\;\;\;$ a) Find the average corners conceded/gained at home & away per game.<br>$\;\;\;\;\;\;$ b) Plot the result as a bar plot.<font/>

# #### <font color=violet>We group the corners for home side by home team and also group the corners for away side by away team.<br>We add them to get total corners gained<br>We group the corners for away side by home team and also group the corners for home side by away team.<br>We add them to get total corners conceded<br><font/>

# In[ ]:


each_team_home_corners_gained = decade_data_by_seasons['H_Corner'].groupby(decade_data_by_seasons['HomeTeam']).sum()
each_team_home_corners_conceded = decade_data_by_seasons['A_Corner'].groupby(decade_data_by_seasons['HomeTeam']).sum()

each_team_away_corners_gained = decade_data_by_seasons['A_Corner'].groupby(decade_data_by_seasons['AwayTeam']).sum()
each_team_away_corners_conceded = decade_data_by_seasons['H_Corner'].groupby(decade_data_by_seasons['AwayTeam']).sum()


each_team_corner_stats = pd.DataFrame(index = each_team_games.index)
each_team_corner_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_corners_gained), 
                                    pd.DataFrame(each_team_home_corners_conceded),
                                    pd.DataFrame(each_team_away_corners_gained), 
                                    pd.DataFrame(each_team_away_corners_conceded), 
                                    pd.DataFrame(each_team_home_corners_gained + each_team_away_corners_gained), 
                                    pd.DataFrame(each_team_home_corners_conceded + each_team_away_corners_conceded)], 
                                   axis=1)

#CG - Corners Gained; CC - Corners Conceded
each_team_corner_stats.columns = ['Total Games', 'CG@Home', 'CC@Home', 
                                  'CG@Away', 'CC@Away', 
                                  'TotalCG', 'TotalCC']

#HCG - HomeCornersGained; HCC - HomeCornersConceded; ACG - AwayCornersGained; ACC - AwayCornersConceded
each_team_corner_stats['AvgHCG'] = (each_team_corner_stats['CG@Home']/
                                    (each_team_corner_stats['Total Games']/2)).round(2)

each_team_corner_stats['AvgHCC'] = (each_team_corner_stats['CC@Home']/
                                    (each_team_corner_stats['Total Games']/2)).round(2)

each_team_corner_stats['AvgACG'] = (each_team_corner_stats['CG@Away']/
                                    (each_team_corner_stats['Total Games']/2)).round(2)

each_team_corner_stats['AvgACC'] = (each_team_corner_stats['CC@Away']/
                                    (each_team_corner_stats['Total Games']/2)).round(2)

each_team_corner_stats


# In[ ]:


each_team_corner_stats.plot(y=["TotalCG", "TotalCC"], 
                            kind="bar", legend=True, figsize=(40,15), color = ['green', 'red'], 
                            label=['Total Corners Gained', 'Total Corners Conceded'])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Corners gained/conceded between 2009 - 2019',fontsize=20)
plt.xlabel('Teams',fontsize=20)
plt.title('Total corners gained/conceded between 2009 - 2019 in EPL',fontsize=20, color='red') 
plt.show()


# #### <font color=violet>To extract the corner stats of the teams who played all 10 seasons,<br>we match the index of each_team_corner_stats with the index of ten_season_teams.<font/>

# In[ ]:


corner_stats_ten_season_teams = each_team_corner_stats[each_team_corner_stats.index.isin(ten_season_teams.index)]
corner_stats_ten_season_teams


# In[ ]:


corner_stats_ten_season_teams.plot(y=["AvgHCG", "AvgHCC", "AvgACG", "AvgACC"], 
                                   kind="barh", legend=True, figsize=(15,15), color = ['green', 'deeppink' , 'lime', 'red'], 
                                   label = ['Avg Corners Gained @Home per game', 'Avg Corners Conceded @Home per game', 
                                           'Avg Corners Gained @Away per game', 'Avg Corners Conceded @Away per game'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Average number of corners gained/conceded per game',fontsize=20)
plt.ylabel('Teams',fontsize=20)
plt.title('Average number of corners gained/conceded per game between 2009 - 2019 in EPL',fontsize=20, color='red') 
plt.show()


# ## <font color=blue>25. Find the EPL champions of each year with their points.<br>$\;\;\;\;$ If 2 teams have equal points in a season, find the champion by goal difference.<br>$\;\;\;\;$ (Total Goals Scored - Total Goals Conceded)<br>$\;\;\;\;$ Find howmany teams became EPL champions during 2009 - 2019. List their names.<br>$\;\;\;\;$ Find the team which won the EPL more times between 2009 - 2019.<br>$\;\;\;\;$ Plot the teams with number of EPL trophies in the decade as a donut plot.<font/>

# #### <font color=violet>To find the teams which earned maximum points every season,<br>we group the points earned by each team by season and get the maximum for every season.<font/>

# In[ ]:


max_points_teams_per_season = season_points_per_team.groupby([season_points_per_team.
                                                    index])['Points'].transform(max) == season_points_per_team['Points']

season_points_per_team[max_points_teams_per_season]


# #### <font color=red>From the above data frame we notice that, in the season 11-12, there are two teams with same points.<br>In such a case, we have to consider the goal difference to figure out which team were champions. <font/>

# #### <font color=violet>We make a new dataframe with the goal difference of each team per season.<font/>

# In[ ]:


#Find goals scored by each team per season (home + away)
season_home_goals_scored = pd.DataFrame( decade_data_by_seasons['FT_Home_Goal']
                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.HomeTeam]).sum()
                                .unstack().fillna(0).stack()).reset_index()

season_home_goals_scored.columns = ['Season', 'Team', 'HGoals_Scored']

season_away_goals_scored = pd.DataFrame( decade_data_by_seasons['FT_Away_Goal']
                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.AwayTeam]).sum()
                                .unstack().fillna(0).stack()).reset_index()

season_away_goals_scored.columns = ['Season', 'Team', 'AGoals_Scored']


season_total_goals_scored = pd.DataFrame()
season_total_goals_scored = pd.concat([season_home_goals_scored.Season, season_home_goals_scored.Team, 
                                       pd.DataFrame(season_home_goals_scored['HGoals_Scored'] + 
                                                    season_away_goals_scored['AGoals_Scored'])],
                                      axis=1)


season_total_goals_scored.columns = ['Season', 'Team', 'Total_Goals_Scored']


#Find goals conceded by each team per season (home + away)
season_home_goals_conceded = pd.DataFrame( decade_data_by_seasons['FT_Away_Goal']
                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.HomeTeam]).sum()
                                .unstack().fillna(0).stack()).reset_index()

season_home_goals_conceded.columns = ['Season', 'Team', 'HGoals_Conceded']

season_away_goals_conceded = pd.DataFrame( decade_data_by_seasons['FT_Home_Goal']
                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.AwayTeam]).sum()
                                .unstack().fillna(0).stack()).reset_index()

season_away_goals_conceded.columns = ['Season', 'Team', 'AGoals_Conceded']


season_total_goals_conceded = pd.DataFrame()
season_total_goals_conceded = pd.concat([season_home_goals_conceded.Season, season_home_goals_conceded.Team, 
                                       pd.DataFrame(season_home_goals_conceded['HGoals_Conceded'] + 
                                                    season_away_goals_conceded['AGoals_Conceded'])],
                                      axis=1)

season_total_goals_conceded.columns = ['Season', 'Team', 'Total_Goals_Conceded']


#Make a new df with Goal Difference

season_total_gd = pd.DataFrame()
season_total_gd = pd.concat([season_total_goals_scored.Season, season_total_goals_scored.Team, 
                             season_total_goals_scored.Total_Goals_Scored, season_total_goals_conceded.Total_Goals_Conceded],
                            axis=1)

season_total_gd['Goal_Difference'] = season_total_gd['Total_Goals_Scored'] - season_total_gd['Total_Goals_Conceded']


season_total_gd = season_total_gd[(season_total_gd.Total_Goals_Scored != 0)  &  (season_total_gd.Total_Goals_Conceded != 0)]
season_total_gd.set_index('Season')
#season_total_gd
season_total_gd.head()


# #### <font color=violet>We concatenate the data frame having goal difference with the dataframe having points of each team per season.<font/>

# In[ ]:


season_total_gd.index = season_points_per_team.index


season_points_per_team_with_goal_diff = pd.concat([season_points_per_team, season_total_gd.Total_Goals_Scored, 
                                                   season_total_gd.Total_Goals_Conceded, season_total_gd.Goal_Difference], 
                                                  axis=1, sort=False)

#season_points_per_team_with_goal_diff
season_points_per_team_with_goal_diff.head()


# #### <font color=violet>To find the teams which earned maximum points every season,<br>we group the points earned by each team by season and get the maximum for every season.<font/>

# In[ ]:


max_points_teams_each_season = season_points_per_team_with_goal_diff.groupby([
    season_points_per_team_with_goal_diff.index])['Points'].transform(max) == season_points_per_team_with_goal_diff['Points']


max_points_teams_per_season = season_points_per_team_with_goal_diff[max_points_teams_each_season]
max_points_teams_per_season = max_points_teams_per_season.reset_index()
max_points_teams_per_season


# #### <font color=violet>Now we can also make use of the goal difference in case two teams have same points.<br>We make a new dataframe and use the dataframe max_points_teams_per_season to filter only the team with the highest goal difference in case more than one team had same points at the end of the season.<font/>

# In[ ]:


champions = pd.DataFrame()

for season in max_points_teams_per_season.Season.unique():
    seasonal_top_pointers = max_points_teams_per_season[max_points_teams_per_season['Season']==season]
    seasonal_top_pointers_with_gd = seasonal_top_pointers[seasonal_top_pointers['Goal_Difference']==
                                                          seasonal_top_pointers['Goal_Difference'].max()]
    champions = champions.append(seasonal_top_pointers_with_gd)

champions = champions.set_index('Season')
champions


# In[ ]:


print('The different teams to be crowned EPL champions between 2009 - 2019 are : \n' + str(champions.Team.unique()))


# In[ ]:


champions_with_trophy_count = pd.DataFrame(champions['Team'].value_counts())
champions_with_trophy_count.columns = ['Trophy_Number']
champions_with_trophy_count


# In[ ]:


team_with_max_epl_trophy = champions_with_trophy_count[champions_with_trophy_count['Trophy_Number']==
                                                          champions_with_trophy_count['Trophy_Number'].max()]
team_with_max_epl_trophy


# In[ ]:


def value_and_percentage(x): 
    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)


plt.figure(figsize=(9,9))
values = champions_with_trophy_count.Trophy_Number
labels = champions_with_trophy_count.index.unique()
total = np.sum(values)
colors = ['#8BC34A','dodgerblue','#FE7043','Turquoise']
plt.pie (values , colors = colors ,  labels = labels,
         startangle=45 , autopct=value_and_percentage, pctdistance=0.85, 
         textprops={'fontsize': 14}, explode=[0.05,0,0,0] )

my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Number of EPL trophies between 2009 - 2019',fontsize=20, color='red')
plt.show()


# ## <font color=blue>A lot more analysing can be carried out on this data set.<br>I invite you to try analysing this data set and find new results.<br>https://www.kaggle.com/aj7amigo/english-premier-league-data-2009-2019<font/>

# <h2 align="center"> <font color=green>Suggestions to improve this notebook are welcome.<font/> </h2>

# ### Thank you,<br> <font color=brown>AKHIL JAMES<font/> 

# <img src="https://miro.medium.com/max/681/1*-BoC1TULqBUhCw-TdGaBgQ.png" alt="Drawing" style="width: 400px;"/>
