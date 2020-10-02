#!/usr/bin/env python
# coding: utf-8

# ####                                  <font color='green'> First project on Kaggle</font>
# I am at the initial stage of this project I'll keep updating it. 

# ### European Soccer Dataset

# Dataset given has been collected from https://www.kaggle.com/hugomathien/soccer and it is in the SQLite form. So to read this kind of file in Python we'll have to import the libray sqlite3. If it's not present in your IDE the use '!pip istall sqlite3' in jupyter notebook and it'll get installed in a while. SQL is a conceptual language for working with data stored in databases. In our case, SQLite is the specific implementation. Eventually, we will use SQL lunguage to write queries that would pull data from the DB, manipulate it, sort it, and extract it. The most important component of the DB is its tables - that's where all the data stored. Usually the data would be devided to many tables, and not stored all in one place. Here, most of the time will be spent handling data using SQL in tables.

# In[ ]:


#Importing libraries

import numpy as np  #Linear Algebra
import pandas as pd  #Data Processing
import seaborn as sns  #Visualization
import matplotlib.pyplot as plt
import sqlite3  


# ### <font color='orange'>Importing the data from SQLite database</font>
# 
# ##### SO here we'll check our Database and how many tables does it contain. 
# The basic structure of the query is very simple: You define what you want to see after the SELECT, * means all possible columns You choose the table after the FROM You add the conditions for the data you want to use from the table(s) after the WHERE
# 
# The stracture, and the order of the sections matter, while spaces, new lines, capital words and indentation are there to make the code easier to read.

# In[ ]:


path = "../input/"  
database = path + 'database.sqlite'
conn = sqlite3.connect(database)
query = "SELECT * FROM sqlite_master WHERE type='table'"

tables = pd.read_sql_query(query,conn)
tables


# ###  List of Countries 

# In[ ]:


countries = pd.read_sql('SELECT *FROM Country', conn)
countries


# ### List of Leagues in Different Countries
#     
# JOIN is used when you want to connect two tables to each other. It works when you have a common key in each of them. Understanding the concept of Keys is crucial for connecting between data set (tables). A key is uniquely identifies each record in a table. It can consinst of one value - usually ID, or from a combination of values that are unique in the table.
# Specify the common value that is used to connect the tables (the id of the country in that case).
# Make sure that at least one of the values has to be a key in its table. In our case, it's the Country.id. The League.country_id is not unique, as there can be more than one league in the same country

# In[ ]:


leagues = pd.read_sql("""SELECT *
                        FROM League
                        JOIN Country ON Country.id = League.country_id;""", conn)
leagues


# ## List of Teams
# 
# ###### ORDER BY defines the sorting of the output - ascending or descending (DESC)
# 
# ###### LIMIT, limits the number of rows in the output - after the sorting    

# In[ ]:


teams = pd.read_sql('SELECT *FROM Team ORDER BY team_long_name', conn)
teams.head()


# ### List of Matches
#     
# In this exapmle we will show only the columns that interests us, so instead of * we will use the exact names.
# 
# Some of the cells have the same name (Country.name,League.name). We will rename them using AS.
# 
# As you can see, this query has much more joins. The reasons is because the DB is designed in a star structure - one table (Match) with all the "performance" and metrics, but only keys and IDs, while all the descriptive information stored in other tables (Country, League, Team)    
# And then I am creating different tables for all the leagues from the dataset matches which I'll later store in Matches.

# In[ ]:


matches_Spain = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Spain' 
                                ORDER by date
                               ;""", conn)
#matches.head() #It gives us first 5 matches in Spanish League in season 2008/09

matches_Eng = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'England' 
                                ORDER by date
                               ;""", conn)

matches_Ger = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Germany' 
                                ORDER by date
                               ;""", conn)

matches_Por = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Portugal' 
                                ORDER by date
                               ;""", conn)

matches_Italy = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Italy' 
                                ORDER by date
                               ;""", conn)

matches_France = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'France' 
                                ORDER by date
                               ;""", conn)

matches_Bel = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Belgium' 
                                ORDER by date
                               ;""", conn)

matches_Ned = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Netherlands' 
                                ORDER by date
                               ;""", conn)

matches_Pol = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Poland' 
                                ORDER by date
                               ;""", conn)

matches_Scot = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Scotland' 
                                ORDER by date
                               ;""", conn)

matches_Swt = pd.read_sql("""SELECT Match.id, 
                            Country.name AS country_name, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            date,
                            HT.team_long_name AS  home_team,
                            AT.team_long_name AS away_team,
                            home_team_goal, 
                            away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'Switzerland' 
                                ORDER by date
                               ;""", conn)


# ### Making a list of all the leagues matches</font>
# 
# So here we created a list of matches in all respective leagues so that we can iterate over all the leagues to do other calculations easily.

# In[ ]:


matches = [matches_Spain,matches_Eng,matches_Italy,matches_Ger,matches_Por,matches_France,matches_Swt,matches_Pol,matches_Scot,matches_Ned,matches_Bel]


#  Here the function **goal_diff** is creating a new column goal difference in each league's table.

# In[ ]:


def goal_diff(fun):
    for i in fun:
        i['goal_diff'] = i['home_team_goal']-i['away_team_goal']
        print(i.tail())

goal_diff(matches)   
  


#  The function **des_stats** is giving us the standard deviation and mean for the  **Home goals,Away goals and the Goal difference** respectively for all the leagues.

# In[ ]:


def des_stats(fun):
    for i in fun:
        print(i.country_name[0])
        print('')
        print('Mean')
        print('Home Goals =',i['home_team_goal'].mean())
        print('Away Goals =',i['away_team_goal'].mean())
        print('Goal Diff =',i['goal_diff'].mean())
        print('')
        print('Standard Deviation')
        print('Home Goals =',i['home_team_goal'].std())
        print('Away Goals =',i['away_team_goal'].std())
        print('Goal Diff =',i['goal_diff'].std())
        print('\n')
        


# In[ ]:


des_stats(matches)


#  The function  **result** will give us the  **win,draw and lose**> percentage for all the leagues so that we can get it easier to visualize. Also the  **win,draw and lose are with respective of the home team.**

# In[ ]:


def result(fun):
    for i in fun:
        i['win'] = ((i['goal_diff']>0)*1)
        i['draw'] = ((i['goal_diff']==0)*1)
        i['lose'] = ((i['goal_diff']<0)*1)
        print(i.country_name[0])
        print('')
        print('win =' ,(i['win'].mean())*100,'%')
        print('draw =',(i['draw'].mean())*100,'%')
        print('lose =',(i['lose'].mean())*100,'%')
        print('\n')


# In[ ]:


result(matches)


#  To make the visualization process easy we'll make these  **win,lose and draw %ages** into lists so that we can plot it .

# In[ ]:


list_win = []
for i in matches:
    list_win.append(i['win'].mean())
#print(list_win) 


list_draw = []
for i in matches:
    list_draw.append(i['draw'].mean())
#print(list_draw)

list_lose = []
for i in matches:
    list_lose.append(i['lose'].mean())
    #print(list_lose)

list_countries = []

for i in matches:
    list_countries.append(i['country_name'][0])
    
#print(list_countries)    


#  The graph gives us the ratio of  **win,lose and draw** for the  **Home Teams in all leagues.**

# In[ ]:


plt.figure(figsize = (15,7))

x = np.arange(len(list_win))

plt.bar(x,list_win,color='green', width = 0.5,label = 'Win')
plt.bar(x,list_draw,color= 'orange',bottom = list_win,width = 0.5,label = 'Draw')
plt.bar(x,list_lose,color= 'grey',bottom =[x+y for x,y in zip(list_win,list_draw)],width = 0.5,label = 'Lose')

#label

plt.xticks(x,list_countries, fontsize = 13)
plt.title('Home-Win-Draw-Lose',fontsize = 22)
plt.xlabel('Countries', fontsize = 16)
plt.ylabel('Ratio', fontsize = 16)


plt.legend(loc = 'upper right')

plt.savefig('Win_Draw_Lose.png')
plt.show()


# In[ ]:


for i in matches:
    i['winner'] = np.where(i['win']==1, i['home_team'],i['away_team'])


# #### So here I created a column  winner in all the leagues' table which prints out the name of the team that's winning a match. And then we'll drop the values of draws so that we can take out wins and can find the total number of matches won by a team and then we extract the teams that have won the maximum matches in respective leagues.

# In[ ]:


matches_Spain.ix[matches_Spain['draw']==1]
matches_won = matches_Spain.ix[matches_Spain['draw']==1]


# In[ ]:


top_teams_Spain = pd.crosstab(index = matches_Spain['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Eng = pd.crosstab(index = matches_Eng['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Ger = pd.crosstab(index = matches_Ger['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_France = pd.crosstab(index = matches_France['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Italy = pd.crosstab(index = matches_Italy['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Pol = pd.crosstab(index = matches_Pol['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Swt = pd.crosstab(index = matches_Swt['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Ned = pd.crosstab(index = matches_Ned['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Scot = pd.crosstab(index = matches_Scot['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Por = pd.crosstab(index = matches_Por['winner'], columns = 'count').nlargest(10,'count')['count']
top_teams_Bel = pd.crosstab(index = matches_Bel['winner'], columns = 'count').nlargest(10,'count')['count']

top_teams = [top_teams_Spain,top_teams_Eng,top_teams_Ger,top_teams_France,top_teams_Italy,top_teams_Pol,top_teams_Swt,
            top_teams_Ned,top_teams_Scot,top_teams_Por,top_teams_Bel]


#  The plot shows you all the teams which have largest wins in their respective leagues so I have taken  **Ten Best Teams** from each league and then after plotting them you'll get the  **Top Ten Teams in Europe** during the seasons 2008-16.

# In[ ]:


plt.figure(figsize=(19,7))
#defining all the plots with respect to all the leagues
ax1 = plt.plot(top_teams_Spain,'o',label = 'Spain');
ax2 = plt.plot(top_teams_Eng,'o',label = 'England');
ax3 = plt.plot(top_teams_Ger,'o',label = 'Germany');
ax4 = plt.plot(top_teams_France,'o',label = 'France');
ax5 = plt.plot(top_teams_Italy,'o',label = 'Italy');
ax6 = plt.plot(top_teams_Pol,'o',label = 'Poland');
ax7 = plt.plot(top_teams_Swt,'o',label = 'Switzerland');
ax8 = plt.plot(top_teams_Ned,'o',label = 'Netherlands');
ax9 = plt.plot(top_teams_Scot,'o',label = 'Scotland');
ax10 = plt.plot(top_teams_Por,'o',label = 'Portugal');
ax11 = plt.plot(top_teams_Bel,'ko',label = 'Belgium');
plt.legend(loc = 'best');
plt.title('Top Teams in Europe', fontsize= 20)
plt.xlabel('Leagues',fontsize = 15)
plt.ylabel('Matches Won',fontsize = 15)

#Annotations for to Top Ten Teams
plt.annotate('Barcelona',(1,265))
plt.annotate('Real Madrid',(2,252))
plt.annotate('Celtic',(81,245))
plt.annotate('Man United',(11,233))
plt.annotate('Juventus',(41,225))
plt.annotate('Bayern Munich',(21,223))
plt.annotate('PSG',(31,219))
plt.annotate('Man City',(12,218))
plt.annotate('FC Basel',(61,218))
plt.annotate('Ajax',(71,216))



plt.savefig('Top Teams in Europe.png')
    


# ### Distribution Plot to show the Competitiveness in the Top 5 Leagues in Europe
# The distribution plot shows that the leagues that are **normally distributed** have average competition because these leagues show that the teams that win highest and lowest matches in the league have approximate same distance from the mean matches won by a team. France for example shows an approximately normal distribution which indicates that **League 1** (French League) has average competition among the teams.
# 
# **Serie A**(Italy) plot is slightly right skewed showing more than average competition while **Bundesliga**(Germany) has a negatively skewed distribution showing that only a handful of teams are leading in the league. 
# 
# In **La Liga**(Spain) we have normal distribution but there's a buldge in the curve showing that most of the teams perform around mean while few teams perform exceptionally well.
# 
# The most interesting pattern is in **English Premier League** which is rightly skewed showing that there's no Clear Leader in the den and the league is highly competitive.

# In[ ]:


plt.figure(figsize = (15,7))
ax1 = sns.distplot(top_teams_Spain,hist = False,label = 'La Liga');
ax2 = sns.distplot(top_teams_Eng,hist = False,label = 'EPL');
ax3 = sns.distplot(top_teams_Ger,hist = False,label = 'Bundesliga');
ax4 = sns.distplot(top_teams_France,hist = False,label = 'League 1');
ax5 = sns.distplot(top_teams_Italy,hist = False,label = 'Serie A');
#ax6 = sns.distplot(top_teams_Pol,hist = False,label = 'Poland');
#ax7 = sns.distplot(top_teams_Swt,hist = False,label = 'Switzerland');
#ax8 = sns.distplot(top_teams_Ned,hist = False,label = 'Netherlands');
#ax9 = sns.distplot(top_teams_Scot,hist = False,label = 'Scotland');
#ax10 = sns.distplot(top_teams_Por,hist = False,label = 'Portugal');
#ax11 = sns.distplot(top_teams_Bel,hist = False,label = 'Belgium');

plt.title('Distribution Plot to show the Competitiveness in the Top 5 Leagues in Europe',fontsize=16)
plt.xlabel('Matches Won',fontsize = 14);
plt.ylabel('KDE',fontsize = 14)

plt.savefig('Competitiveness in the Top 5 Leagues in Europe.png')
    


# In[ ]:




