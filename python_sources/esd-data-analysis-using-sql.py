#!/usr/bin/env python
# coding: utf-8

# # Intro
# I'm using the European Soccer Database available in Kaggle to apply the content from the excllent Datacamp course Intermediate SQL. https://www.datacamp.com/courses/intermediate-sql I'm doing this in order help me ingest the principles from the course for the longer term. In some cases the SQL queries will be taken directly from the course but tweaked to work on the sqlite ESD database. Other queries will be deviations where new exercises are tried/ new questions are attempted to be answered.

# # Import and Connect
# The below code block sets up pandas, sqlite3 so we can use this python based notebook to run SQL queries

# In[ ]:


#Imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

path = "../input/"  #Insert path here
database = path + 'database.sqlite'


# # Whats in the tables?
# 
# We want to see what is in the tables we have in our database, which we can get with the below query. Since we are actually running SQL queries in a python notebook, we need to put our SQL statments with a python wrapper around them. The key aspect is to place the SQL statement within each side of three speach marks. 

# In[ ]:


conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables


# # Lets see what is the tables.
# What countries are in the European Soccer Database? What teams are present? What does the table listing all the matches look like? What does the league table look like?

# In[ ]:


countries = pd.read_sql("""SELECT *
                        FROM Country;""", conn)
countries


# In[ ]:


teams = pd.read_sql("""SELECT *
                        FROM Team
                        Where team_long_name ='Tottenham Hotspur';""", conn)
teams


# In[ ]:


match = pd.read_sql("""SELECT *
                        FROM Match;""", conn)
match


# In[ ]:


league = pd.read_sql("""SELECT *
                        FROM League;""", conn)
league


# # List of leagues and their country 
# From the name of the league is pretty easy to guess which county each league belongs to. However lets check by finding the name of country by joining data from the country table onto the league table. We can do this by joining on the country id column which is in the league table, to the id column in the country table. 

# In[ ]:


leagues = pd.read_sql("""SELECT *
                        FROM League
                        LEFT JOIN Country 
                        ON Country.id = League.country_id;""", conn)
leagues


# As it turns out the id and country_id columns in the league table are the same so we could have used either column to join onto country. However there might one day be a more complex version of the database which contains more than one league per country, for example the English Football League Championship in England.

# # List of matches and goals for English Premier League
# Lets join the tables to make a dataset that football fans would probably want to look at first. What matches were played in a particular league (English Premier League) during a particular season (2015/16) and what was the score? 
# 
# For this query we join the match table to the country table, then to the league table, and then to the team table twice.
# 
# Some of the fields in different tables have the same name (Country.name,League.name) so will rename them using AS in the query results.
# 
# This query has more joins, as it joins both fact and dimension tables. For example one table, Match is a fact table with all the facts or metrics or numbers, in this case about football matches, and other that that has only keys and IDs to link to, the dimension tables, which in this case are country, league and team. Dimension tables would contain information that people would normally like to filter on.
# 
# In a live database, the match table would be updated every time there is a match. The dimension tables would only need to be updated much less often.
# 
# For the query results to make sense,note that "team long name" from the team table is selected twice and aliased as either home team or away team. The team table is also aliased twice, as HT for home team and AT for away team. Then to actually distinquish which team was the home and away team the information is joined onto the match table where the information matches to the home and away team api.

# In[ ]:


detailed_matches = pd.read_sql("""SELECT Match.id, 
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
                                Left JOIN Country on Country.id = Match.country_id
                                Left JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name = 'England'
                                AND season = '2015/2016'
                                ORDER by date
                                LIMIT 10;""", conn)
detailed_matches


# # A CASE WHEN statement telling us when Spurs won or lost

# After looking at all the results for the league, perhaps a football fan would want to only look at the match results of the team they follow. The below query only looks at Spurs's away matches and using a CASE WHEN statement, assigns a result to every match.

# In[ ]:



tot = pd.read_sql("""SELECT m.date,
t.team_long_name AS 'opponent',
CASE WHEN m.home_team_goal < m.away_team_goal THEN 'Spurs win!'
        WHEN m.home_team_goal > m.away_team_goal THEN 'Spurs loss :(' 
        ELSE 'Tie' END AS outcome
                        FROM match m
                        LEFT JOIN Team t
                        ON m.home_team_api_id = t.team_api_id
                        WHERE m.away_team_api_id = 8586
                        ;""", conn)
tot
         


# # Across different countries/leagues, what percentage of matches end in a draw?

# Lets see what percentage of matches ended in a draw across different nations, for season 2013/14 and 2014/15. We select the country, then go into a CASE WHEN which specifies the season and counts a 1 when there is a draw. This is placed in within the AVG function to create the average. Country and Match tables are joined. The three fields (country, percentage draws 2013/14, percentage draws 2014/15) are grouped by country.

# In[ ]:


per = pd.read_sql("""SELECT 
c.name AS country,
ROUND(AVG(CASE WHEN m.season='2013/2014' AND m.home_team_goal = m.away_team_goal THEN 1
 WHEN m.season='2013/2014' AND m.home_team_goal != m.away_team_goal THEN 0
END),2) AS pct_ties_2013_2014,
ROUND(AVG(CASE WHEN m.season='2014/2015' AND m.home_team_goal = m.away_team_goal THEN 1
 WHEN m.season='2014/2015' AND m.home_team_goal != m.away_team_goal THEN 0
END),2) AS pct_ties_2014_2015
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
GROUP BY country
;""", conn)

per


# Which League has the most number of goals on average?
# 
# 

# In[ ]:


leages_by_season = pd.read_sql("""

SELECT 
    l.name AS league,
    avg(m.home_team_goal + m.away_team_goal) AS avg_goals
    FROM league as L
    LEFT JOIN match AS m
ON l.id = m.country_id


WHERE m.season = '2013/2014'
    GROUP BY league
    ;""", conn)
leages_by_season
    


# In[ ]:


leages_by_season = pd.read_sql("""

    SELECT 
    l.name AS league,
    ROUND(avg(m.home_team-goal + m.away_team_goal), 2) AS avg_goals,
    (SELECT ROUND(avg(home_goal + away_goal), 2) 
     FROM match
     WHERE season = '2013/2014') AS overall_avg

FROM 

league AS l
LEFT JOIN match AS m
ON l.id = m.country_id

WHERE m.season = '2013/2014'
GROUP BY league
    ;""", conn)
leages_by_season


# In[ ]:


leages_by_season = pd.read_sql("""SELECT 
                                        League.name AS league_name, 
                                        season,
                                
                                        avg(home_team_goal) AS avg_home_team_scors, 
                                        avg(away_team_goal) AS avg_away_team_goals, 
                                        avg(home_team_goal-away_team_goal) AS avg_goal_dif, 
                                        avg(home_team_goal+away_team_goal) AS avg_goals, 
                                        sum(home_team_goal+away_team_goal) AS total_goals
                                     
                                FROM Match
                             
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                               
                                GROUP BY  League.name, season
                              
                                ;""", conn)
leages_by_season


# # Let's do some basic analytics
# Here we are starting to look at the data at more aggregated level. Instead of looking on the raw data we will start to grouping it to different levels we want to examine.
# In this example, we will base it on the previous query, remove the match and date information, and look at it at the country-league-season level.
# 
# The functionality we will use for that is GROUP BY, that comes between the WHERE and ORDER
# 
# Once you chose what level you want to analyse, we can devide the select statement to two:
# * Dimensions - those are the values we describing, same that we group by later.
# * Metrics - all the metrics have to be aggregated using functions.. 
# The common functions are: sum(), count(), count(distinct), avg(), min(), max()
# 
# Note - it is very important to use the same dimensions both in the select, and in the GROUP BY. Otherwise the output might be wrong.
# 
# Another functionality that can be used after grouping, is HAVING. This adds another layer of filtering the data, this time the output of the table **after** the grouping. A lot of times it is used to clean the output.
# 

# In[ ]:


leages_by_season = pd.read_sql("""SELECT Country.name AS country_name, 
                                        League.name AS league_name, 
                                        season,
                                        count(distinct stage) AS number_of_stages,
                                        count(distinct HT.team_long_name) AS number_of_teams,
                                        avg(home_team_goal) AS avg_home_team_scors, 
                                        avg(away_team_goal) AS avg_away_team_goals, 
                                        avg(home_team_goal-away_team_goal) AS avg_goal_dif, 
                                        avg(home_team_goal+away_team_goal) AS avg_goals, 
                                        sum(home_team_goal+away_team_goal) AS total_goals                                       
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                WHERE country_name in ('Spain', 'Germany', 'France', 'Italy', 'England')
                                GROUP BY Country.name, League.name, season
                                HAVING count(distinct stage) > 10
                                ORDER BY Country.name, League.name, season DESC
                                ;""", conn)
leages_by_season


# In[ ]:


df = pd.DataFrame(index=np.sort(leages_by_season['season'].unique()), columns=leages_by_season['country_name'].unique())

df.loc[:,'Germany'] = list(leages_by_season.loc[leages_by_season['country_name']=='Germany','avg_goals'])
df.loc[:,'Spain']   = list(leages_by_season.loc[leages_by_season['country_name']=='Spain','avg_goals'])
df.loc[:,'France']   = list(leages_by_season.loc[leages_by_season['country_name']=='France','avg_goals'])
df.loc[:,'Italy']   = list(leages_by_season.loc[leages_by_season['country_name']=='Italy','avg_goals'])
df.loc[:,'England']   = list(leages_by_season.loc[leages_by_season['country_name']=='England','avg_goals'])

df.plot(figsize=(12,5),title='Average Goals per Game Over Time')


# In[ ]:


df = pd.DataFrame(index=np.sort(leages_by_season['season'].unique()), columns=leages_by_season['country_name'].unique())

df.loc[:,'Germany'] = list(leages_by_season.loc[leages_by_season['country_name']=='Germany','avg_goal_dif'])
df.loc[:,'Spain']   = list(leages_by_season.loc[leages_by_season['country_name']=='Spain','avg_goal_dif'])
df.loc[:,'France']   = list(leages_by_season.loc[leages_by_season['country_name']=='France','avg_goal_dif'])
df.loc[:,'Italy']   = list(leages_by_season.loc[leages_by_season['country_name']=='Italy','avg_goal_dif'])
df.loc[:,'England']   = list(leages_by_season.loc[leages_by_season['country_name']=='England','avg_goal_dif'])

df.plot(figsize=(12,5),title='Average Goals Difference Home vs Out')


# # Query Run Order
# Now that we are familiar with most of the functionalities being used in a query, it is very important to understand the order that code runs.
# 
# First, order of how we write it (reminder):
# * SELECT
# * FROM
# * JOIN
# * WHERE
# * GROUP BY
# * HAVING
# * ORDER BY
# * LIMIT
# 
# Now, the actul order that things happens.
# First, you can think of this part as creating a new temporal table in the memory:
# * Define which tables to use, and connect them (FROM + JOIN)
# * Keep only the rows that apply to the conditions (WHERE)
# * Group the data by the required level (if need) (GROUP BY)
# * Choose what information you want to have in the new table. It can have just rawdata (if no grouping), or combination of dimensions (from the grouping), and metrics
# Now, you chose that to show from the table:
# * Order the output of the new table (ORDER BY)
# * Add more conditions that would filter the new created table (HAVING) 
# * Limit to number of rows - would cut it according the soring and the having filtering (LIMIT)
# 

# # Sub Queries and Functions 
# 
# Using subqueries is an essential tool in SQL, as it allows manipulating the data in very advanced ways without the need of any external scripts, and especially important when your tables stractured in such a way that you can't be joined directly.
# 
# In our example, I'm trying to join between a table that holds player's basic details (name, height, weight), to a table that holds more attributes. The problem is that while the first table holds one row for each player, the key in the second table is player+season, so if we do a regular join, the result would be a Cartesian product, and each player's basic details would appear as many times as this player appears in the attributes table. The problem with of course is that the average would be skewed towards players that appear many times in the attribute table.
# 
# The solution, is to use a subquery.  We would need to group the attributes table, to a different key - player level only (without season). Of course we would need to decide first how we would want to combine all the attributes to a single row. I used average, but one can also decide on maximum, latest season and etc. 
# Once both tables have the same keys, we can join them together (think of the subquery as any other table, only temporal), knowing that we won't have duplicated rows after the join.
# 
# In addition, you can see here two examples of how to use functions:
# * Conditional function is an important tool for data manipulation. While IF statement is very popular in other languages, SQLite is not supporting it, and it's implemented using CASE + WHEN + ELSE statement. 
# As you can see, based on the input of the data, the query would return different results.
# 
# * ROUND - straight sorward.
# Every SQL languages comes with a lot of usefull functions by default.

# In[ ]:


players_height = pd.read_sql("""SELECT CASE
                                        WHEN ROUND(height)<165 then 165
                                        WHEN ROUND(height)>195 then 195
                                        ELSE ROUND(height)
                                        END AS calc_height, 
                                        COUNT(height) AS distribution, 
                                        (avg(PA_Grouped.avg_overall_rating)) AS avg_overall_rating,
                                        (avg(PA_Grouped.avg_potential)) AS avg_potential,
                                        AVG(weight) AS avg_weight 
                            FROM PLAYER
                            LEFT JOIN (SELECT Player_Attributes.player_api_id, 
                                        avg(Player_Attributes.overall_rating) AS avg_overall_rating,
                                        avg(Player_Attributes.potential) AS avg_potential  
                                        FROM Player_Attributes
                                        GROUP BY Player_Attributes.player_api_id) 
                                        AS PA_Grouped ON PLAYER.player_api_id = PA_Grouped.player_api_id
                            GROUP BY calc_height
                            ORDER BY calc_height
                                ;""", conn)
players_height


# In[ ]:


players_height.plot(x=['calc_height'],y=['avg_overall_rating'],figsize=(12,5),title='Potential vs Height')


# In[ ]:


att = pd.read_sql("""SELECT *
                        FROM Player_Attributes;""", conn)
att

