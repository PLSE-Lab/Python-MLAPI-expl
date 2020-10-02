#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

database = "../input/database.sqlite"  #Insert path here


# **1. show tables**

# In[ ]:


conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table'
                     """, conn)
tables


# **2. select 'League' and 'Player' tables**

# In[ ]:


leagues = pd.read_sql(''' SELECT * 
                          FROM League
                      ''', conn)
leagues


# In[ ]:


players = pd.read_sql(''' SELECT * 
                          FROM Player
                      ''', conn)
players.head(10)


# ## ORDER BY
# 
# **3. ORDER BY team_fifa_api_id**
# 
#     Select (* namely everything)
# 
#     From (Table_Name)
# 
#     Where (give a condition)
# 
#     Order by (order by the given feature)
#    

# In[ ]:


teams = pd.read_sql(''' SELECT * 
                        FROM Team
                        WHERE team_fifa_api_id < 20
                        ORDER BY team_fifa_api_id
                    ''', conn)
teams


# ## WITH...AS
# 
# **4. Create a new table called *premier_league*:** When the query is too long, we can create sub tables in order for decreasing the complexity of the query. Here it is not necessary, but just to show how it works
# 
#    premier_league is a new table (so I can use it in SQL code)
# 
#    p_league is a new dataframe

# In[ ]:


p_league = pd.read_sql(''' WITH premier_league AS 
                        (
                        SELECT * 
                        FROM Team
                        WHERE team_fifa_api_id < 20
                        )
                        
                        SELECT *
                        FROM premier_league
                        ORDER BY team_fifa_api_id
                    ''', conn)
p_league


# 4.1. premier_league except Arsenal, Liverpool and Chelsea

# In[ ]:


exclude_specific_teams = pd.read_sql(''' WITH premier_league AS 
                        (
                        SELECT * 
                        FROM Team
                        WHERE team_fifa_api_id < 20
                        )
                        
                        SELECT *
                        FROM premier_league
                        WHERE team_long_name NOT IN ('Arsenal', 'Liverpool', 'Chelsea')
                    ''', conn)
exclude_specific_teams


# 4.2 in premier_league, find team_long_name without space charecter

# In[ ]:


one_word_teams = pd.read_sql(''' WITH premier_league AS 
                        (
                        SELECT * 
                        FROM Team
                        WHERE team_fifa_api_id < 20
                        )
                        
                        SELECT *
                        FROM premier_league
                        WHERE team_long_name NOT LIKE '% %'
                    ''', conn)
one_word_teams


# 4.3 team_long_name that does not contain the word 'united'

# In[ ]:


not_united_teams = pd.read_sql(''' WITH premier_league AS 
                        (
                        SELECT * 
                        FROM Team
                        WHERE team_fifa_api_id < 20
                        )
                        
                        SELECT *
                        FROM premier_league
                        WHERE team_long_name NOT LIKE '%united%'
                    ''', conn)
not_united_teams


# **5. tall AND short players **

# In[ ]:


players = pd.read_sql(''' SELECT * 
                          FROM Player
                      ''', conn)
players.head()


# In[ ]:


tall_and_short_players = pd.read_sql(''' SELECT * 
                                         FROM Player
                                         WHERE height > 200 OR height < 170
                                     ''', conn)
tall_and_short_players.head()


# **6. tall OR short and young players **

# In[ ]:


tall_or_young_and_short_players = pd.read_sql(''' SELECT player_name, birthday, height   
                                                  FROM Player
                                                  WHERE height > 200 OR (height < 170 AND birthday > '1996')
                                              ''', conn)
tall_or_young_and_short_players


# ## 7. Math in SQL

# 7.1. simple math operations

# In[ ]:


one_plus_two = pd.read_sql (''' SELECT 1+2 ''', conn)
one_plus_two


# **7.2**. **give a name to a new column**
# 
#    using AS 

# In[ ]:


one_plus_two = pd.read_sql (''' SELECT 1+2 AS Addition''', conn)
one_plus_two


# In[ ]:


five_minus_2 = pd.read_sql (''' SELECT 5 - 2 AS Substraction''', conn)
five_minus_2


# In[ ]:


five_times_2 = pd.read_sql (''' SELECT 5 * 2 AS Multiplication''', conn)
five_times_2


# In[ ]:


five_divide_by_2 = pd.read_sql (''' SELECT 5 / 2 AS Division''', conn)
five_divide_by_2


# In[ ]:


five_divide_by_2_point_zero = pd.read_sql (''' SELECT 5 / 2.0 AS Division''', conn)
five_divide_by_2_point_zero


# In[ ]:


five_modulo_2 = pd.read_sql (''' SELECT 5 % 2 AS Modulo''', conn)
five_modulo_2


# - Calculate players power: height + weight

# In[ ]:


players_power = pd.read_sql(''' SELECT player_name, height, weight, height + weight AS power  
                                FROM Player
                            ''', conn)
players_power.head()


# - Calculate players power = 0.7 x height + 0.3 x weight
#     
#     and sort them in descending order 

# In[ ]:


players_power = pd.read_sql(''' SELECT player_name, height, weight, (0.7 * height) + (0.3 * weight) AS power  
                                FROM Player
                                ORDER BY power DESC
                            ''', conn)
players_power.head()


# **8. String concatenating**

# In[ ]:


hello_world = pd.read_sql(''' SELECT 'Hello ' || 'World' AS expression
                          ''', conn)
hello_world


# In[ ]:


team_attributes = pd.read_sql(''' SELECT * 
                                  FROM Team_Attributes
                              ''', conn)
team_attributes.head()


# 8.1 **concatenate** buildUpPlaySpeedClass, buildUpPlayDribblingClass, buildUpPlayPassingClass

# In[ ]:


string_concat = pd.read_sql(''' SELECT team_fifa_api_id, buildUpPlaySpeedClass || ' ' || buildUpPlayDribblingClass || ' ' || buildUpPlayPassingClass
                                AS Speed_Dripling_Passing  
                                FROM Team_Attributes
                            ''', conn)
string_concat.head()


# 8.2 concatenating string and number

# In[ ]:


string_int_concat = pd.read_sql(''' SELECT team_fifa_api_id, buildUpPlaySpeedClass || ' ' || buildUpPlaySpeed
                                    AS Speed_Class_and_Values  
                                    FROM Team_Attributes
                                ''', conn)
string_int_concat.head()


# 8.3 Order by using more than 1 feature

# In[ ]:


string_int_concat = pd.read_sql(''' SELECT team_fifa_api_id, buildUpPlaySpeedClass || ' ' || buildUpPlaySpeed
                                    AS Speed_Class_and_Values  
                                    FROM Team_Attributes
                                    ORDER BY team_fifa_api_id, Speed_Class_and_Values 
                                ''', conn)
string_int_concat.head(10)


# **9 Distinct or unique values in a column**

# In[ ]:


distinct_speed_classes = pd.read_sql(''' SELECT DISTINCT buildUpPlaySpeedClass
                                         FROM Team_Attributes
                                     ''', conn)
distinct_speed_classes


# 9..1  **DISTINCT: ** it finds distinct rows chosen by select keyword
#     
#    for example there is only one column with buildUpPlaySpeedClass = Balanced **AND** team_fifa_api_id = 434 

# In[ ]:


distinct_speed_classes_and_team_id = pd.read_sql(''' SELECT DISTINCT buildUpPlaySpeedClass, team_fifa_api_id
                                                     FROM Team_Attributes
                                                 ''', conn)
distinct_speed_classes_and_team_id.head(10)


# **10 GROUP BY**

# In[ ]:


group_by_speed_classes = pd.read_sql(''' SELECT buildUpPlaySpeedClass, COUNT(buildUpPlaySpeed) AS count
                                         FROM Team_Attributes
                                         GROUP BY buildUpPlaySpeedClass
                                     ''', conn)
group_by_speed_classes


# **11. TO FIND NULL VALUES**
# 
#    when we want to seach null values, we use **IS NULL** 
#    
#    do not use:  = NULL 

# In[ ]:


null_dribling = pd.read_sql(''' SELECT DISTINCT team_fifa_api_id, buildUpPlayDribblingClass, buildUpPlayDribbling
                                FROM Team_Attributes
                                WHERE buildUpPlayDribbling IS NULL
                            ''', conn)
null_dribling.head(10)


# **IS NOT NULL**

# In[ ]:


not_null_dribling = pd.read_sql(''' SELECT DISTINCT team_fifa_api_id, buildUpPlayDribblingClass, buildUpPlayDribbling
                                    FROM Team_Attributes
                                    WHERE buildUpPlayDribbling IS NOT NULL
                                ''', conn)
not_null_dribling.head(10)


# check also whether we have **EMPTY** places
# 
# in this case we do not have any

# In[ ]:


empty_dribling = pd.read_sql(''' SELECT DISTINCT team_fifa_api_id, buildUpPlayDribblingClass, buildUpPlayDribbling
                                 FROM Team_Attributes
                                 WHERE buildUpPlayDribbling = ' '
                             ''', conn)
empty_dribling.head(10)


# EMPTY together with NULL values

# In[ ]:


empty_null_dribling = pd.read_sql(''' SELECT DISTINCT team_fifa_api_id, buildUpPlayDribblingClass, buildUpPlayDribbling
                                      FROM Team_Attributes
                                      WHERE buildUpPlayDribbling = ' ' OR buildUpPlayDribbling IS NULL
                                  ''', conn)
empty_null_dribling.head(10)


# **12. CASE**

# **12.1**

# In[ ]:


team_attributes = pd.read_sql(''' SELECT *
                                  FROM Team_Attributes
                              ''', conn)
team_attributes.head()


# In[ ]:


case_ = pd.read_sql(''' SELECT team_api_id, buildUpPlayDribbling, buildUpPlayDribblingClass, 
                                     CASE buildUpPlayDribblingClass
                                         WHEN 'Little' THEN 'L'
                                         WHEN 'Normal' THEN 'N'
                                         ELSE 'O'
                                     END AS class_name_that_I_created
                        FROM Team_Attributes
                     ''', conn)
case_.head()


# **12.2**

# In[ ]:


players = pd.read_sql(''' SELECT * 
                          FROM Player
                      ''', conn)
players.head()


# In[ ]:


players_height_class = pd.read_sql(''' SELECT player_name, height,
                                            CASE
                                                WHEN height < 170.00 THEN 'Short'
                                                WHEN height BETWEEN 170.00 AND 185.00 THEN 'Medium'
                                                WHEN height > 185.00 THEN 'Tall'    
                                            END AS height_class
                                       FROM Player
                                   ''', conn)
players_height_class.head(10)


# **13. SUM**

# In[ ]:


sum_weight = pd.read_sql(''' SELECT SUM(weight) AS total_weight
                             FROM Player
                         ''', conn)
sum_weight.head()


# In[ ]:


young_weight = pd.read_sql(''' SELECT SUM(weight) AS young_weight
                               FROM Player
                               WHERE birthday > 1996
                           ''', conn)
young_weight.head()


# **14. AVERAGE: AVG**

# In[ ]:


young_AVG_weight = pd.read_sql(''' SELECT AVG(weight) AS young_avg_weight
                                   FROM Player
                                   WHERE birthday > 1996
                               ''', conn)
young_AVG_weight.head()


# **15. COUNT**

# In[ ]:


count_young = pd.read_sql(''' SELECT COUNT(id) AS Count_Youngs
                                     FROM Player
                                     WHERE birthday > 1996
                                 ''', conn)
count_young.head()


# In[ ]:


count_all = pd.read_sql(''' SELECT COUNT(*) AS Count_All
                              FROM Player
                         ''', conn)
count_all.head()


# **16. MIN and MAX**

# In[ ]:


Shortest_player = pd.read_sql(''' SELECT MIN(height) AS shortest_player
                                  FROM Player
                              ''', conn)
Shortest_player.head()


# In[ ]:


Tallest_player = pd.read_sql(''' SELECT MAX(height) AS tallest_player
                                 FROM Player
                             ''', conn)
Tallest_player.head()


# In[ ]:


rank_heights = pd.read_sql(''' SELECT id, height AS rank_heights
                               FROM Player
                               ORDER BY height DESC
                             ''', conn)
rank_heights.head()


# In[ ]:


Tallest_and_Shortest_player = pd.read_sql(''' SELECT MAX(height) AS tallest_player, 
                                                     MIN(height) AS shortest_player
                                              FROM Player
                                          ''', conn)
Tallest_and_Shortest_player.head()


# **17. COUNT DISTINCT values**

# In[ ]:


count_distinct_speed_classes = pd.read_sql(''' SELECT COUNT(DISTINCT buildUpPlaySpeedClass) 
                                               AS count_distint_classes
                                               FROM Team_Attributes
                                           ''', conn)
count_distinct_speed_classes


# **18. INNER JOIN (SECOND TABLE NAME) ON**
# 
#    - it is used to join two tables by utilizing the same column in the two tables
#    - for example, both 'Player' table and 'Player_Attributes' table have 'player_api_id' column
#    - we can use this column to find the name of the player depicted in the 'Player_Attributes' table by joining the two table

# In[ ]:


players = pd.read_sql(''' SELECT * 
                          FROM Player
                      ''', conn)
players.head()


# In[ ]:


player_attributes = pd.read_sql(''' SELECT * 
                                    FROM Player_Attributes
                                ''', conn)
player_attributes.head(3)


# In[ ]:


join_player_and_player_attributes_tables = pd.read_sql(''' SELECT 
                                                                p.player_name, 
                                                                p.birthday, 
                                                                pa.overall_rating, 
                                                                pa.potential
                                                            FROM Player AS p
                                                            INNER JOIN Player_Attributes AS pa 
                                                            ON p.player_api_id = pa.player_api_id
                                                       ''', conn)
join_player_and_player_attributes_tables.head()


# **19. Question : How many players do we have that were born in the same years **
# - **strftime() **
# 
#     - start_date,strftime('%Y',start_date) as "Year",
#     - strftime('%m',start_date) as "Month",
#     - strftime('%d',start_date) as "Day"

# In[ ]:


count_players_with_the_same_bitrhday = pd.read_sql(''' SELECT 
                                                            COUNT(p.player_name) AS number_of_players, 
                                                            strftime('%Y',p.birthday) AS "year_born"
                                                        FROM Player AS p
                                                        INNER JOIN Player_Attributes AS pa 
                                                        ON p.player_api_id = pa.player_api_id
                                                        GROUP BY year_born
                                                   ''', conn)
count_players_with_the_same_bitrhday.head(8)


# In[ ]:


count_players_with_the_same_bitrhday_and = pd.read_sql('''SELECT 
                                                            COUNT(p.player_name) AS number_of_players, 
                                                            strftime('%Y',p.birthday) AS "year_born",
                                                            MIN(pa.overall_rating) AS min_overall_rating,
                                                            MAX(pa.overall_rating) AS max_overall_rating, 
                                                            AVG(pa.overall_rating) AS average_overall_rating
                                                        FROM Player AS p
                                                        INNER JOIN Player_Attributes AS pa 
                                                        ON p.player_api_id = pa.player_api_id
                                                        GROUP BY year_born
                                                   ''', conn)
count_players_with_the_same_bitrhday_and.head()


# **20. Question: Number of players born in the same year, which have overall_rating greater than 80**

# In[ ]:


num_of_players_same_year_min_overall_rating = pd.read_sql('''SELECT 
                                                                COUNT(p.player_name) AS number_of_players, 
                                                                strftime('%Y',p.birthday) AS "year_born"
                                                            FROM Player AS p
                                                            INNER JOIN Player_Attributes AS pa 
                                                            ON p.player_api_id = pa.player_api_id
                                                            WHERE pa.overall_rating > 90
                                                            GROUP BY year_born
                                                       ''', conn)
num_of_players_same_year_min_overall_rating.head()


# **21. Same question in 19. But this time, what if we would like to select players born after 1990**
# - use **HAVING** 

# In[ ]:


count_young_players_with_the_same_bitrhday = pd.read_sql(''' SELECT 
                                                                COUNT(p.player_name) AS number_of_players, 
                                                                strftime('%Y',p.birthday) AS "year_born"
                                                             FROM Player AS p
                                                             INNER JOIN Player_Attributes AS pa 
                                                             ON p.player_api_id = pa.player_api_id
                                                             GROUP BY year_born
                                                             HAVING year_born > '1990'
                                                          ''', conn)
count_young_players_with_the_same_bitrhday.head()


# **22. plot number of players born in the same year**

# In[ ]:


import matplotlib.pyplot as plt
x = count_players_with_the_same_bitrhday.year_born
y = count_players_with_the_same_bitrhday.number_of_players
plt.figure(figsize = (12,7))
plt.plot(x,y)
plt.xlabel('birthyear')
plt.ylabel('number of players')
plt.title('number of players born in the same year')
plt.xticks(rotation = 90)
plt.grid()
plt.show()


# **23. plot average overall rating of players born in the same year**

# In[ ]:


x1 = count_players_with_the_same_bitrhday_and.year_born
y1 = count_players_with_the_same_bitrhday_and.average_overall_rating
plt.figure(figsize = (12,7))
plt.plot(x1,y1)
plt.xlabel('birthyear')
plt.ylabel('average overall rating')
plt.title('average overall rating of players born in the same year')
plt.xticks(rotation = 90)
plt.grid()
plt.show()


# ## 24. SUMMARY

# **Write your queries in this order **
# 
# - SELECT *column1, column2, ...*
# - FROM *table name 1*
# - INNER JOIN *table name 2* ON *equation*
# - WHERE *condition*
# - GROUP BY *column1, column2,...*
# - HAVING *condition*
# - ORDER BY  *column1, column2, ...*  
# - LIMIT *n*
