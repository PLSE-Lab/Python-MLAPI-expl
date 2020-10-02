#!/usr/bin/env python
# coding: utf-8

# <h1>NFL ANALYSIS - Series 1: Obtaining the Final Results for Each Game</h1>
# <body>
#     <p>This Data origionates from the curtousy of Kaggle user Max Horowitz (@ https://www.kaggle.com/maxhorowitz). He has used nflscrapR to obtain the data and also represents https://www.cmusportsanalytics.com/. nflscrapeR is an R package engineered to scrape NFL data by researchers Maksim Horowitz, Ron Yurko, and Sam Ventura.</p>
#     <p>**Series 1 - Obtaining The Final Result for Each Game:** In this part of the series, I'm going to undergo an ETL processing approach of the data set. There are various issues with calculating the final scores for each game and I will show my methodology to overcome the problems that are associated with doing so.</p>
# </body>

# **This notebook's purpose is to explore NFL data from 2009-2017. The goal is to hopefully provide useful analysis for others to use or to provide useful code for others to learn from.**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# <h1>
# Importing Data
# </h1>

# In[ ]:


df = pd.read_csv("../input/NFL Play by Play 2009-2017 (v4).csv")
df.head(10)


# <h1>
# Exploratory Analysis
# </h1>

# <h3>
# Looking at the Data
# </h3>
# <body>
#     We can see that the data is particularily encompassing. 102 Columns of data is a lot more data than we'll need while building potential models. This is because more attributes can typically cause over-fitting for any method we decide to impliment.
#     </body>

# In[ ]:


df.info()


# In[ ]:


print("Rows: ",len(df))


# <h3>
# Further Exploration
# </h3>
# <body>
#     <p>Taking a look at turnovers:</p>
# </body>

# In[ ]:


total_interceptions = len(df[df.InterceptionThrown == 1])
print("Total Count of Interceptions Thrown: ", total_interceptions)

total_fumbles = len(df[df.Fumble == 1])
print("Total Count of Fumbles: ", total_fumbles)

total_punts_blocked = len(df[df.PuntResult == 'Blocked'])
print("Total Count of Blocked Punts: ", total_punts_blocked)

total_fg_unsuccessful = len(df[(df.FieldGoalResult == 'Blocked') | (df.FieldGoalResult == 'No Good')])
print("Total Missed/Blocked Field Goals: ", total_fg_unsuccessful)

# because I'm not taking into account the hefty logic to determine turnover on downs -- "Giveaways"
total_giveaways = total_interceptions + total_fumbles + total_punts_blocked + total_fg_unsuccessful
print("Total Giveaways: ", total_giveaways)


# create a dict object containing the above calculations
giveaways = {'Interceptions': [total_interceptions],
             'Fumbles': [total_fumbles],
             'Blocked Punts': [total_punts_blocked],
             'Missed/Blocked FG': [total_fg_unsuccessful]}
giveaways_df = pd.DataFrame.from_dict(giveaways)

# plot the results in a simple bar chart
giveaways_df.plot(kind='bar')


# <body>
#     Over the 9 seasons from 2009-2017, the most common turnover was a fumble.
# </body>

# <h1>Preparing the Data - Data Cleansing</h1>
# <body>
#     <p>The data I'm working with is particulariy meaty - 102 columns is a ton of data to work with. The rows and columns hold various indicators that I want to extract for further analysis. One indicator I'm particularily interested in and believe to be sufficiently important to endeavor the process of data cleansing for is the score at the end of the game.</p>
#     <p>There are rows in the column 'PlayType' that indicates the 'End of Game', but unfortunately this is only available for 1917 or the 2304 total games. The games that do not have this column vary in a ton of different ways, which means we'll have to endear quite an extensive cleaning process involving a lot of logic.</p>
# </body>

# <h3>Franchises that Changed Citys</h3>
# <body>
#     <p>Because teams like the Las Angeles Chargers and Las Angeles Rams used to be the San Diego Chargers and St.Louis Rams respectively, we must alter the records to be consistent:</p>
# </body>

# In[ ]:


# update data for consistent team names to just label franchises instead of cities to avoid outliers/ambiguity:

# LA --> LAR (avoid ambiguity with LAC)
df.loc[df.posteam == 'LA', 'posteam'] = 'LAR'
df.loc[df.DefensiveTeam == 'LA', 'DefensiveTeam'] = 'LAR'
df.loc[df.HomeTeam == 'LA', 'HomeTeam'] = 'LAR'
df.loc[df.AwayTeam == 'LA', 'AwayTeam'] = 'LAR'
df.loc[df.RecFumbTeam == 'LA', 'RecFumbTeam'] = 'LAR'
df.loc[df.PenalizedTeam == 'LA', 'PenalizedTeam'] = 'LAR'
df.loc[df.SideofField == 'LA', 'PenalizedTeam'] = 'LAR'

# STL --> LAR
df.loc[df.posteam == 'STL', 'posteam'] = 'LAR'
df.loc[df.DefensiveTeam == 'STL', 'DefensiveTeam'] = 'LAR'
df.loc[df.HomeTeam == 'STL', 'HomeTeam'] = 'LAR'
df.loc[df.AwayTeam == 'STL', 'AwayTeam'] = 'LAR'
df.loc[df.RecFumbTeam == 'STL', 'RecFumbTeam'] = 'LAR'
df.loc[df.PenalizedTeam == 'STL', 'PenalizedTeam'] = 'LAR'
df.loc[df.SideofField == 'STL', 'PenalizedTeam'] = 'LAR'

# SD --> LAC
df.loc[df.posteam == 'SD', 'posteam'] = 'LAC'
df.loc[df.DefensiveTeam == 'SD', 'DefensiveTeam'] = 'LAC'
df.loc[df.HomeTeam == 'SD', 'HomeTeam'] = 'LAC'
df.loc[df.AwayTeam == 'SD', 'AwayTeam'] = 'LAC'
df.loc[df.RecFumbTeam == 'SD', 'RecFumbTeam'] = 'LAC'
df.loc[df.PenalizedTeam == 'SD', 'PenalizedTeam'] = 'LAC'
df.loc[df.SideofField == 'SD', 'PenalizedTeam'] = 'LAC'

# JAC --> JAX
df.loc[df.posteam == 'JAC', 'posteam'] = 'JAX'
df.loc[df.DefensiveTeam == 'JAC', 'DefensiveTeam'] = 'JAX'
df.loc[df.HomeTeam == 'JAC', 'HomeTeam'] = 'JAX'
df.loc[df.AwayTeam == 'JAC', 'AwayTeam'] = 'JAX'
df.loc[df.RecFumbTeam == 'JAC', 'RecFumbTeam'] = 'JAX'
df.loc[df.PenalizedTeam == 'JAC', 'PenalizedTeam'] = 'JAX'
df.loc[df.SideofField == 'JAC', 'PenalizedTeam'] = 'JAX'


# <h1>Computing Turnovers</h1>
# <body>
#     <p>This is important to know before breaking anything else down. Here, I'm using a lambda function to check for various plays that indicate the offensive team explicitly gave the ball to the other team. The only instance I am not checking for in this attribute is whether a team turned the ball over on downs. This is because a defensive team cannot score a touchdown on a turnover on downs, vs. the others I'm checking for</p>
# </body>

# In[ ]:


# update data to have an attribute for turnovers:
df['Turnover'] = df.apply(lambda x: 1 
                                 if ((x.InterceptionThrown == 1) | 
                                     (x.Fumble == 1) |
                                     (x.FieldGoalResult == 'Blocked') |
                                     (x.FieldGoalResult == 'No Good') |
                                     (x.PuntResult == 'Blocked'))
                                 else 0, axis=1)


# <h1>
# Final Outcomes for Games
# </h1>
# <body>
#     <p>Before we dive into segmenting our data, we'll need to aquire the final scores for each game. Because the PlayType attribute has an 'End of Game' entry to signify the final results of a game, we can use that to appropriately get the final results; but the issue is that the scores are held in the "PosTeamScore" and "DefTeamScore" columns, which are 'NA' for the columns where PLayType = 'End of Game' because there isn't a possession team or a defensive team. To appropriately get the scores in a subset, we will have to clean the data and appropriatley match the scores with the home and away teams.</p>
#     <p>To begin, the columns indicating score are PosTeamScore and DefTeamScore. One way we can alter the data to show score is by creating 2 new columns that indicate HomeTeamScore and AwayTeamScore, then use that to apply logic in order to populate the 'End of Game' PlayTypes to essentially reflect the final score of each game.</p>
#     <p>Below, I am using a lambda function to properly place the Scores into new columns. The logic is: </p>
#     <p>**If:** the PosTeam is the same as the HomeTeam, then copy the PosTeamScore into the HomeTeamScore.</p>
#     <p>**Else:** Copy the DefTeamScore into the HomeTeamScore column (repeated for AwayTeam as well)</p>
# </body>

# In[ ]:


# disable chained assignments --> for the logic I'm using this is simply an annoying warning and is populating correctly
pd.options.mode.chained_assignment = None 

# minimze the dataset for computational efficiency
df_scores = df[(df.qtr >= 4)]

# copy the columns(attributes) we declare in the list to a new dataframe to modify
results_attributes = ['HomeTeam','AwayTeam','posteam','PosTeamScore','DefensiveTeam','DefTeamScore','GameID','Date','qtr','PlayType','sp','Touchdown','FieldGoalResult','ExPointResult','TwoPointConv','Turnover','Safety','TimeSecs','Drive']
df_scores = df_scores[results_attributes]

# apply the lambda funstion to copy the PosTeamScores/DefTeamScores into HomeTeamScores and AwayTeam Scores
df_scores['HomeTeamScore'] = df_scores.apply(lambda x: x.PosTeamScore if x.HomeTeam == x.posteam else x.DefTeamScore, axis=1)
df_scores['AwayTeamScore'] = df_scores.apply(lambda x: x.PosTeamScore if x.AwayTeam == x.posteam else x.DefTeamScore, axis=1)

results_attributes = ['HomeTeam','HomeTeamScore','AwayTeam','AwayTeamScore','posteam','PosTeamScore','DefensiveTeam','DefTeamScore','GameID','Date','qtr','PlayType','sp','Touchdown','FieldGoalResult','ExPointResult','TwoPointConv','Turnover','Safety','TimeSecs','Drive']
df_scores = df_scores[results_attributes]
df_scores.head(20)


# <body>
# <p>We can see that the scores were correctly copied into new columns. Now, lets deal with those pesky 'NA' values. To do this, I'm finding a list of indices for the rows containing a "PlayType" thats indicating the End of Game. Then, by subtracting 1 from each index in the list I can access the row that proceeds the End of Game row. This allows me to copy the values in those rows to the End of Game Rows.</p>
# </body>

# In[ ]:


# get a list of the indices for the rows that indicate the End of Game
idx = df_scores[df_scores['PlayType'] == 'End of Game'].index.tolist()

# subtract 1 from the indices to use for accessing the row above the End of Game row
idx[:] = [x - 1 for x in idx]

# iterate over the list to access the values and copy them into the End of Game rows
for x in idx:
    home_score = df_scores.loc[x, 'HomeTeamScore']
    away_score = df_scores.loc[x, 'AwayTeamScore']
    y = x + 1
    if((df_scores.loc[y, 'PlayType'] == 'End of Game')):
        df_scores.loc[y, 'HomeTeamScore'] = home_score
        df_scores.loc[y, 'AwayTeamScore'] = away_score

# subset the dataframe to only include end of game results
Final_Results = df_scores[df_scores['PlayType'] == 'End of Game']
Final_Results.head(5)


# <h4>Adjusting Final Scores for instances where there is not an 'End of Game' indication</h4>
# <body>
#     <p>Because not all games end with the incidation of 'End of Game', we must dive further and find the other final scores. This is fairly complicated because of the various end results. A game can end in a Field Goal, Touchdown, Extra Point, Interception, Defensive TD, QB Kneel, Pass, etc.</p>
#     <p>Where there is no End of Game, we must look into other ways achieving the final solution. I think the easiest route is to drop the duplicated values and obtain the last indexed value to obtain these plays, which is done at the beginning of the below cell.
#     </p>
#     <p>Before I go on, it is also important to know that when there is a scoring play to end a game, the information in the 'scores' attribute is not updated before moving onto the final game. Due to this, we must mannually update and incriment the team scores depending on how the team has scored. We must also consider Defensive touchdowns, so the turnover attribute I had created earlier will come in handy.</p>
#     <p>Below, I will first using indexing to my advantage to Read, Update, and Delete records that need to be changed. This is the first, and the more complicated method to clean this dataset up:</p>
# </body>

# In[ ]:


# the GameID's that are not already in Final Results - because we already have the final scores for those games
df_scores = df_scores[~df_scores.GameID.isin(Final_Results.GameID)]

# Lets filter to only look at scoring plays first
df_sp = df_scores[df_scores.sp == 1]

#remove dups
df_sp = df_sp.drop_duplicates(subset=['GameID'], keep='last')

# function to add points to the Defensive Team's score whether they are Home or Away
def update_def_score(row):
    if (row['DefensiveTeam'] == row['HomeTeam']):
        if (row.Safety == 1): # There is a safety
            row.HomeTeamScore = row.HomeTeamScore+2
        else: row.HomeTeamScore = row.HomeTeamScore+6
    else:
        if (row.Safety == 1): # There is a safety
            row.AwayTeamScore = row.AwayTeamScore+2
        else: row.AwayTeamScore = row.AwayTeamScore+6
        return row

def update_score(row):
    if (row['posteam'] == row['AwayTeam']): #posteam is home team
        if (row.Touchdown == 1): # Touchdown ends game
            row.AwayTeamScore = row.AwayTeamScore+6
        elif ((row.PlayType == 'Field Goal')&(row.FieldGoalResult == 'Good')): # Field Goal to win game
            row.AwayTeamScore = row.AwayTeamScore+3
        elif ((row.PlayType == 'Extra Point')&(row.ExPointResult == 'made')): # Extra Point seals W
            row.AwayTeamScore = row.AwayTeamScore+1
        elif (row.TwoPointConv == 'Success'):
            row.AwayTeamScore = row.AwayTeamScore+2 # 2-pt conversion successful to win game
    elif(row['posteam'] == row['HomeTeam']): # posteam is away team
        if (row.Touchdown == 1): # Touchdown ends game
            row.HomeTeamScore = row.HomeTeamScore+6
        elif ((row.PlayType == 'Field Goal')&(row.FieldGoalResult == 'Good')): # Field Goal to win game
            row.HomeTeamScore = row.HomeTeamScore+3
        elif ((row.PlayType == 'Extra Point')&(row.ExPointResult == 'made')): # Extra Point seals W
            row.HomeTeamScore = row.HomeTeamScore+1
        elif (row.TwoPointConv == 'Success'):
            row.HomeTeamScore = row.HomeTeamScore+2 # 2-pt conversion successful to win game
    return row


# update the scores using apply(function)
d_sp = df_sp[((df_sp['Turnover'] == 1)&(df_sp['Touchdown'] == 1))&(df_sp['Safety'] == 1)].apply(update_def_score, axis = 1)
o_sp = df_sp[(df_sp['Turnover'] == 0) & (df_sp['Safety'] == 0)].apply(update_score, axis = 1)
d_sp = d_sp.append(o_sp)


# <body>
#     <p>Now the logic is complete and we've successfully updated scores for the rows containing the last instances of the Game ID for the games that do not have a PlayType = End of Game. Each record in the DataFrame containing the game id's for plays that had an ending in which there was a score is now updated with the actualy final score. Now that we know each record is correct, we can safely append it to the Final_Results DataFrame.</p>
# </body>

# In[ ]:


# append to Final Results DF
Final_Results = Final_Results.append(d_sp)


# <body>
#     <p>Moving onto the plays where sp = 0, which indicates that the final record for the game was not a scoring play.</p>
# </body>

# In[ ]:


# the GameID's that are not already in Final Results - because we already have the final scores for those games
df_scores = df_scores[~df_scores.GameID.isin(Final_Results.GameID)]

#remove dups
df_scores = df_scores.drop_duplicates(subset=['GameID'], keep='last')


# append the Non-Scoring endings to the Final Results
Final_Results = Final_Results.append(df_scores, sort = True)


# <h3>Checking The Multiplicity and Adding Parameters</h3>

# In[ ]:


# remove the duplicates
Final_Results = Final_Results.drop_duplicates(subset=['GameID'], keep='last')
print('Total Unique Games in the data:             ',len(df.GameID.unique()))
print('Total Games in the Final Results DF:        ',len(Final_Results))
print('Total Unique Games in the Final Results DF: ',len(Final_Results.GameID.unique()))



# <body>
#     <p>Now that we have our final scores for the Home and Away teams, we can remove the columns that we will no longer need, to further clean things up.</p>
#     <p>Secondly, I want to add a column that contains the winning and losing teams for simple aggregation and comparison purposes, followed by computing the absolute difference between the scores.</p>
#     </body>

# In[ ]:


# drop the listed columns
not_needed = ['PosTeamScore', 'DefTeamScore','posteam','DefensiveTeam','qtr','PlayType','sp','Touchdown','FieldGoalResult','ExPointResult','TwoPointConv','Turnover','Safety','TimeSecs']
Final_Results = Final_Results.drop(columns=not_needed)

# winning team
Final_Results['WinningTeam'] = Final_Results.apply(lambda x: x.HomeTeam if x.HomeTeamScore > x.AwayTeamScore else x.AwayTeam, axis=1)

# point differential between the winning and losing teams
Final_Results['PointSpread'] = Final_Results['HomeTeamScore'] - Final_Results['AwayTeamScore']
Final_Results['PointSpread'] = Final_Results['PointSpread']

# taking a look:
Final_Results.head(5)


# <body>
#     <p>It looks as if the unique counts line up now!</p>
# </body>

# <h1>Graphical Representation</h1>

# In[ ]:


WinningTeam = Final_Results['WinningTeam'].value_counts()
ax = WinningTeam.plot.bar(figsize=(22,10),rot=0,)
ax.set_title("Each Team's Number of Games Won", fontsize=24,)
ax.set_xlabel("Team", fontsize=18)
ax.set_ylabel("# of Wins", fontsize=14)
ax.set_alpha(0.8)

# set individual bar lables using above list
for i in ax.patches:
    # get_x: width; get_height: verticle
    ax.text(i.get_x()+.02, i.get_height()+1, str(round((i.get_height()), 2)), fontsize=10, color='black',rotation=0)


# <p>**As a Browns Fan:** I don't know why I'm doing this...</p>
# <h2>Average Score Differential</h2>

# In[ ]:


result_df = Final_Results.groupby('HomeTeam')['PointSpread'].mean()
ax = result_df.plot.bar(figsize=(22,10),rot=0,color='orange', width=1)
ax.set_title("Average Point Differentials by Team", fontsize=24)
ax.set_xlabel("Team", fontsize=18)
ax.set_ylabel("Average Point Differential", fontsize=14)
ax.set_alpha(0.6)


# <p>That's it for now, but after the Brownies beat Pit in Game 1 I'm sure I'll be back for add on to this!</p>

# In[ ]:




