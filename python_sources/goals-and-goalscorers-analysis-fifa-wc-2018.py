#!/usr/bin/env python
# coding: utf-8

# <font size="4">Introduction</font><br>
# <br>
# <b>2018 FIFA World Cup</b> in Russia was the biggest sport event held in that particular year. Football once again proved to be the king of sports as millions around the world were waiting for highly anticipated 21st edition of the FIFA World Cup to kick off.<br>
# <br>
# And the world did truly get what it was waiting for.<br>
# <br>
# The competition was full of wonderful goals and great performances, both team and individual. Benjamin Pavard got a well-deserved Goal of the tournament award for a stunning outside-the-box right foot volley against Argentina in Round of 16. In this kernel, I will make an analysis of the goals scored in the competition, as well as the analysis of the goalscorers.

# <font size="4">Goals anaylsis</font>

# Import statements

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the dataset and have a glance at it.

# In[ ]:


goals = pd.read_csv("/kaggle/input/fifa-world-cup-2018-goals/FifaWorldCup2018Goals.csv")
goals.head()


# Check are there any missing values.

# In[ ]:


goals.isnull().values.any()


# Now, we will check how many matches are there in the dataset. FIFA World Cup was expanded from 24 teams to 32 teams in 1998. The group stage consists of 8 groups with 4 teams and the 2 best teams advance to the round of 16. This format consists of 64 matches (including a 3rd place match).

# In[ ]:


matches = goals[['Home', 'Away', 'Stage']].drop_duplicates()
matches


# As we can see, the data shows us the result of 63. This means that there was 1 scoreless match and actually that match was Denmark-France match from the group C. In comparison, the previous FIFA World Cup from 2014 had 7 scoreless matches.<br>
# <br>
# Lets have a look at the number of goals score by every county.

# In[ ]:


total_goals_scored_by_team = goals['ScoringTeam'].value_counts()

plt.figure(figsize=(12, 6))
sns.countplot(x='ScoringTeam', data=goals, order=total_goals_scored_by_team.index)
plt.xticks(rotation=-60)
plt.xlabel('Team')
plt.ylabel('Goals scored')
print("Goals scored by every country bar plot:")


# The best scoring teams are: Belgium (16), Croatia (14), France (14), England (12) and Russia (11). Then there is a 3 goals gap from the next best scoring team which is Brazil. Every team scored at least 2 goals.<br>
# <br>
# In the world of football, Gary Lineker's quote about Germany is very well known:<br>
# <center><i>'Football is a simple game. 22 men chase the ball for 90 minutes and at the end, Germans always win.'</i></center><br>
# But yeah... Not this time.<br>
# Current world champions at that time were a huge disapointement and got eliminated in the group stage, something that never happened to them (before this they got eliminated only once at the begining of tournament, in 1938, but at that World cup there were no group matches played as the tournament started directly with the round of 16).<br>
# <br>
# Now, we are going to calculate the number of matches played for every team.

# In[ ]:


home_teams = matches['Home'].value_counts()
away_teams = matches['Away'].value_counts()
# During group stage every team needs to be both home side and away side at least once - we can just simple '+' for addition
total_matches_by_team = home_teams + away_teams

# Denmark and France played the only scoreless match so we will add it manually
total_matches_by_team['France'] += 1
total_matches_by_team['Denmark'] += 1
total_matches_by_team = pd.DataFrame(total_matches_by_team, columns=['MatchPlayed'])
total_matches_by_team.head()


# Based on the number of matches played, we will provide a category value of the reached stage for each team. That value is going to be used in one future plot.

# In[ ]:


def get_stage(row):
    if row['MatchPlayed'] == 3:
        return 'Group'
    if row['MatchPlayed'] == 4:
        return 'R16'
    if row['MatchPlayed'] == 5:
        return 'QF'
    
    return 'FinalFour'

total_matches_by_team['ReachedStage'] = total_matches_by_team.apply(lambda row: get_stage(row), axis=1)
total_matches_by_team.head()


# As the next step, we will calculate the average goals scored per match value and make a bar plot.

# In[ ]:


average_scored_goals_per_match = total_goals_scored_by_team / total_matches_by_team['MatchPlayed']
average_scored_goals_per_match.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
plt.xticks(rotation=-60)
plt.xlabel('Team')
plt.ylabel('Average goals scored')
print("Average goals scored per match for each country bar plot:")


# Values range from 0,67 to 2,28. Best scoring teams per match are: Belgium, Russia, Croatia and France. It is interesing to see Tunisia as the 7th best scoring team even though they finished the tournament in the group stage.<br>
# <br>
# Lets calculate conceded goals as well.

# In[ ]:


goals_conceded_by_home_team = goals[goals['Home'] != goals['ScoringTeam']]['Home'].value_counts()
goals_conceded_by_away_team = goals[goals['Away'] != goals['ScoringTeam']]['Away'].value_counts()
# There are some teams that did not concede a goal as an away team so we need to use add()
total_goals_conceded_by_team = goals_conceded_by_home_team.add(goals_conceded_by_away_team, fill_value=0)
total_goals_conceded_by_team.sort_values(ascending=False).head()


# And average conceded goals per match to see which teams had the best defense...

# In[ ]:


average_goals_conceded_per_match = total_goals_conceded_by_team / total_matches_by_team['MatchPlayed']
average_goals_conceded_per_match.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
plt.xticks(rotation=-60)
plt.xlabel('Team')
plt.ylabel('Average goals conceded')
print("Average goals conceded per match for each country bar plot:")


# Panama was, by far, the worst defensive team in the tournament.<br>
# We can see that none of the semifinal teams (France, Croatia, Belgium and England) is in top 5 defensive teams (France is best and ranked just 8th).<br>
# Another famous quote, by this one widely used across multiple sports, is:<br>
# <center><i>Offense wins games. Defense wins championships.</i></center>
# Well... Not on this World cup as we can see.<br>
# <br>
# Lets combine average scored/conceded goals and add the data about the reached stage for each team into one dataframe.

# In[ ]:


average_goals = pd.concat([average_scored_goals_per_match, average_goals_conceded_per_match], axis=1)
average_goals.columns = ['GoalsScored', 'GoalsConceded']

stage_and_average_goals = average_goals.merge(total_matches_by_team, left_index=True, right_index=True)
stage_and_average_goals.head()


# Plot the average goals values and consider the reached stage.

# In[ ]:


sns.lmplot(x='GoalsScored', y='GoalsConceded', data=stage_and_average_goals,
           fit_reg=True,
           hue='ReachedStage', hue_order=['Group','R16','QF','FinalFour'])

# Mark particular points
# Note: These are not the real average goals values.
# The annotation values are put in order to place the label on a good spot
plt.annotate('Panama', (0.7, 3.45))
plt.annotate('Senegal', (1.3, 1.15))
plt.annotate('Tunisia', (1.55, 2.8))
plt.annotate('Denmark', (0.75, 0.3))
plt.annotate('Russia', (2.1, 1.5))
plt.annotate('Belgium', (2.15, 0.65))


# Football logic should follow the rule where better teams score more and concede less (are closer to the bottom right corner) while worse teams score less and concede more (are closer to the upper left corner).<br>
# If we take a look at the graph we can clearly see that is true as the regression lines move from upper-left position to the bottom-right position. Futhermore, a look at the groups of teams that reached the same stage and comparison of groups shows that teams that reached:<br>
# - round of 16 had very similar defense as teams which got eliminated in the group stage, but much better offense<br>
# - quarterfinals have siminar offense as teams from round of 16, but better defense<br>
# - final four actually had a bit worse defense than the teams from quarterfinals, but better offense<br>
# <br>
# I guess the quote should be:<br>
# <center><i>Offense wins round of 16. Defense wins quarterfinals. A bit of luck wins championships.</i></center>

# Looking at numbers and position of teams in graph we can see that Panama was clearly the worst team in the tournament while Belgium (finished 3rd) was the best. It is noticeable that there are some big outliers:<br>
# - Tunisia is much closer to the round of 16 group than the group stage group<br>
# - Senegal is in the middle of round of 16 group, but it was eliminated in the group stage*<br>
# - Denmark is closer to the group stage group<br>
# - Russia is not only close to the final four group, but actually went over to the better scoring side of that group<br>
# 
# Clearly we can see that some teams tournament results were far better than the performance (considering the goal statistics) like Denmark and some deserved far better result like Senegal and Russia.<br>
# <br>
# *Senegal definitely had a performance which should have deserved them the round of 16, but they were unfortunately eliminated in the group stage by the number of yellow cards given to them since Japan had the same points, goals scored, goals conceded and their matchup finished as a draw. This was the first time in history that yellow cards decided the team that goes through.<br>
# <br>
# Now, lets take a look at the number of goals scored by each stage/round. All knockout matches are combined into a signle group in order to have the same amount of matches (16) in every group.

# In[ ]:


goals_per_stage = goals['Stage'].value_counts()

round16 = goals_per_stage['Round16']
quarterfinals = goals_per_stage['Quarterfinals']
semifinals = goals_per_stage['Semifinals']
third_place = goals_per_stage['ThirdPlace']
final = goals_per_stage['Final']

goals_per_stage = goals_per_stage.drop(labels=['Round16', 'Quarterfinals', 'Semifinals', 'ThirdPlace', 'Final'])
goals_per_stage['Knockout'] = round16 + quarterfinals + semifinals + third_place + final
goals_per_stage.sort_values().plot(kind='barh')
plt.xlabel('Average goals scored')
plt.ylabel('Stage')


# As it turns out, group stage matchday 2 and knockout stage had much more goals. While the group matchday 2 is not a surprise considering all the teams are still very motivated (unlike matchday 3) because the winners from matchday 1 want to keep the winning strike and losers have to go 'all-in' to pass the group stage, the fact that the knockout stage had the most goals was surprising to me. There were 47 goals in total in the knockout stage, while in 2014 FIFA World cup there were 'just' 35.<br>
# <br>
# Lets take a look at the goal type distribution.

# In[ ]:


sns.countplot(x='Type', data=goals, order=goals['Type'].value_counts().index)


# Of course, most of the goals were scored from inside the box. It is interesting to see a high value for penalties - 22. Compraing to the last World cup 2014 Brazil were the value was 12, there is a 83.3% increase. Obviously, Video Assistant Referee (VAR) made a huge impact to the game.<br>
# Usually, in football, statistics say that there is one penalty every four matches, but in this World cup that number increased to one in every 3 matches.<br>
# <br>
# Lets take a look at which teams scored the most goals from standard situations (situations which can be practiced during training). Standard situations in football are penalties, freekicks and corner kicks. This dataset does not provide information about which goals were scored after corner kick, but anyone who watched most of the tournament can probably remember that a good portion of penalties were given a result of corner kicks situations.

# In[ ]:


goals[(goals['Type'] == 'Penalty') | (goals['Type'] =='Freekick')]['ScoringTeam'].value_counts().plot(kind='bar')
plt.xticks(rotation=-60)
plt.xlabel('Team')
plt.ylabel('Standard situation goals')


# As we can see England tops that list. Lets explore England's goal distribution by type.

# In[ ]:


goals[goals['ScoringTeam'] == 'England']['Type'].value_counts().plot(kind='pie')


# Not only that they scored the most goals from standard situations, but these goals make 1/3 of their goals which obviously shows that they were particulary dangerous from these situations. Actually, if we take a look at the match highlights we can see that they scored 5 more goals after corner kicks and freekicks (which are marked as inside goals in the dataset) which increases the total sum of goals after standard situations to 9 or stunning 3/4 of their goals. Nice job, Gareth Southgate!<br>
# <br>
# <font size='4'>Goalscorers analysis</font><br>
# <br>
# First, we will take all goals except own goals.

# In[ ]:


regular_goals = goals[goals['Type'] != 'Own']
regular_goals.head(10)


# Now, lets take a look at the number of scorers which scored particular amount of goals.

# In[ ]:


regular_goals['Scorer'].value_counts().value_counts().plot(kind='barh')
plt.xlabel('Number of goalscorers')
plt.ylabel('Goals scored')


# Of course, most of the scorers scored just one goal.<br>
# <br>
# Who were the best scorers?

# In[ ]:


best_scorers = regular_goals['Scorer'].value_counts()
best_scorers.head(15)


# As we can see, Harry Kane was the best goalscorer which provided him the Golden Boot award.<br>
# <br>
# In my personal opinion, the basic goal count is not good and fair condition for this award. It is not the same if someone scores 3 penalties in group stage while somebody else scores 3 outside of the box goals in the knockout stage. So I decided to make a scoring system of my own with the following rules:<br>
# - goal worth by type  
#                       - penalty                  0.75
#                       - inside the box           1.00
#                       - outside the box          1.50
#                       - freekick                 2.00
# - goal worth by stage 
#                       - group stage              1.00
#                       - round of 16              1.20
#                       - quarterfinals            1.50
#                       - semifinals and 3rd place 1.75
#                       - final                    2.00
# - goal worth by result after the goal was scored
#                       - losing goal              1.00
#                       - equalizer                1.20
#                       - final equalizer          1.50
#                       - leading goal             1.50
#                       - final leading/winner     2.00
# <br>
# Goal worth = 'type' \* 'stage' \* 'result'

# In[ ]:


def calculate_goal(row):
    return type_coefficient(row['Type']) * stage_coefficient(row['Stage']) * result_coefficient(row)
    
def type_coefficient(goal_type):
    if goal_type == 'Penalty':
        return 0.75
    if goal_type == 'Inside':
        return 1.0
    if goal_type == 'Outside':
        return 1.5
    if goal_type == 'Freekick':
        return 2.0
    
def stage_coefficient(stage):
    if stage.startswith('Group'):
        return 1.0
    if stage == 'Round16':
        return 1.25
    if stage == 'Quarterfinals':
        return 1.5
    if stage == 'Semifinals' or stage == 'ThirdPlace':
        return 1.75
    if stage == 'Final':
        return 2.0
    
def result_coefficient(row):
    result = row['Result'].split('-')
    
    if result[0] == result[1]:
            # Final equalizer
            if row['Result'] == row['FinalResult']:
                return 1.5
            
            # Just equalizer
            return 1.2
        
    if row['ScoringTeam'] == row['Home']:
        # Goal for home team lead
        if result[0] > result[1]:
            # Winning goal home team
            if row['Result'] == row['FinalResult']:
                return 2
            
            # Just lead
            return 1.5
        
        return 1
    
    # Goal for away team lead
    if result[0] < result[1]:
        # Winning goal away team
        if row['Result'] == row['FinalResult']:
            return 2
            
        # Just lead
        return 1.5
        
    return 1


# Apply the rules and calculate each goal value.

# In[ ]:


regular_goals['Value'] = regular_goals.apply(lambda row: calculate_goal(row), axis=1)
regular_goals.head()


# Lets see who are the best goalscorers by this system.

# In[ ]:


final_score = regular_goals[['Scorer', 'Value']].groupby(['Scorer']).sum().reset_index()
final_score.columns = ['Scorer', 'GoalsValue']
final_score.sort_values('GoalsValue', ascending=False).head(10)


# Congratulations Kylian Mbappe on being the Most Valuable Goalscorer of the tournament!<br>
# <br>
# Lets combine the results with the real goal counts for each player.

# In[ ]:


real_goals = pd.DataFrame(regular_goals['Scorer'].value_counts())
real_goals.columns = ['RealGoals']

final_results = final_score.merge(real_goals, left_on='Scorer', right_index=True)
final_results


# For the end lets plot the real goals count and result values on a graph.

# In[ ]:


sns.lmplot(x='RealGoals', y='GoalsValue', data=final_results)

# Mark particular points
# Note: These are not the real goals count and result values.
# The annotation values are put in order to place the label on a good spot
plt.annotate('Trippier', (1.0, 5.6))
plt.annotate('Jedinak', (1.7, 1.5))
plt.annotate('Mandzukic', (2.5, 7.7))
plt.annotate('Mbappe', (3.7, 10.6))
plt.annotate('Kane', (5.8, 8.0))


# Harry Kane did win the Golden Boot, but he is the biggest overachiever in the Golden Boot award race. His bad result in the new scoring system was affected by the facts that he scored 3 penalties and did not score any goal from round of 16 onwards. Another big overachiever is Mile Jedinak who scored 2 penalties (his only goals on the tournament).<br>
# Some players deserve much more praise for their efforts like Mbappe and Mandzukic. It is interesting to see Tripper's huge result with scoring only one goal (but that goal was a semifinal freekick leading goal).<br>
# <br>
# <br>
# This is my first public kernel. I hope you like it!
